
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from mmdet.models.necks import FPN

from ..builder import ROTATED_NECKS


# ============================================================
# 1. LayerNorm2d
# ============================================================

class LayerNormFunction(torch.autograd.Function):
    """Channel-wise layer normalization for 2D feature maps."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps

        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)

        y = (x - mu) / torch.sqrt(var + eps)

        ctx.save_for_backward(y, var, weight)

        y = weight.view(1, -1, 1, 1) * y
        y = y + bias.view(1, -1, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors

        g = grad_output * weight.view(1, -1, 1, 1)

        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)

        gx = (
            1.0 / torch.sqrt(var + eps)
            * (g - y * mean_gy - mean_g)
        )

        grad_weight = (grad_output * y).sum(
            dim=(0, 2, 3)
        )

        grad_bias = grad_output.sum(
            dim=(0, 2, 3)
        )

        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    """Layer normalization across channels at each spatial location."""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()

        self.weight = nn.Parameter(
            torch.ones(channels)
        )

        self.bias = nn.Parameter(
            torch.zeros(channels)
        )

        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(
            x,
            self.weight,
            self.bias,
            self.eps
        )


# ============================================================
# 2. MSPA: Multi-Scale Spatial Attention
# ============================================================

class MSPA(nn.Module):
    """
    Multi-scale spatial feature extraction and attention.

    Directional convolutions followed by dilated convolutions
    at dilation rates 1, 2, and 3.
    """

    def __init__(self, channels, N=3):
        super(MSPA, self).__init__()

        self.N = N

        self.conv_1xN = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, N),
            padding=(0, N // 2),
            bias=False
        )

        self.conv_Nx1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(N, 1),
            padding=(N // 2, 0),
            bias=False
        )

        self.conv_d1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False
        )

        self.conv_d2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False
        )

        self.conv_d3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            bias=False
        )

        self.fusion = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        # Directional spatial feature extraction
        x = self.conv_1xN(x)
        x = self.conv_Nx1(x)

        # Multi-scale dilated feature extraction
        x_d1 = self.conv_d1(x)
        x_d2 = self.conv_d2(x)
        x_d3 = self.conv_d3(x)

        # Concatenate multi-scale features
        x_cat = torch.cat(
            [x_d1, x_d2, x_d3],
            dim=1
        )

        # Generate spatial-channel attention map
        attn = self.fusion(x_cat)
        attn = torch.sigmoid(attn)

        return attn


# ============================================================
# 3. FBR: Frequency Band Refinement
# ============================================================

class FBR(nn.Module):
    """
    Frequency-band refinement using FFT.

    Splits the Fourier magnitude into low, middle, and high
    radial frequency bands and independently gates each band.

    The Fourier phase is preserved during reconstruction.
    """

    def __init__(
        self,
        channels,
        low_ratio=0.25,
        high_ratio=0.60
    ):
        super(FBR, self).__init__()

        if not (0.0 < low_ratio < high_ratio < 1.0):
            raise ValueError(
                "Require 0 < low_ratio < high_ratio < 1."
            )

        self.channels = channels
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

        def make_gate():
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=1,
                    bias=True
                ),
                nn.LeakyReLU(
                    negative_slope=0.1,
                    inplace=True
                ),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=1,
                    bias=True
                ),
                nn.Sigmoid()
            )

        self.low_gate = make_gate()
        self.mid_gate = make_gate()
        self.high_gate = make_gate()

    def _get_radial_masks(
        self, H, W, device, dtype
    ):

        y = torch.arange(
            H,
            device=device,
            dtype=dtype
        ) - (H // 2)

        x = torch.arange(
            W,
            device=device,
            dtype=dtype
        ) - (W // 2)

        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij"
        )

        radius = torch.sqrt(
            xx.pow(2) + yy.pow(2)
        )

        max_radius = radius.max().clamp_min(1.0)

        normalized_radius = radius / max_radius

        low_mask = (
            normalized_radius <= self.low_ratio
        )

        mid_mask = (
            (normalized_radius > self.low_ratio)
            & (normalized_radius <= self.high_ratio)
        )

        high_mask = (
            normalized_radius > self.high_ratio
        )

        return (
            low_mask.to(dtype=dtype),
            mid_mask.to(dtype=dtype),
            high_mask.to(dtype=dtype)
        )

    def forward(self, x):

        # FFT operations are performed in float32 for
        # numerical stability and mixed-precision compatibility.
        input_dtype = x.dtype
        x_float = x.float()

        B, C, H, W = x_float.shape

        # 2D Fourier transform and frequency centering
        X = torch.fft.fft2(
            x_float,
            dim=(-2, -1),
            norm="ortho"
        )

        X_shifted = torch.fft.fftshift(
            X,
            dim=(-2, -1)
        )

        magnitude = torch.abs(X_shifted)
        phase = torch.angle(X_shifted)

        # Construct radial frequency masks
        low_mask, mid_mask, high_mask = (
            self._get_radial_masks(
                H,
                W,
                x.device,
                magnitude.dtype
            )
        )

        low_mask = low_mask.view(1, 1, H, W)
        mid_mask = mid_mask.view(1, 1, H, W)
        high_mask = high_mask.view(1, 1, H, W)

        # Separate the frequency bands
        M_low = magnitude * low_mask
        M_mid = magnitude * mid_mask
        M_high = magnitude * high_mask

        # Independent band-wise gates
        G_low = self.low_gate(M_low)
        G_mid = self.mid_gate(M_mid)
        G_high = self.high_gate(M_high)

        # Refine each frequency band
        M_low_ref = M_low * G_low
        M_mid_ref = M_mid * G_mid
        M_high_ref = M_high * G_high

        magnitude_refined = (
            M_low_ref
            + M_mid_ref
            + M_high_ref
        )

        # Reconstruct complex spectrum using original phase
        X_refined_shifted = torch.polar(
            magnitude_refined,
            phase
        )

        X_refined = torch.fft.ifftshift(
            X_refined_shifted,
            dim=(-2, -1)
        )

        # Inverse FFT
        out = torch.fft.ifft2(
            X_refined,
            dim=(-2, -1),
            norm="ortho"
        )

        return out.real.to(dtype=input_dtype)


# ============================================================
# 4. CGB: Contextual Gating Block
# ============================================================

class CGB(nn.Module):
    """
    Contextual Gating Block.

    Combines a feature-preserving branch and a depthwise
    spatial suppression branch using complementary gates.
    """

    def __init__(self, channels):
        super(CGB, self).__init__()

        # Feature-preserving branch
        self.preserve = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        # Depthwise spatial suppression branch
        self.suppress = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

        # Spatial contextual gate
        self.gate = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        Xp = self.preserve(x)
        Xs = self.suppress(x)

        Gp = torch.sigmoid(
            self.gate(x)
        )

        Gs = 1.0 - Gp

        return (
            Gp * Xp
            + Gs * Xs
        )


# ============================================================
# 5. GFCRNet: FPN + LayerNorm2d + MSPA + FBR + CGB
# ============================================================

@ROTATED_NECKS.register_module()
class GFCRNet(FPN):
    """
    Frequency- and context-refined Feature Pyramid Network.

    Architecture:
        FPN
        -> LayerNorm2d
        -> MSPA
        -> MSPA attention multiplication
        -> FBR
        -> CGB
        -> Residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        low_ratio=0.25,
        high_ratio=0.60,
        mspa_kernel_size=3,
        **kwargs
    ):

        super(GFCRNet, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs
        )

        # One independent refinement pipeline per FPN level
        self.norms = nn.ModuleList()
        self.mspa = nn.ModuleList()
        self.fbr = nn.ModuleList()
        self.cgb = nn.ModuleList()

        for _ in range(num_outs):

            self.norms.append(
                LayerNorm2d(out_channels)
            )

            self.mspa.append(
                MSPA(
                    out_channels,
                    N=mspa_kernel_size
                )
            )

            self.fbr.append(
                FBR(
                    out_channels,
                    low_ratio=low_ratio,
                    high_ratio=high_ratio
                )
            )

            self.cgb.append(
                CGB(out_channels)
            )

    @auto_fp16()
    def forward(self, inputs):

        # Standard FPN feature pyramid
        outs = super(GFCRNet, self).forward(inputs)

        refined_feats = []

        for i, feat in enumerate(outs):

            identity = feat

            # 1. Channel-wise normalization
            x = self.norms[i](feat)

            # 2. Multi-scale spatial attention
            attn = self.mspa[i](x)
            x = x * attn

            # 3. Frequency-band refinement
            x = self.fbr[i](x)

            # 4. Contextual gating
            x = self.cgb[i](x)

            # 5. Residual fusion with original FPN feature
            x = identity + x

            refined_feats.append(x)

        return tuple(refined_feats)
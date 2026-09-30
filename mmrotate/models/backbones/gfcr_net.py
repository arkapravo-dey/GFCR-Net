import torch
import torch.nn as nn

from mmcv.runner import auto_fp16

from ..builder import ROTATED_BACKBONES


# ============================================================
# MSPA: Multi-Scale Spatial Attention
# ============================================================

class MSPA(nn.Module):
    """
    Multi-Scale Spatial Attention.

    Spatial attention is generated using:
        1. Asymmetric convolution
        2. Dilated convolutions with dilation 1, 2 and 3
        3. Feature fusion
    """

    def __init__(self, channels, N=3):

        super(MSPA, self).__init__()

        self.N = N

        # ----------------------------------------------------
        # Asymmetric convolution
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Multi-scale dilated convolutions
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        self.fusion = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        # Asymmetric spatial processing
        x = self.conv_1xN(x)
        x = self.conv_Nx1(x)

        # Multi-scale processing
        x_d1 = self.conv_d1(x)
        x_d2 = self.conv_d2(x)
        x_d3 = self.conv_d3(x)

        # Concatenate multi-scale features
        x_cat = torch.cat(
            [x_d1, x_d2, x_d3],
            dim=1
        )

        # Generate attention map
        attn = self.fusion(x_cat)
        attn = torch.sigmoid(attn)

        return attn


# ============================================================
# FBR: Frequency Band Refinement
# ============================================================

class FBR(nn.Module):
    """
    Frequency Band Refinement.

    The Fourier magnitude is divided into:
        - Low frequency
        - Mid frequency
        - High frequency

    Each frequency band is independently gated.

    The original Fourier phase is preserved.
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

        # ----------------------------------------------------
        # Frequency gate
        # ----------------------------------------------------

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

    # ========================================================
    # Radial frequency masks
    # ========================================================

    def _get_radial_masks(
        self,
        H,
        W,
        device,
        dtype
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

        normalized_radius = (
            radius / max_radius
        )

        # ----------------------------------------------------
        # Low frequency
        # ----------------------------------------------------

        low_mask = (
            normalized_radius <= self.low_ratio
        )

        # ----------------------------------------------------
        # Mid frequency
        # ----------------------------------------------------

        mid_mask = (
            (normalized_radius > self.low_ratio)
            &
            (normalized_radius <= self.high_ratio)
        )

        # ----------------------------------------------------
        # High frequency
        # ----------------------------------------------------

        high_mask = (
            normalized_radius > self.high_ratio
        )

        return (
            low_mask.to(dtype=dtype),
            mid_mask.to(dtype=dtype),
            high_mask.to(dtype=dtype)
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        input_dtype = x.dtype

        # FFT is performed in FP32
        x_float = x.float()

        B, C, H, W = x_float.shape

        # ----------------------------------------------------
        # Fourier transform
        # ----------------------------------------------------

        X = torch.fft.fft2(
            x_float,
            dim=(-2, -1),
            norm="ortho"
        )

        # Shift zero frequency to center
        X_shifted = torch.fft.fftshift(
            X,
            dim=(-2, -1)
        )

        # Magnitude and phase
        magnitude = torch.abs(X_shifted)
        phase = torch.angle(X_shifted)

        # ----------------------------------------------------
        # Frequency masks
        # ----------------------------------------------------

        (
            low_mask,
            mid_mask,
            high_mask
        ) = self._get_radial_masks(
            H,
            W,
            x.device,
            magnitude.dtype
        )

        low_mask = low_mask.view(
            1,
            1,
            H,
            W
        )

        mid_mask = mid_mask.view(
            1,
            1,
            H,
            W
        )

        high_mask = high_mask.view(
            1,
            1,
            H,
            W
        )

        # ----------------------------------------------------
        # Split magnitude
        # ----------------------------------------------------

        M_low = magnitude * low_mask
        M_mid = magnitude * mid_mask
        M_high = magnitude * high_mask

        # ----------------------------------------------------
        # Learnable frequency gates
        # ----------------------------------------------------

        G_low = self.low_gate(M_low)
        G_mid = self.mid_gate(M_mid)
        G_high = self.high_gate(M_high)

        # ----------------------------------------------------
        # Refine frequency bands
        # ----------------------------------------------------

        M_low_ref = M_low * G_low
        M_mid_ref = M_mid * G_mid
        M_high_ref = M_high * G_high

        magnitude_refined = (
            M_low_ref
            + M_mid_ref
            + M_high_ref
        )

        # ----------------------------------------------------
        # Reconstruct using original phase
        # ----------------------------------------------------

        X_refined_shifted = torch.polar(
            magnitude_refined,
            phase
        )

        # Reverse FFT shift
        X_refined = torch.fft.ifftshift(
            X_refined_shifted,
            dim=(-2, -1)
        )

        # ----------------------------------------------------
        # Inverse FFT
        # ----------------------------------------------------

        out = torch.fft.ifft2(
            X_refined,
            dim=(-2, -1),
            norm="ortho"
        )

        return out.real.to(
            dtype=input_dtype
        )


# ============================================================
# CGB: Contextual Gating Block
# ============================================================

class CGB(nn.Module):
    """
    Contextual Gating Block.

    Preserve branch:
        1x1 convolution

    Suppress branch:
        Depthwise 3x3 convolution + ReLU

    The two branches are adaptively combined using
    a learned gate.
    """

    def __init__(self, channels):

        super(CGB, self).__init__()

        # ----------------------------------------------------
        # Preserve branch
        # ----------------------------------------------------

        self.preserve = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        # ----------------------------------------------------
        # Suppress branch
        # ----------------------------------------------------

        self.suppress = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Gate
        # ----------------------------------------------------

        self.gate = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        # Preserve branch
        Xp = self.preserve(x)

        # Suppress branch
        Xs = self.suppress(x)

        # Adaptive gating
        Gp = torch.sigmoid(
            self.gate(x)
        )

        Gs = 1.0 - Gp

        # Contextual fusion
        return (
            Gp * Xp
            +
            Gs * Xs
        )


# ============================================================
# GFCR BLOCK
# ============================================================

class GFCRBlock(nn.Module):
    """
    Basic GFCR feature extraction block.

        Input
          |
         MSPA
          |
         FBR
          |
         CGB
          |
       Residual
          |
        Output
    """

    def __init__(
        self,
        channels,
        low_ratio=0.25,
        high_ratio=0.60
    ):

        super(GFCRBlock, self).__init__()

        self.mspa = MSPA(
            channels=channels
        )

        self.fbr = FBR(
            channels=channels,
            low_ratio=low_ratio,
            high_ratio=high_ratio
        )

        self.cgb = CGB(
            channels=channels
        )

    def forward(self, x):

        identity = x

        # ----------------------------------------------------
        # MSPA
        # ----------------------------------------------------

        attn = self.mspa(x)

        x = x * attn

        # ----------------------------------------------------
        # FBR
        # ----------------------------------------------------

        x = self.fbr(x)

        # ----------------------------------------------------
        # CGB
        # ----------------------------------------------------

        x = self.cgb(x)

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        x = x + identity

        return x


# ============================================================
# GFCRNet BACKBONE
# ============================================================

@ROTATED_BACKBONES.register_module()
class GFCRNet(nn.Module):
    """
    GFCRNet backbone.

    Four sequential feature extraction stages.

    Architecture:

        Input
          |
        Stem
          |
        GFCR Block 1
          |
         C2
          |
      Downsample
          |
        GFCR Block 2
          |
         C3
          |
      Downsample
          |
        GFCR Block 3
          |
         C4
          |
      Downsample
          |
        GFCR Block 4
          |
         C5
          |
          ▼
         FPN
          |
          ▼
    Oriented R-CNN

    Each GFCR block:

        MSPA → FBR → CGB → Residual
    """

    def __init__(
        self,
        in_channels=3,
        channels=(64, 128, 256, 512),
        low_ratio=0.25,
        high_ratio=0.60,
        out_indices=(0, 1, 2, 3)
    ):

        super(GFCRNet, self).__init__()

        self.out_indices = out_indices

        # ====================================================
        # STEM
        # ====================================================

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True
            ),

            nn.GELU()
        )

        # ====================================================
        # STAGE 1
        # ====================================================

        self.stage1 = GFCRBlock(
            channels=channels[0],
            low_ratio=low_ratio,
            high_ratio=high_ratio
        )

        # ====================================================
        # DOWN SAMPLE 1
        # ====================================================

        self.down1 = nn.Sequential(

            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=2,
                stride=2,
                bias=True
            ),

            nn.GELU()
        )

        # ====================================================
        # STAGE 2
        # ====================================================

        self.stage2 = GFCRBlock(
            channels=channels[1],
            low_ratio=low_ratio,
            high_ratio=high_ratio
        )

        # ====================================================
        # DOWN SAMPLE 2
        # ====================================================

        self.down2 = nn.Sequential(

            nn.Conv2d(
                channels[1],
                channels[2],
                kernel_size=2,
                stride=2,
                bias=True
            ),

            nn.GELU()
        )

        # ====================================================
        # STAGE 3
        # ====================================================

        self.stage3 = GFCRBlock(
            channels=channels[2],
            low_ratio=low_ratio,
            high_ratio=high_ratio
        )

        # ====================================================
        # DOWN SAMPLE 3
        # ====================================================

        self.down3 = nn.Sequential(

            nn.Conv2d(
                channels[2],
                channels[3],
                kernel_size=2,
                stride=2,
                bias=True
            ),

            nn.GELU()
        )

        # ====================================================
        # STAGE 4
        # ====================================================

        self.stage4 = GFCRBlock(
            channels=channels[3],
            low_ratio=low_ratio,
            high_ratio=high_ratio
        )

    # ========================================================
    # FORWARD
    # ========================================================

    @auto_fp16()
    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        x = self.stem(x)

        # ----------------------------------------------------
        # Stage 1 → C2
        # ----------------------------------------------------

        x = self.stage1(x)

        C2 = x

        # ----------------------------------------------------
        # Downsample
        # ----------------------------------------------------

        x = self.down1(x)

        # ----------------------------------------------------
        # Stage 2 → C3
        # ----------------------------------------------------

        x = self.stage2(x)

        C3 = x

        # ----------------------------------------------------
        # Downsample
        # ----------------------------------------------------

        x = self.down2(x)

        # ----------------------------------------------------
        # Stage 3 → C4
        # ----------------------------------------------------

        x = self.stage3(x)

        C4 = x

        # ----------------------------------------------------
        # Downsample
        # ----------------------------------------------------

        x = self.down3(x)

        # ----------------------------------------------------
        # Stage 4 → C5
        # ----------------------------------------------------

        x = self.stage4(x)

        C5 = x

        # ----------------------------------------------------
        # Return multi-scale features
        # ----------------------------------------------------

        outputs = (
            C2,
            C3,
            C4,
            C5
        )

        return tuple(
            outputs[i]
            for i in self.out_indices
        )
import torch
import torch.nn as nn

from mmcv.runner import auto_fp16
from mmdet.models.necks import FPN

from ..builder import ROTATED_NECKS


# ============================================================
# FBR: Frequency Band Refinement
# ============================================================

class FBR(nn.Module):
    """
    Frequency Band Refinement.

    Splits the Fourier magnitude into:
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

    # --------------------------------------------------------
    # Radial frequency masks
    # --------------------------------------------------------

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

        low_mask = (
            normalized_radius <= self.low_ratio
        )

        mid_mask = (
            (normalized_radius > self.low_ratio)
            &
            (normalized_radius <= self.high_ratio)
        )

        high_mask = (
            normalized_radius > self.high_ratio
        )

        return (
            low_mask.to(dtype=dtype),
            mid_mask.to(dtype=dtype),
            high_mask.to(dtype=dtype)
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, x):

        input_dtype = x.dtype

        # FFT in FP32 for numerical stability
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
            1, 1, H, W
        )

        mid_mask = mid_mask.view(
            1, 1, H, W
        )

        high_mask = high_mask.view(
            1, 1, H, W
        )

        # ----------------------------------------------------
        # Split magnitude into frequency bands
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
        # Reconstruct using ORIGINAL phase
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
# FBR-FPN
# ============================================================

@ROTATED_NECKS.register_module()
class GFCRNet(FPN):
    """
    FBR applied ONLY to C2 and C3.

    Architecture:

        C2 ── FBR ──┐
        C3 ── FBR ──┤
        C4 ─────────┤── FPN
        C5 ─────────┘

    The FPN itself remains standard.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        low_ratio=0.25,
        high_ratio=0.60,
        **kwargs
    ):

        super(GFCRNet, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs
        )

        # ----------------------------------------------------
        # FBR only for C2 and C3
        #
        # ResNet-50:
        # C2 = 256 channels
        # C3 = 512 channels
        # ----------------------------------------------------

        self.fbr_c2 = FBR(
            channels=in_channels[0],
            low_ratio=low_ratio,
            high_ratio=high_ratio
        )

        self.fbr_c3 = FBR(
            channels=in_channels[1],
            low_ratio=low_ratio,
            high_ratio=high_ratio
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    @auto_fp16()
    def forward(self, inputs):

        assert len(inputs) >= 4, (
            "GFCRNet expects C2, C3, C4 and C5 "
            "from the ResNet backbone."
        )

        # ----------------------------------------------------
        # ResNet features
        # ----------------------------------------------------

        C2 = inputs[0]
        C3 = inputs[1]
        C4 = inputs[2]
        C5 = inputs[3]

        # ----------------------------------------------------
        # FBR ONLY on high-resolution features
        # ----------------------------------------------------

        C2 = self.fbr_c2(C2)

        C3 = self.fbr_c3(C3)

        # ----------------------------------------------------
        # C4 and C5 remain untouched
        # ----------------------------------------------------

        refined_inputs = (
            C2,
            C3,
            C4,
            C5
        )

        # ----------------------------------------------------
        # Standard FPN
        # ----------------------------------------------------

        outs = super(GFCRNet, self).forward(
            refined_inputs
        )

        return outs
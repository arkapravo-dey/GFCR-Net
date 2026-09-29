import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmdet.models.necks import FPN
from mmcv.cnn.bricks.transformer import MultiheadAttention

from ..builder import ROTATED_NECKS


class GradientGuidedBoundaryEnhancement(nn.Module):
    """Gradient-Guided Boundary Enhancement (GGBE) Module."""
    def __init__(self, channels):
        super(GradientGuidedBoundaryEnhancement, self).__init__()
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Compute discrete horizontal and vertical derivatives
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # Pad to keep spatial dimensions intact
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        
        # Channel concatenation and gating response calculation
        concat = torch.cat([x, grad_magnitude], dim=1)
        weights = self.weight_net(concat)
        
        return x * weights


class FrequencyGatedBoostingModule(nn.Module):
    """Frequency-Gated Boosting Module (FGBM) for calibration masks."""
    def __init__(self, out_channels):
        super(FrequencyGatedBoostingModule, self).__init__()
        self.fpn_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1),
            nn.Sigmoid()
        )
        self.excite_conv = ConvModule(
            out_channels, out_channels, 1, padding=0, act_cfg=None)
        self.high_boost_param = nn.Parameter(torch.zeros(out_channels, 1, 1))
        
    def forward(self, x):
        s_min = torch.min(x, dim=1, keepdim=True)[0]
        c_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        spatial_min = s_min * c_min
        
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq
        
        high_boost = self.fpn_gate(low_freq) + self.high_boost_param * (self.fpn_gate(high_freq))
        excitation = torch.sigmoid(self.excite_conv(spatial_min + high_boost))
        return torch.chunk(excitation, 2, dim=1)


class GlobalRelationalContextModeling(nn.Module):
    """Global Relational Context Modeling (GRCM) block."""
    def __init__(self, channels, apply_sigmoid=False):
        super(GlobalRelationalContextModeling, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.channel_fc = nn.Linear(channels, 1)
        self.atten_layer = MultiheadAttention(embed_dims=channels // 8, num_heads=4)
        
    def forward(self, feat):
        mask = torch.mean(feat, 1, keepdim=True)
        mask = mask + self.channel_fc(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        b, c, h, w = feat.size()
        flat = feat.view(b, c, -1)
        
        avg_pool = torch.mean(flat, 2).view(b, 8, c // 8)
        max_pool = torch.max(flat, 2)[0].view(b, 8, c // 8)
        
        if self.apply_sigmoid:
            avg_pool, max_pool = self.sigmoid(avg_pool), self.sigmoid(max_pool)
            
        attn_out = self.atten_layer(avg_pool, avg_pool, max_pool)
        desc = attn_out.view(b, c, 1, 1)
        return mask * desc


@ROTATED_NECKS.register_module()
class GFCRNet(FPN):
    """Gradient-Guided and Frequency-Gated Context Refinement Network (GFCR-Net)."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 **kwargs):
        super(GFCRNet, self).__init__(
            in_channels, out_channels, num_outs,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)

        self.mid_channels = out_channels // 2

        # 1. Gradient-Guided Boundary Enhancement (GGBE) Module
        self.ggbe = nn.ModuleList([
            GradientGuidedBoundaryEnhancement(out_channels) for _ in range(num_outs)
        ])

        # 2. Dual-Stream Contextual Refinement (DSCR) Blocks
        self.fgbm = nn.ModuleList([
            FrequencyGatedBoostingModule(out_channels) for _ in range(num_outs)
        ])
        
        self.encoder_conv = ConvModule(
            out_channels, out_channels, 3, padding=1, groups=out_channels, act_cfg=act_cfg)
        
        self.fusion_conv = ConvModule(
            out_channels, out_channels, 1, padding=0, act_cfg=act_cfg)
            
        self.peak_grcm = GlobalRelationalContextModeling(self.mid_channels, apply_sigmoid=False)
        self.avg_grcm = GlobalRelationalContextModeling(out_channels, apply_sigmoid=True)

    @auto_fp16()
    def forward(self, inputs):
        outs = super(GFCRNet, self).forward(inputs)
        
        refined_feats = []
        for i, feat in enumerate(outs):
            # Apply GGBE
            x_enh = self.ggbe[i](feat)
            
            # Dual-Stream Split Route Processing (DSCR)
            split_feat = self.encoder_conv(x_enh)
            x1, x2 = torch.split(split_feat, self.mid_channels, dim=1)
            
            # Apply FGBM
            a1, a2 = self.fgbm[i](x_enh)
            
            # Peak-driven path & complementary structural path fusion
            x1_hat = self.peak_grcm(x1 * (1 + a1))
            x2_hat = x2 * (1 + a2)
            
            fused = torch.cat([x2_hat, x1_hat], dim=1)
            compressed = self.fusion_conv(fused)
            
            # Scene-level context aggregation via Average-driven GRCM
            f_out = compressed + self.avg_grcm(compressed)
            
            refined_feats.append(f_out + feat)
            
        return tuple(refined_feats)
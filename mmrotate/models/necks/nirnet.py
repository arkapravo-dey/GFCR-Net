import torch

from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmdet.models.necks import FPN
from mmcv.cnn.bricks.transformer import MultiheadAttention

from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class NIRNet(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 **kwargs):
        super(NIRNet, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.encoder_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            groups=out_channels,
            inplace=False)
        self.fusion_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.excite_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.fcm_ip_channel_fc = torch.nn.Linear(out_channels // 2, 1)
        self.fcm_ip_channel_atten = MultiheadAttention(embed_dims=16, num_heads=8)
        self.fcm_sp_channel_atten = MultiheadAttention(embed_dims=16, num_heads=8)
        self.fcm_sp_channel_fc = torch.nn.Linear(out_channels // 2, 1)

    @auto_fp16()
    def forward(self, inputs):
        outs = super(NIRNet, self).forward(inputs)
        outs = self.dpic(outs)
        return tuple(outs)

    def dpic(self, feats):

        encoder_feats = []
        
        for feat in feats:
            split_feat = self.encoder_conv(feat)
            dw_feat, pw_feat = torch.split(split_feat, split_size_or_sections=128, dim=1)
            # noise prediction module
            weight1, weight2 = self.npm(feat)
            # noise incentive mechanism
            fcm_sp_feat = self.fcm_sp(pw_feat * (1 + weight1))
            fcm_ip_feat = self.fcm_ip(dw_feat * (1 + weight2))
            # concatenation and fusion
            fusion_feat = self.fusion_conv(torch.cat([fcm_ip_feat , fcm_sp_feat], dim=1))
            encoder_feats.append(fusion_feat + feat)
            
        return encoder_feats
    
    def fcm_ip(self, feat):
        # optimal mask
        pixel_feat = torch.max(feat, 1, keepdim=True)[0]
        pixel_feat = pixel_feat + self.fcm_ip_channel_fc(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # optimal descriptor
        flatten_feat = feat.view(feat.size(0), feat.size(1), -1)
        channel_feat_mean = torch.mean(flatten_feat, 2, keepdim=True).view(feat.size(0), 8, 16)
        channel_feat_max = torch.max(flatten_feat, 2, keepdim=True)[0].view(feat.size(0), 8, 16)
        channel_feat = self.fcm_ip_channel_atten(channel_feat_mean, channel_feat_mean, channel_feat_max)
        channel_feat = channel_feat.view(channel_feat.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        # optimal feature
        fcm_ip_feat = pixel_feat * channel_feat 
        return fcm_ip_feat
    
    def fcm_sp(self, feat):
        # holistic mask
        pixel_feat = torch.mean(feat, 1, keepdim=True)
        pixel_feat = pixel_feat + self.fcm_sp_channel_fc(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # holistic descriptor
        flatten_feat = feat.view(feat.size(0), feat.size(1), -1)
        channel_feat_mean = self.sigmoid(torch.mean(flatten_feat, 2, keepdim=True).view(feat.size(0), 8, 16))
        channel_feat_max = self.sigmoid(torch.max(flatten_feat, 2, keepdim=True)[0].view(feat.size(0), 8, 16))
        channel_feat = self.fcm_sp_channel_atten(channel_feat_mean, channel_feat_mean, channel_feat_max)
        channel_feat = channel_feat.view(channel_feat.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        # holistic feature
        fcm_sp_feat = pixel_feat * channel_feat
        return fcm_sp_feat

    def npm(self, feat):
        pixel_feat = torch.min(feat, 1, keepdim=True)[0]
        channel_feat = torch.min(torch.min(feat, 2, keepdim=True)[0], 3, keepdim=True)[0]
        pcmin_feat = pixel_feat * channel_feat
        excitation = self.excite_conv(pcmin_feat)
        excitation = self.sigmoid(excitation)
        excitation1, excitation2 = torch.split(excitation, split_size_or_sections=128, dim=1)
        return excitation1, excitation2

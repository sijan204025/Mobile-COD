# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MobileBackbone(nn.Module):
    def __init__(self, backbone_type='mobilenet_v3_small'):
        super().__init__()
        if backbone_type == 'mobilenet_v3_small':
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.features = backbone.features
            self.channels = [16, 24, 40, 96, 576]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

    def forward(self, x):
        features = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in [1, 3, 6, 9, 12]:
                features.append(x)
        return features

class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )

    def forward(self, x):
        boundary_features = self.boundary_conv(x)
        # Remove sigmoid here - return raw logits
        boundary_attention = boundary_features  # Remove torch.sigmoid()
        return boundary_features, boundary_attention

class MobileFeatureFusion(nn.Module):
    def __init__(self, backbone_channels, boundary_channels=1):
        super().__init__()
        self.fusion_blocks = nn.ModuleList()
        for ch in backbone_channels:
            input_channels = ch + boundary_channels
            fusion_block = nn.Sequential(
                nn.Conv2d(input_channels, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            self.fusion_blocks.append(fusion_block)

    def forward(self, backbone_features, boundary_attention):
        fused_features = []
        for i, feat in enumerate(backbone_features):
            boundary_resized = F.interpolate(boundary_attention, size=feat.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([feat, boundary_resized], dim=1)
            fused = self.fusion_blocks[i](combined)
            fused_features.append(fused)
        return fused_features

class MobileDecoder(nn.Module):
    def __init__(self, fused_channels_list):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        
        for i in range(len(fused_channels_list)-1, 0, -1):
            in_ch = fused_channels_list[i] + fused_channels_list[i-1]
            out_ch = fused_channels_list[i-1]
            up_block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.up_blocks.append(up_block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(fused_channels_list[0], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
            # Remove sigmoid here - return raw logits
        )

    def forward(self, fused_features):
        x = fused_features[-1]
        for i, up_block in enumerate(self.up_blocks):
            target_feat = fused_features[-(i+2)]
            x = F.interpolate(x, size=target_feat.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, target_feat], dim=1)
            x = up_block(x)
        
        output = self.final_conv(x)
        output = F.interpolate(output, size=256, mode='bilinear', align_corners=False)
        return output  # Remove torch.sigmoid()

class LightweightBoundaryAwareCOD(nn.Module):
    def __init__(self, backbone_type='mobilenet_v3_small'):
        super().__init__()
        self.backbone = MobileBackbone(backbone_type)
        self.boundary_aware = BoundaryAwareModule()
        self.feature_fusion = MobileFeatureFusion(self.backbone.channels)
        self.decoder = MobileDecoder(self.backbone.channels)

    def forward(self, x):
         backbone_features = self.backbone(x)
         boundary_features, boundary_attention = self.boundary_aware(x)
         fused_features = self.feature_fusion(backbone_features, boundary_attention)
         camouflage_map = self.decoder(fused_features)
    
    # Return 3 values to match original expectation
         return camouflage_map, boundary_features, boundary_attention

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
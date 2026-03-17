"""
Module 3: Multi-Task Decoders with Skip Connections

Supports ablation: when skip_channels=0, decoders operate without skip connections.

Progressive dimensions (with skips):
    H/16 x W/16 x 256  (upsample from F_hat_32)
    H/8  x W/8  x 128  (+ skip F16)
    H/4  x W/4  x 64   (+ skip F8)
    H/2  x W/2  x 32   (+ skip F4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    Single upsampling block: bilinear 2x upsample -> concat skip -> conv refine.
    Handles skip=None gracefully for ablation.
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_channels = skip_channels
        
        # After concatenation, input channels = in_channels + skip_channels
        total_in = in_channels + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip=None):
        # Bilinear 2x upsample
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Concatenate skip connection if provided
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', 
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class TaskDecoder(nn.Module):
    """
    Generic task decoder: 4-stage progressive upsampling.
    
    When encoder_channels f4/f8/f16 are 0, skip connections are disabled
    and the decoder operates purely from F_hat_32.
    """
    
    def __init__(self, encoder_channels=None):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = {'f4': 24, 'f8': 40, 'f16': 112, 'f32': 160}
        
        self.has_skips = encoder_channels.get('f16', 0) > 0
        
        # Stage 1: project F_hat_32 and upsample
        self.stage1 = nn.Sequential(
            nn.Conv2d(encoder_channels['f32'], 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Stage 2: upsample + optional skip F16
        skip_f16 = encoder_channels.get('f16', 0)
        self.stage2 = DecoderBlock(256, skip_f16, 128)
        
        # Stage 3: upsample + optional skip F8
        skip_f8 = encoder_channels.get('f8', 0)
        self.stage3 = DecoderBlock(128, skip_f8, 64)
        
        # Stage 4: upsample + optional skip F4
        skip_f4 = encoder_channels.get('f4', 0)
        self.stage4 = DecoderBlock(64, skip_f4, 32)
    
    def forward(self, f_hat_32, skip_features):
        # Stage 1: project channels (160->256), upsample to H/16
        x = self.stage1(f_hat_32)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 2-4: upsample with optional skips
        x = self.stage2(x, skip_features.get('f16'))
        x = self.stage3(x, skip_features.get('f8'))
        x = self.stage4(x, skip_features.get('f4'))
        
        return x


class DepthDecoder(nn.Module):
    """Depth decoder: produces single-channel inverse depth map."""
    
    def __init__(self, encoder_channels=None):
        super().__init__()
        self.decoder = TaskDecoder(encoder_channels)
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, f_hat_32, skip_features, target_size=None):
        x = self.decoder(f_hat_32, skip_features)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', 
                              align_corners=False)
        return x


class SegmentationDecoder(nn.Module):
    """Segmentation decoder: produces C-class probability map."""
    
    def __init__(self, num_classes=19, encoder_channels=None):
        super().__init__()
        self.decoder = TaskDecoder(encoder_channels)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, f_hat_32, skip_features, target_size=None):
        x = self.decoder(f_hat_32, skip_features)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', 
                              align_corners=False)
        return x


class NormalDecoder(nn.Module):
    """Surface normal decoder: produces 3-channel L2-normalized normal field."""
    
    def __init__(self, encoder_channels=None):
        super().__init__()
        self.decoder = TaskDecoder(encoder_channels)
        self.head = nn.Conv2d(32, 3, kernel_size=1)
    
    def forward(self, f_hat_32, skip_features, target_size=None):
        x = self.decoder(f_hat_32, skip_features)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', 
                              align_corners=False)
        x = F.normalize(x, p=2, dim=1)
        return x


if __name__ == '__main__':
    B, H, W = 1, 256, 512
    
    # With skips
    enc_ch = {'f4': 24, 'f8': 40, 'f16': 112, 'f32': 160}
    f_hat = torch.randn(B, 160, H//32, W//32)
    skips = {'f4': torch.randn(B, 24, H//4, W//4),
             'f8': torch.randn(B, 40, H//8, W//8),
             'f16': torch.randn(B, 112, H//16, W//16)}
    
    print("=== With skips ===")
    for name, Dec in [('depth', DepthDecoder), ('seg', SegmentationDecoder), ('normal', NormalDecoder)]:
        if name == 'seg':
            dec = Dec(19, enc_ch)
        else:
            dec = Dec(enc_ch)
        out = dec(f_hat, skips, (H, W))
        params = sum(p.numel() for p in dec.parameters())
        print(f"  {name}: {out.shape}, {params/1e6:.3f}M")
    
    # Without skips
    enc_ch_no_skip = {'f4': 0, 'f8': 0, 'f16': 0, 'f32': 160}
    skips_none = {'f4': None, 'f8': None, 'f16': None}
    
    print("\n=== Without skips ===")
    for name, Dec in [('depth', DepthDecoder), ('seg', SegmentationDecoder), ('normal', NormalDecoder)]:
        if name == 'seg':
            dec = Dec(19, enc_ch_no_skip)
        else:
            dec = Dec(enc_ch_no_skip)
        out = dec(f_hat, skips_none, (H, W))
        params = sum(p.numel() for p in dec.parameters())
        print(f"  {name}: {out.shape}, {params/1e6:.3f}M")

"""
EdgeLite-MTAN: Full Model Assembly with Ablation Support

Combines:
    Module 1: SharedEncoder (MobileNetV3-Large)
    Module 2: MultiTaskAttention (channel-wise 1x1 conv, 50% reduction)
    Module 3: Task Decoders (Depth, Segmentation, Surface Normal)

Ablation flags:
    use_attention: If False, F32 goes directly to all decoders (no task-specific modulation)
    use_skips: If False, decoders only receive F_hat_32 (no F4/F8/F16 skip connections)
"""

import torch
import torch.nn as nn

from .encoder import SharedEncoder
from .attention import MultiTaskAttention
from .decoder import DepthDecoder, SegmentationDecoder, NormalDecoder


class EdgeLiteMTAN(nn.Module):
    """
    EdgeLite-MTAN: Lightweight Multi-Task Attention Network.
    
    Args:
        num_classes: Number of segmentation classes (default 19 for Cityscapes)
        pretrained_encoder: Use ImageNet pretrained MobileNetV3-Large
        use_attention: Enable task-specific attention modules (ablation)
        use_skips: Enable skip connections from encoder to decoders (ablation)
    """
    
    def __init__(self, num_classes=19, pretrained_encoder=True,
                 use_attention=True, use_skips=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_skips = use_skips
        
        # Module 1: Shared encoder
        self.encoder = SharedEncoder(pretrained=pretrained_encoder)
        encoder_channels = SharedEncoder.CHANNEL_DIMS
        
        # Module 2: Task-specific attention (only if enabled)
        if self.use_attention:
            self.attention = MultiTaskAttention(
                in_channels=encoder_channels['f32'],
                reduction=2,
            )
        
        # Module 3: Task decoders
        # If skips disabled, pass zero skip channels so decoders don't expect them
        if self.use_skips:
            dec_channels = encoder_channels
        else:
            dec_channels = {'f4': 0, 'f8': 0, 'f16': 0, 'f32': encoder_channels['f32']}
        
        self.depth_decoder = DepthDecoder(dec_channels)
        self.seg_decoder = SegmentationDecoder(num_classes, dec_channels)
        self.normal_decoder = NormalDecoder(dec_channels)
    
    def forward(self, x):
        """
        Args:
            x: Input RGB image [B, 3, H, W]
            
        Returns:
            dict with:
                'depth':  [B, 1, H, W]
                'seg':    [B, C, H, W]
                'normal': [B, 3, H, W]
        """
        target_size = (x.shape[2], x.shape[3])
        
        # Module 1: Extract multi-scale features
        features = self.encoder(x)
        
        # Skip connections (bypass attention -- free taps)
        if self.use_skips:
            skip_features = {
                'f4': features['f4'],
                'f8': features['f8'],
                'f16': features['f16'],
            }
        else:
            skip_features = {
                'f4': None,
                'f8': None,
                'f16': None,
            }
        
        # Module 2: Task-specific attention on F32
        if self.use_attention:
            task_features = self.attention(features['f32'])
        else:
            # No attention: same F32 goes to all decoders
            task_features = {
                'depth': features['f32'],
                'seg': features['f32'],
                'normal': features['f32'],
            }
        
        # Module 3: Decode each task
        depth = self.depth_decoder(
            task_features['depth'], skip_features, target_size)
        seg = self.seg_decoder(
            task_features['seg'], skip_features, target_size)
        normal = self.normal_decoder(
            task_features['normal'], skip_features, target_size)
        
        return {
            'depth': depth,
            'seg': seg,
            'normal': normal,
        }
    
    def get_param_count(self):
        """Return parameter counts per component."""
        counts = {}
        counts['encoder'] = sum(p.numel() for p in self.encoder.parameters())
        if self.use_attention:
            counts['attention'] = sum(p.numel() for p in self.attention.parameters())
        else:
            counts['attention'] = 0
        counts['depth_decoder'] = sum(p.numel() for p in self.depth_decoder.parameters())
        counts['seg_decoder'] = sum(p.numel() for p in self.seg_decoder.parameters())
        counts['normal_decoder'] = sum(p.numel() for p in self.normal_decoder.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


if __name__ == '__main__':
    print("=== Full Model ===")
    model = EdgeLiteMTAN(num_classes=19, pretrained_encoder=False,
                         use_attention=True, use_skips=True)
    dummy = torch.randn(1, 3, 256, 512)
    out = model(dummy)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    counts = model.get_param_count()
    for k, v in counts.items():
        print(f"  {k}: {v/1e6:.3f}M")
    
    print("\n=== Without Attention ===")
    model_no_attn = EdgeLiteMTAN(num_classes=19, pretrained_encoder=False,
                                  use_attention=False, use_skips=True)
    out2 = model_no_attn(dummy)
    counts2 = model_no_attn.get_param_count()
    print(f"  total: {counts2['total']/1e6:.3f}M")
    
    print("\n=== Without Skips ===")
    model_no_skips = EdgeLiteMTAN(num_classes=19, pretrained_encoder=False,
                                   use_attention=True, use_skips=False)
    out3 = model_no_skips(dummy)
    counts3 = model_no_skips.get_param_count()
    print(f"  total: {counts3['total']/1e6:.3f}M")

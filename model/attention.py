"""
Module 2: Task-Specific Attention
Channel-wise attention via 1x1 pointwise convolutions with 50% reduced
intermediate channels (160 -> 80 -> 160).

Each task produces modulated features F_hat_32 = M_t * F_32 (element-wise).
Only F32 is processed; F4/F8/F16 bypass attention entirely.
"""

import torch
import torch.nn as nn


class TaskAttention(nn.Module):
    """
    Single task attention module.
    
    Implements Eq. (2)-(3) from the paper:
        M_t = sigmoid(W2 * ReLU(W1 * F32))
        F_hat_t = M_t ⊙ F32
    
    where W1: R^160 -> R^80 (50% channel reduction)
          W2: R^80 -> R^160 (restore dimensionality)
    """
    
    def __init__(self, in_channels=160, reduction=2):
        super().__init__()
        mid_channels = in_channels // reduction  # 160 -> 80
        
        self.attention = nn.Sequential(
            # W1: channel reduction via 1x1 pointwise conv
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # W2: restore channels via 1x1 pointwise conv
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, f32):
        """
        Args:
            f32: Deepest encoder features [B, 160, H/32, W/32]
            
        Returns:
            Modulated features [B, 160, H/32, W/32]
        """
        mask = self.attention(f32)      # Channel-wise attention mask
        return mask * f32               # Element-wise modulation


class MultiTaskAttention(nn.Module):
    """
    Three parallel task attention modules (depth, segmentation, surface normals).
    Each independently modulates F32 to produce task-specific features.
    """
    
    def __init__(self, in_channels=160, reduction=2):
        super().__init__()
        
        self.depth_attention = TaskAttention(in_channels, reduction)
        self.seg_attention = TaskAttention(in_channels, reduction)
        self.normal_attention = TaskAttention(in_channels, reduction)
    
    def forward(self, f32):
        """
        Args:
            f32: Shared encoder features [B, 160, H/32, W/32]
            
        Returns:
            dict with 'depth', 'seg', 'normal' task-specific features,
            each [B, 160, H/32, W/32]
        """
        return {
            'depth': self.depth_attention(f32),
            'seg': self.seg_attention(f32),
            'normal': self.normal_attention(f32),
        }


if __name__ == '__main__':
    attn = MultiTaskAttention(160, 2)
    dummy_f32 = torch.randn(1, 160, 8, 16)
    out = attn(dummy_f32)
    for k, v in out.items():
        print(f"{k}: {v.shape}")
    
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"\nAttention params: {total_params / 1e6:.4f}M")

"""
Module 1: Shared Lightweight Encoder
MobileNetV3-Large backbone producing features at 4 scales.
F32 -> task attention, F4/F8/F16 -> skip connections (free taps, no addition/concatenation).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SharedEncoder(nn.Module):
    """
    MobileNetV3-Large shared encoder.

    Extracts multi-scale features at 4 resolutions:
        F4:  H/4  x W/4  x 24   (after layer 3)
        F8:  H/8  x W/8  x 40   (after layer 6)
        F16: H/16 x W/16 x 112  (after layer 12)
        F32: H/32 x W/32 x 160  (after layer 15, before classifier)

    These are independent free taps -- no vector addition or concatenation
    between the feature maps at different scales.
    """

    # Channel dimensions at each tap point
    CHANNEL_DIMS = {
        "f4": 24,
        "f8": 40,
        "f16": 112,
        "f32": 160,
    }

    def __init__(self, pretrained=True):
        super().__init__()

        # Load MobileNetV3-Large
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            mobilenet = models.mobilenet_v3_large(weights=weights)
        else:
            mobilenet = models.mobilenet_v3_large(weights=None)

        features = mobilenet.features

        # Split into 4 sequential groups based on stride boundaries
        # MobileNetV3-Large feature layer indices and their output strides:
        #   layers 0-3:   stride 4,  channels 24
        #   layers 4-6:   stride 8,  channels 40
        #   layers 7-12:  stride 16, channels 112
        #   layers 13-16: stride 32, channels 160

        self.stage1 = nn.Sequential(*features[0:4])  # -> H/4 x W/4 x 24
        self.stage2 = nn.Sequential(*features[4:7])  # -> H/8 x W/8 x 40
        self.stage3 = nn.Sequential(*features[7:13])  # -> H/16 x W/16 x 112
        self.stage4 = nn.Sequential(*features[13:16])  # -> H/32 x W/32 x 160

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            dict with keys 'f4', 'f8', 'f16', 'f32' containing feature tensors.
            Each is an independent tap from the sequential forward pass.
        """
        f4 = self.stage1(x)  # [B, 24, H/4, W/4]
        f8 = self.stage2(f4)  # [B, 40, H/8, W/8]
        f16 = self.stage3(f8)  # [B, 112, H/16, W/16]
        f32 = self.stage4(f16)  # [B, 160, H/32, W/32]

        return {
            "f4": f4,
            "f8": f8,
            "f16": f16,
            "f32": f32,
        }


if __name__ == "__main__":
    # Quick sanity check
    encoder = SharedEncoder(pretrained=False)
    dummy = torch.randn(1, 3, 256, 512)
    features = encoder(dummy)
    for k, v in features.items():
        print(f"{k}: {v.shape}")

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder params: {total_params / 1e6:.2f}M")

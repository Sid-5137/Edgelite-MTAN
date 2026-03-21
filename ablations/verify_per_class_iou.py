"""Verify per-class IoU from best checkpoint."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from model import EdgeLiteMTAN
from data.cityscapes import get_dataloaders

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeLiteMTAN(num_classes=19, pretrained_encoder=False).to(device)
ckpt = torch.load('checkpoints/best.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

_, val_loader = get_dataloaders('cityscapes/', batch_size=2, img_size=(512, 1024))

intersection = torch.zeros(19)
union = torch.zeros(19)
gt_count = torch.zeros(19)

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        pred = model(images)['seg'].argmax(dim=1).cpu()
        gt = targets['seg']
        valid = (gt != 255)

        for c in range(19):
            pred_c = (pred == c) & valid
            gt_c = (gt == c) & valid
            intersection[c] += (pred_c & gt_c).sum().float()
            union[c] += (pred_c | gt_c).sum().float()
            gt_count[c] += gt_c.sum().float()

iou = intersection / (union + 1e-6)
freq = gt_count / gt_count.sum()

print(f"\n{'Class':<20} {'IoU':>8} {'GT Pixels %':>12} {'Status'}")
print('-' * 55)
for i, name in enumerate(CLASS_NAMES):
    status = ''
    if iou[i] < 0.1:
        status = '  << FAILING'
    elif iou[i] < 0.3:
        status = '  < WEAK'
    print(f"{name:<20} {iou[i]*100:>7.2f}% {freq[i]*100:>10.2f}% {status}")

print('-' * 55)
print(f"{'mIoU':<20} {iou.mean()*100:>7.2f}%")
print(f"\nClasses with IoU < 10%: {sum(iou < 0.1).item()}")
print(f"Classes with IoU < 30%: {sum(iou < 0.3).item()}")

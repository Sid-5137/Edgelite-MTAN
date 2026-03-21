"""
Generate per-class IoU bar chart for EdgeLite-MTAN paper.
Run: python generate_per_class_iou_chart.py
Output: figures/per_class_iou.png
"""
import matplotlib.pyplot as plt
import numpy as np

# Per-class IoU from best.pth evaluation (epoch 72)
classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

iou_values = [
    97.23, 79.05, 90.05, 48.95, 50.37,
    52.48, 58.89, 70.53, 90.56,
    58.77, 93.99, 74.18, 53.58, 92.61,
    70.68, 76.38, 53.14, 50.36, 69.25
]

mean_iou = np.mean(iou_values)

# Color coding by category
colors = []
# flat: road, sidewalk, terrain
# construction: building, wall, fence
# object: pole, traffic light, traffic sign
# nature: vegetation, sky
# human: person, rider
# vehicle: car, truck, bus, train, motorcycle, bicycle
category_colors = {
    'flat': '#4CAF50',
    'construction': '#2196F3',
    'object': '#FF9800',
    'nature': '#8BC34A',
    'human': '#E91E63',
    'vehicle': '#9C27B0'
}
category_map = {
    'road': 'flat', 'sidewalk': 'flat', 'terrain': 'flat',
    'building': 'construction', 'wall': 'construction', 'fence': 'construction',
    'pole': 'object', 'traffic light': 'object', 'traffic sign': 'object',
    'vegetation': 'nature', 'sky': 'nature',
    'person': 'human', 'rider': 'human',
    'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle',
    'train': 'vehicle', 'motorcycle': 'vehicle', 'bicycle': 'vehicle'
}

for c in classes:
    colors.append(category_colors[category_map[c]])

# Sort by IoU value for better visualization
sorted_indices = np.argsort(iou_values)
sorted_classes = [classes[i] for i in sorted_indices]
sorted_values = [iou_values[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(range(len(sorted_classes)), sorted_values, color=sorted_colors, edgecolor='white', linewidth=0.5, height=0.7)

# Add value labels
for i, (val, bar) in enumerate(zip(sorted_values, bars)):
    ax.text(val + 0.8, i, f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')

# Mean IoU line
ax.axvline(x=mean_iou, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'mIoU = {mean_iou:.1f}%')

ax.set_yticks(range(len(sorted_classes)))
ax.set_yticklabels(sorted_classes, fontsize=9)
ax.set_xlabel('IoU (%)', fontsize=11)
ax.set_xlim(0, 105)
ax.legend(fontsize=10, loc='lower right')
ax.set_title('Per-Class IoU on Cityscapes Validation Set', fontsize=12, fontweight='bold')

# Category legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=category_colors['flat'], label='Flat'),
    Patch(facecolor=category_colors['construction'], label='Construction'),
    Patch(facecolor=category_colors['object'], label='Object'),
    Patch(facecolor=category_colors['nature'], label='Nature'),
    Patch(facecolor=category_colors['human'], label='Human'),
    Patch(facecolor=category_colors['vehicle'], label='Vehicle'),
]
ax2 = ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2, title='Category', title_fontsize=9)
ax.add_artist(ax2)
# Re-add mIoU legend
ax.axvline(x=mean_iou, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(mean_iou + 1, len(sorted_classes) - 1, f'mIoU = {mean_iou:.1f}%', color='red', fontsize=9, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/per_class_iou.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/per_class_iou.pdf', bbox_inches='tight')
print("Saved figures/per_class_iou.png and .pdf")

"""
Visualization utilities for EdgeLite-MTAN.
- Qualitative multi-task result grids
- Radar charts for per-category SNE analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def visualize_predictions(image, predictions, targets=None, save_path=None):
    """
    Create a visualization grid: RGB | Depth | Segmentation | Normals
    
    Args:
        image: [3, H, W] normalized tensor
        predictions: dict with 'depth', 'seg', 'normal'
        targets: optional dict with ground truth
        save_path: path to save figure
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    depth = predictions['depth'].squeeze().cpu().numpy()
    seg = predictions['seg'].argmax(dim=0).cpu().numpy()
    normal = predictions['normal'].cpu().permute(1, 2, 0).numpy()
    normal_vis = (normal + 1) / 2  # Map [-1,1] to [0,1] for display
    
    ncols = 4
    nrows = 2 if targets is not None else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    
    if nrows == 1:
        axes = [axes]
    
    # Predictions row
    axes[0][0].imshow(img)
    axes[0][0].set_title('Input RGB')
    axes[0][1].imshow(depth, cmap='magma_r')
    axes[0][1].set_title('Pred Depth')
    axes[0][2].imshow(seg, cmap='tab20')
    axes[0][2].set_title('Pred Segmentation')
    axes[0][3].imshow(np.clip(normal_vis, 0, 1))
    axes[0][3].set_title('Pred Normals')
    
    # GT row
    if targets is not None:
        gt_depth = targets['depth'].squeeze().cpu().numpy()
        gt_seg = targets['seg'].cpu().numpy()
        gt_normal = targets['normal'].cpu().permute(1, 2, 0).numpy()
        gt_normal_vis = (gt_normal + 1) / 2
        
        axes[1][0].imshow(img)
        axes[1][0].set_title('Input RGB')
        axes[1][1].imshow(gt_depth, cmap='magma_r')
        axes[1][1].set_title('GT Depth')
        axes[1][2].imshow(gt_seg, cmap='tab20')
        axes[1][2].set_title('GT Segmentation')
        axes[1][3].imshow(np.clip(gt_normal_vis, 0, 1))
        axes[1][3].set_title('GT Normals')
    
    for row in axes:
        for ax in row:
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_radar_chart(per_category_mae, per_category_wae, save_path=None):
    """
    Generate radar charts for per-category SNE metrics.
    
    Args:
        per_category_mae: dict {category: mae_degrees} for EdgeLite-MTAN
        per_category_wae: dict {category: wae_degrees} for EdgeLite-MTAN
        save_path: path to save figure
    """
    categories = list(per_category_mae.keys())
    N = len(categories)
    
    if N == 0:
        print("No per-category data available for radar chart")
        return
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    mae_vals = [per_category_mae[c] for c in categories]
    wae_vals = [per_category_wae.get(c, mae_vals[i]) for i, c in enumerate(categories)]
    
    # For radar: outer = better, so invert (max - value)
    max_val = max(max(mae_vals), max(wae_vals)) + 5
    
    mae_radar = [max_val - v for v in mae_vals] + [max_val - mae_vals[0]]
    wae_radar = [max_val - v for v in wae_vals] + [max_val - wae_vals[0]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw=dict(polar=True))
    
    # (a) MAE
    ax = axes[0]
    ax.plot(angles, mae_radar, color='#43A047', linewidth=2.5, marker='D', markersize=5)
    ax.fill(angles, mae_radar, color='#43A047', alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=10, fontweight='bold')
    ax.set_title('(a) Mean Angular Error\n(uniform pixel weighting)', fontsize=12, 
                 fontweight='bold', pad=25)
    
    # (b) WAE
    ax = axes[1]
    ax.plot(angles, wae_radar, color='#43A047', linewidth=2.5, marker='D', markersize=5)
    ax.fill(angles, wae_radar, color='#43A047', alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=10, fontweight='bold')
    ax.set_title('(b) Depth-Weighted Angular Error\n(near-field emphasis)', fontsize=12, 
                 fontweight='bold', pad=25)
    
    plt.suptitle('Surface Normal Estimation: Category-Level Performance',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Radar chart saved to {save_path}")

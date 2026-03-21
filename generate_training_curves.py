"""
Generate training curves for EdgeLite-MTAN paper.
Run: python generate_training_curves.py
Input: logs/training_log.json
Output: figures/training_curves.png
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load training log
with open('logs/training_log.json', 'r') as f:
    log = json.load(f)

epochs = [entry['epoch'] for entry in log]
train_loss = [entry['train_losses']['loss_total'] for entry in log]
val_miou = [entry['val_metrics']['seg/miou'] * 100 for entry in log]
val_delta1 = [entry['val_metrics']['depth/delta1'] * 100 for entry in log]
val_mae = [entry['val_metrics']['normal/mae'] for entry in log]
val_loss = [entry['val_metrics']['loss_total'] for entry in log]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Training & Validation Loss
axes[0, 0].plot(epochs, train_loss, color='#2196F3', linewidth=1.5, label='Train')
axes[0, 0].plot(epochs, val_loss, color='#F44336', linewidth=1.5, label='Val')
axes[0, 0].set_xlabel('Epoch', fontsize=10)
axes[0, 0].set_ylabel('Total Loss', fontsize=10)
axes[0, 0].set_title('Training & Validation Loss', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)

# Plot 2: Validation mIoU
axes[0, 1].plot(epochs, val_miou, color='#4CAF50', linewidth=1.5)
best_miou_idx = np.argmax(val_miou)
best_epoch_miou = epochs[best_miou_idx]
best_miou = val_miou[best_miou_idx]
axes[0, 1].axhline(y=best_miou, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].scatter([best_epoch_miou], [best_miou], color='red', zorder=5, s=40)
axes[0, 1].annotate(f'Best: {best_miou:.1f}% (ep {best_epoch_miou})',
                    xy=(best_epoch_miou, best_miou),
                    xytext=(best_epoch_miou - 15, best_miou - 4),
                    fontsize=8, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
axes[0, 1].set_xlabel('Epoch', fontsize=10)
axes[0, 1].set_ylabel('mIoU (%)', fontsize=10)
axes[0, 1].set_title('Validation mIoU (19-class)', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)

# Plot 3: Validation delta1
axes[1, 0].plot(epochs, val_delta1, color='#FF9800', linewidth=1.5)
best_d1_idx = np.argmax(val_delta1)
best_epoch_d1 = epochs[best_d1_idx]
best_d1 = val_delta1[best_d1_idx]
axes[1, 0].axhline(y=best_d1, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].scatter([best_epoch_d1], [best_d1], color='red', zorder=5, s=40)
axes[1, 0].annotate(f'Best: {best_d1:.1f}% (ep {best_epoch_d1})',
                    xy=(best_epoch_d1, best_d1),
                    xytext=(best_epoch_d1 - 15, best_d1 - 2),
                    fontsize=8, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
axes[1, 0].set_xlabel('Epoch', fontsize=10)
axes[1, 0].set_ylabel('$\\delta_1$ Accuracy (%)', fontsize=10)
axes[1, 0].set_title('Validation Depth $\\delta_1$', fontsize=11, fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)

# Plot 4: Normal MAE (lower is better)
axes[1, 1].plot(epochs, val_mae, color='#9C27B0', linewidth=1.5)
best_mae_idx = np.argmin(val_mae)
best_epoch_mae = epochs[best_mae_idx]
best_mae = val_mae[best_mae_idx]
axes[1, 1].axhline(y=best_mae, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].scatter([best_epoch_mae], [best_mae], color='red', zorder=5, s=40)
axes[1, 1].annotate(f'Best: {best_mae:.1f}° (ep {best_epoch_mae})',
                    xy=(best_epoch_mae, best_mae),
                    xytext=(best_epoch_mae - 15, best_mae + 2),
                    fontsize=8, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
axes[1, 1].set_xlabel('Epoch', fontsize=10)
axes[1, 1].set_ylabel('MAE (degrees)', fontsize=10)
axes[1, 1].set_title('Validation Normal MAE', fontsize=11, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].spines['top'].set_visible(False)
axes[1, 1].spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/training_curves.pdf', bbox_inches='tight')
print("Saved figures/training_curves.png and .pdf")
print(f"\nSummary:")
print(f"  Epochs logged: {epochs[0]} to {epochs[-1]} ({len(epochs)} entries)")
print(f"  Best mIoU:    {best_miou:.1f}% at epoch {best_epoch_miou}")
print(f"  Best delta1:  {best_d1:.1f}% at epoch {best_epoch_d1}")
print(f"  Best MAE:     {best_mae:.1f} deg at epoch {best_epoch_mae}")
print(f"  Final train loss: {train_loss[-1]:.4f}")
print(f"  Final val loss:   {val_loss[-1]:.4f}")

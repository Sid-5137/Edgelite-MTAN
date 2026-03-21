"""Extract best results from all ablation training logs into a summary table."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json

ablations = {
    "Full Model": "logs/training_log.json",
    "No Attention": "logs/ablation_no_attn/training_log.json",
    "No Skips": "logs/ablation_no_skips/training_log.json",
    "Depth Only": "logs/ablation_depth_only/training_log.json",
    "Seg Only": "logs/ablation_seg_only/training_log.json",
    "Normal Only": "logs/ablation_normal_only/training_log.json",
}

print(f"{'Variant':<20} {'mIoU':>8} {'d1':>8} {'AbsRel':>8} {'RMSE':>8} {'N-MAE':>8} {'N-WAE':>8}")
print("-" * 80)

for name, path in ablations.items():
    if not os.path.exists(path):
        print(f"{name:<20} {'(not found)':>8}")
        continue

    with open(path) as f:
        log = json.load(f)

    # Find best epoch by combined metric (same as training script)
    best = None
    best_metric = float('inf')

    for entry in log:
        vm = entry['val_metrics']
        parts = []
        miou = vm.get('seg/miou', 0)
        delta1 = vm.get('depth/delta1', 0)
        mae = vm.get('normal/mae', 90)

        if miou > 0:
            parts.append(1 - miou)
        if delta1 > 0:
            parts.append(1 - delta1)
        if mae > 0 and mae < 90:
            parts.append(mae / 100.0)

        combined = sum(parts) / len(parts) if parts else float('inf')
        if combined < best_metric:
            best_metric = combined
            best = entry

    if best is None:
        print(f"{name:<20} {'(empty log)':>8}")
        continue

    vm = best['val_metrics']
    miou = vm.get('seg/miou', 0) * 100
    delta1 = vm.get('depth/delta1', 0) * 100
    absrel = vm.get('depth/absrel', 0)
    rmse = vm.get('depth/rmse', 0)
    nmae = vm.get('normal/mae', 0)
    nwae = vm.get('normal/wae', 0)

    ep = best['epoch']

    miou_s = f"{miou:.1f}%" if miou > 0 else "---"
    delta1_s = f"{delta1:.1f}%" if delta1 > 0 else "---"
    absrel_s = f"{absrel:.3f}" if absrel > 0 else "---"
    rmse_s = f"{rmse:.2f}" if rmse > 0 else "---"
    nmae_s = f"{nmae:.1f}" if nmae > 0 else "---"
    nwae_s = f"{nwae:.1f}" if nwae > 0 else "---"

    print(f"{name:<20} {miou_s:>8} {delta1_s:>8} {absrel_s:>8} {rmse_s:>8} {nmae_s:>8} {nwae_s:>8}  (ep{ep})")

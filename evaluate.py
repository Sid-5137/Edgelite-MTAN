"""
Evaluation Script for EdgeLite-MTAN

Runs full evaluation on Cityscapes val set and generates:
    - All metric tables (depth, seg, surface normal)
    - Per-category radar charts for SNE
    - Qualitative visualizations

Usage:
    python evaluate.py --data_root /path/to/cityscapes --checkpoint checkpoints/best.pth
"""

import os
import argparse
import json
import time

import torch
from tqdm import tqdm

from model import EdgeLiteMTAN
from data.cityscapes import get_dataloaders, CATEGORY_GROUPS
from utils.metrics import (
    depth_metrics,
    segmentation_metrics,
    normal_metrics,
    MetricAccumulator,
)
from utils.visualization import visualize_predictions, plot_radar_chart


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EdgeLite-MTAN")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument(
        "--num_vis",
        type=int,
        default=20,
        help="Number of visualization samples to save",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, category_groups, output_dir, num_vis=20):
    """Full evaluation with per-category analysis."""
    model.eval()

    accumulator = MetricAccumulator()
    vis_count = 0
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # FPS measurement
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    total_frames = 0

    pbar = tqdm(loader, desc="Evaluating")
    for images, targets in pbar:
        images = images.to(device)
        targets_device = {k: v.to(device) for k, v in targets.items()}

        # Forward (with timing)
        predictions = model(images)
        total_frames += images.shape[0]

        # Metrics
        d_met = depth_metrics(
            predictions["depth"], targets_device["depth"], targets_device["valid_depth"]
        )
        s_met = segmentation_metrics(predictions["seg"], targets_device["seg"])
        n_met = normal_metrics(
            predictions["normal"],
            targets_device["normal"],
            targets_device["valid_depth"],
            depth=targets_device["depth"],
            seg_labels=targets_device["seg"],
            category_groups=category_groups,
        )

        accumulator.update(d_met, s_met, n_met)

        # Visualizations
        if vis_count < num_vis:
            for b in range(min(images.shape[0], num_vis - vis_count)):
                save_path = os.path.join(vis_dir, f"sample_{vis_count:04d}.png")
                pred_b = {k: v[b] for k, v in predictions.items()}
                tgt_b = {k: v[b] for k, v in targets.items()}
                visualize_predictions(images[b].cpu(), pred_b, tgt_b, save_path)
                vis_count += 1

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.time() - start_time
    fps = total_frames / total_time

    # Compute final metrics
    results = accumulator.compute()
    results["fps"] = fps
    results["total_frames"] = total_frames
    results["total_time"] = total_time

    return results


def print_results(results, model):
    """Print formatted results matching paper tables."""
    counts = model.get_param_count()

    print("\n" + "=" * 70)
    print("EdgeLite-MTAN EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nModel: {counts['total'] / 1e6:.1f}M parameters")
    print(f"FPS: {results['fps']:.1f}")

    print(f"\n--- Depth Estimation (Table II) ---")
    print(f"  δ1:     {results.get('depth/delta1', 0):.3f}")
    print(f"  AbsRel: {results.get('depth/absrel', 0):.3f}")
    print(f"  RMSE:   {results.get('depth/rmse', 0):.2f}")

    print(f"\n--- Semantic Segmentation (Table I) ---")
    print(f"  mIoU: {results.get('seg/miou', 0) * 100:.1f}%")

    print(f"\n--- Surface Normal Estimation (Table III) ---")
    print(f"  MAE:   {results.get('normal/mae', 0):.1f}°")
    print(f"  MedAE: {results.get('normal/medae', 0):.1f}°")
    print(f"  WAE:   {results.get('normal/wae', 0):.1f}°")
    print(f"  PFE:   {results.get('normal/pfe', 0):.3f}")

    if results.get("per_category_mae"):
        print(f"\n--- Per-Category MAE (Radar Chart a) ---")
        for cat, val in results["per_category_mae"].items():
            print(f"  {cat:12s}: {val:.1f}°")

    if results.get("per_category_wae"):
        print(f"\n--- Per-Category WAE (Radar Chart b) ---")
        for cat, val in results["per_category_wae"].items():
            print(f"  {cat:12s}: {val:.1f}°")

    print("\n" + "=" * 70)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = EdgeLiteMTAN(num_classes=args.num_classes, pretrained_encoder=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # Data
    _, val_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
    )

    # Evaluate
    results = evaluate(
        model,
        val_loader,
        device,
        CATEGORY_GROUPS,
        args.output_dir,
        num_vis=args.num_vis,
    )

    # Print results
    print_results(results, model)

    # Generate radar chart
    if results.get("per_category_mae") and results.get("per_category_wae"):
        radar_path = os.path.join(args.output_dir, "radar_sne_metrics.png")
        plot_radar_chart(
            results["per_category_mae"],
            results["per_category_wae"],
            save_path=radar_path,
        )

    # Save results to JSON
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    json_results = {
        k: v for k, v in results.items() if isinstance(v, (int, float, str, dict))
    }
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

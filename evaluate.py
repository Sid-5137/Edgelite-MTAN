"""
Evaluation Script for MTAN-Lite

Runs full evaluation on val set and generates:
    - All metric tables (depth, seg, surface normal)
    - Per-class IoU bar chart
    - Per-category radar charts for SNE
    - Qualitative visualizations

Usage:
    # Cityscapes
    python evaluate.py --dataset cityscapes --data_root cityscapes/ --checkpoint checkpoints/cityscapes/best.pth

    # NYUv2
    python evaluate.py --dataset nyuv2 --data_root nyuv2/ --checkpoint checkpoints/nyuv2/best.pth
"""

import os
import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from model import EdgeLiteMTAN
from utils.metrics import (
    depth_metrics,
    segmentation_metrics,
    normal_metrics,
    MetricAccumulator,
)
from utils.visualization import visualize_predictions, plot_radar_chart

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# DATASET CONFIGS
# ============================================================

DATASET_CONFIGS = {
    "cityscapes": {
        "num_classes": 19,
        "img_h": 512,
        "img_w": 1024,
        "max_depth": 80.0,
        "class_names": [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle"
        ],
        "category_colors": {
            "flat": "#4CAF50", "construction": "#2196F3",
            "object": "#FF9800", "nature": "#8BC34A",
            "human": "#E91E63", "vehicle": "#9C27B0"
        },
        "class_to_category": {
            "road": "flat", "sidewalk": "flat", "terrain": "flat",
            "building": "construction", "wall": "construction", "fence": "construction",
            "pole": "object", "traffic light": "object", "traffic sign": "object",
            "vegetation": "nature", "sky": "nature",
            "person": "human", "rider": "human",
            "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
            "train": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle"
        },
        "seg_palette": np.array([
            [128, 64, 128], [244, 35, 232], [70, 70, 70],
            [102, 102, 156], [190, 153, 153], [153, 153, 153],
            [250, 170, 30], [220, 220, 0], [107, 142, 35],
            [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
        ], dtype=np.uint8),
    },
    "nyuv2": {
        "num_classes": 13,
        "img_h": 288,
        "img_w": 384,
        "max_depth": 10.0,
        "class_names": [
            "bed", "books", "ceiling", "chair", "floor",
            "furniture", "objects", "painting", "sofa",
            "table", "tv", "wall", "window"
        ],
        "category_colors": {
            "floor": "#4CAF50", "wall": "#2196F3",
            "furniture": "#FF9800", "structure": "#8BC34A",
            "objects": "#E91E63"
        },
        "class_to_category": {
            "bed": "furniture", "books": "objects", "ceiling": "structure",
            "chair": "furniture", "floor": "floor", "furniture": "furniture",
            "objects": "objects", "painting": "objects", "sofa": "furniture",
            "table": "furniture", "tv": "objects", "wall": "wall",
            "window": "structure"
        },
        "seg_palette": np.array([
            [0, 0, 128],     # 0  bed (navy)
            [128, 64, 0],    # 1  books (brown)
            [200, 200, 200], # 2  ceiling (light gray)
            [255, 128, 0],   # 3  chair (orange)
            [128, 64, 128],  # 4  floor (purple)
            [180, 120, 60],  # 5  furniture (tan)
            [255, 0, 255],   # 6  objects (magenta)
            [255, 255, 0],   # 7  painting (yellow)
            [0, 128, 128],   # 8  sofa (teal)
            [128, 128, 0],   # 9  table (olive)
            [0, 255, 255],   # 10 tv (cyan)
            [70, 130, 180],  # 11 wall (steel blue)
            [220, 20, 60],   # 12 window (crimson)
        ], dtype=np.uint8),
    },
}


def load_dataset(name):
    """Dynamic import like train.py"""
    if name == "cityscapes":
        from data.cityscapes import get_dataloaders, CATEGORY_GROUPS
    elif name == "nyuv2":
        from data.nyuv2 import get_dataloaders, CATEGORY_GROUPS
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return get_dataloaders, CATEGORY_GROUPS


def colorize_seg(seg_map, palette):
    """Convert class indices to RGB."""
    h, w = seg_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        mask = seg_map == cls_id
        color[mask] = palette[cls_id]
    return color


# ============================================================
# EVALUATION
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MTAN-Lite")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["cityscapes", "nyuv2"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_vis", type=int, default=20,
                       help="Number of visualization samples")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, config, category_groups, output_dir, num_vis=20):
    """Full evaluation with per-class and per-category analysis."""
    model.eval()
    num_classes = config["num_classes"]
    max_depth = config["max_depth"]
    palette = config["seg_palette"]

    accumulator = MetricAccumulator(num_classes=num_classes)
    vis_count = 0
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)

    # Per-class IoU tracking
    global_intersection = np.zeros(num_classes)
    global_union = np.zeros(num_classes)

    # FPS measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    total_frames = 0

    pbar = tqdm(loader, desc="Evaluating")
    for images, targets in pbar:
        images = images.to(device)
        targets_device = {k: v.to(device) for k, v in targets.items()}

        predictions = model(images)
        total_frames += images.shape[0]

        # Metrics
        d_met = depth_metrics(
            predictions["depth"], targets_device["depth"],
            targets_device["valid_depth"], max_depth=max_depth
        )
        s_met = segmentation_metrics(
            predictions["seg"], targets_device["seg"],
            num_classes=num_classes
        )
        n_met = normal_metrics(
            predictions["normal"], targets_device["normal"],
            targets_device["valid_depth"],
            depth=targets_device["depth"],
            seg_labels=targets_device["seg"],
            category_groups=category_groups,
        )

        accumulator.update(d_met, s_met, n_met)

        # Track per-class IoU
        global_intersection += s_met["intersection"]
        global_union += s_met["union"]

        # Visualizations with correct palette
        if vis_count < num_vis:
            for b in range(min(images.shape[0], num_vis - vis_count)):
                save_path = os.path.join(vis_dir, f"sample_{vis_count:04d}.png")
                pred_b = {k: v[b] for k, v in predictions.items()}
                tgt_b = {k: v[b] for k, v in targets.items()}
                _visualize_sample(images[b].cpu(), pred_b, tgt_b, palette, save_path)
                vis_count += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    fps = total_frames / total_time

    # Compute final metrics
    results = accumulator.compute()
    results["fps"] = fps
    results["total_frames"] = total_frames
    results["total_time"] = total_time

    # Per-class IoU
    per_class_iou = {}
    for c in range(num_classes):
        if global_union[c] > 0:
            per_class_iou[config["class_names"][c]] = float(
                global_intersection[c] / global_union[c] * 100
            )
        else:
            per_class_iou[config["class_names"][c]] = 0.0
    results["per_class_iou"] = per_class_iou

    return results


def _visualize_sample(image, predictions, targets, palette, save_path):
    """Create visualization grid with dataset-specific palette and row labels."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    depth = predictions["depth"].squeeze().cpu().numpy()
    seg = predictions["seg"].argmax(dim=0).cpu().numpy()
    seg_color = colorize_seg(seg, palette)
    normal = predictions["normal"].cpu().permute(1, 2, 0).numpy()
    normal_vis = (normal + 1) / 2

    ncols = 4
    nrows = 2 if targets is not None else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    if nrows == 1:
        axes = [axes]

    # --- Predicted row ---
    axes[0][0].imshow(img)
    axes[0][0].set_title("RGB", fontsize=11, fontweight="bold")
    axes[0][1].imshow(depth, cmap="magma_r")
    axes[0][1].set_title("Depth", fontsize=11, fontweight="bold")
    axes[0][2].imshow(seg_color)
    axes[0][2].set_title("Segmentation", fontsize=11, fontweight="bold")
    axes[0][3].imshow(np.clip(normal_vis, 0, 1))
    axes[0][3].set_title("Normals", fontsize=11, fontweight="bold")
    # Row label
    axes[0][0].set_ylabel("Predicted", fontsize=13, fontweight="bold",
                          rotation=90, labelpad=10)

    if targets is not None:
        gt_depth = targets["depth"].squeeze().cpu().numpy()
        gt_seg = targets["seg"].cpu().numpy()
        gt_seg_color = colorize_seg(gt_seg, palette)
        gt_normal = targets["normal"].cpu().permute(1, 2, 0).numpy()
        gt_normal_vis = (gt_normal + 1) / 2

        # --- Ground Truth row ---
        axes[1][0].imshow(img)
        axes[1][0].set_title("")
        axes[1][1].imshow(gt_depth, cmap="magma_r")
        axes[1][1].set_title("")
        axes[1][2].imshow(gt_seg_color)
        axes[1][2].set_title("")
        axes[1][3].imshow(np.clip(gt_normal_vis, 0, 1))
        axes[1][3].set_title("")
        # Row label
        axes[1][0].set_ylabel("Ground Truth", fontsize=13, fontweight="bold",
                              rotation=90, labelpad=10)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# CHART GENERATION
# ============================================================

def generate_per_class_iou_chart(per_class_iou, config, output_dir):
    """Generate per-class IoU horizontal bar chart."""
    classes = list(per_class_iou.keys())
    values = list(per_class_iou.values())
    mean_iou = np.mean(values)

    # Colors by category
    cat_colors = config["category_colors"]
    cls_to_cat = config["class_to_category"]
    colors = [cat_colors.get(cls_to_cat.get(c, ""), "#888888") for c in classes]

    # Sort by IoU
    sorted_idx = np.argsort(values)
    s_classes = [classes[i] for i in sorted_idx]
    s_values = [values[i] for i in sorted_idx]
    s_colors = [colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.4)))

    bars = ax.barh(range(len(s_classes)), s_values, color=s_colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    for i, (val, bar) in enumerate(zip(s_values, bars)):
        ax.text(val + 0.8, i, f"{val:.1f}%", va="center",
                fontsize=8, fontweight="bold")

    ax.axvline(x=mean_iou, color="red", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.text(mean_iou + 1, len(s_classes) - 1,
            f"mIoU = {mean_iou:.1f}%", color="red",
            fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(s_classes)))
    ax.set_yticklabels(s_classes, fontsize=9)
    ax.set_xlabel("IoU (%)", fontsize=11)
    ax.set_xlim(0, 105)

    dataset_name = "NYUv2" if config["num_classes"] == 13 else "Cityscapes"
    ax.set_title(f"Per-Class IoU on {dataset_name} Validation Set",
                 fontsize=12, fontweight="bold")

    # Category legend
    unique_cats = list(dict.fromkeys(cls_to_cat.values()))
    legend_elements = [Patch(facecolor=cat_colors.get(c, "#888"),
                             label=c.capitalize()) for c in unique_cats]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=8, ncol=2, title="Category", title_fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "per_class_iou.png")
    pdf_path = os.path.join(output_dir, "per_class_iou.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Per-class IoU chart saved to {png_path}")


# ============================================================
# RESULTS PRINTING
# ============================================================

def print_results(results, model, dataset_name):
    """Print formatted results."""
    counts = model.get_param_count()

    print("\n" + "=" * 70)
    print(f"MTAN-Lite EVALUATION RESULTS ({dataset_name.upper()})")
    print("=" * 70)

    print(f"\nModel: {counts['total'] / 1e6:.1f}M parameters")
    print(f"FPS: {results['fps']:.1f}")
    print(f"Frames evaluated: {results['total_frames']}")

    print(f"\n--- Depth Estimation ---")
    print(f"  δ1:     {results.get('depth/delta1', 0):.3f}")
    print(f"  AbsRel: {results.get('depth/absrel', 0):.4f}")
    print(f"  RMSE:   {results.get('depth/rmse', 0):.3f}m")

    print(f"\n--- Semantic Segmentation ---")
    print(f"  mIoU: {results.get('seg/miou', 0) * 100:.1f}%")

    if results.get("per_class_iou"):
        print(f"\n  Per-class IoU:")
        for cls, iou in results["per_class_iou"].items():
            print(f"    {cls:15s}: {iou:.1f}%")

    print(f"\n--- Surface Normal Estimation ---")
    print(f"  MAE:   {results.get('normal/mae', 0):.2f}°")
    print(f"  MedAE: {results.get('normal/medae', 0):.2f}°")
    print(f"  WAE:   {results.get('normal/wae', 0):.2f}°")
    print(f"  PFE:   {results.get('normal/pfe', 0):.4f}")

    if results.get("per_category_mae"):
        print(f"\n  Per-Category MAE:")
        for cat, val in results["per_category_mae"].items():
            print(f"    {cat:12s}: {val:.1f}°")

    if results.get("per_category_wae"):
        print(f"\n  Per-Category WAE:")
        for cat, val in results["per_category_wae"].items():
            print(f"    {cat:12s}: {val:.1f}°")

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    if args.output_dir is None:
        args.output_dir = f"eval_results/{args.dataset}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # Load dataset
    get_dataloaders, CATEGORY_GROUPS = load_dataset(args.dataset)

    # Load model
    model = EdgeLiteMTAN(
        num_classes=config["num_classes"],
        pretrained_encoder=False
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Data
    _, val_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(config["img_h"], config["img_w"]),
    )

    # Evaluate
    results = evaluate(
        model, val_loader, device, config,
        CATEGORY_GROUPS, args.output_dir,
        num_vis=args.num_vis,
    )

    # Print results
    print_results(results, model, args.dataset)

    # Generate per-class IoU chart
    if results.get("per_class_iou"):
        generate_per_class_iou_chart(
            results["per_class_iou"], config, args.output_dir
        )

    # Generate radar chart
    if results.get("per_category_mae") and results.get("per_category_wae"):
        radar_path = os.path.join(args.output_dir, "radar_sne_metrics.png")
        plot_radar_chart(
            results["per_category_mae"],
            results["per_category_wae"],
            save_path=radar_path,
        )

    # Save results JSON
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    json_results = {
        k: v for k, v in results.items()
        if isinstance(v, (int, float, str, dict))
    }
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations in: {os.path.join(args.output_dir, 'visualisations')}/")


if __name__ == "__main__":
    main()

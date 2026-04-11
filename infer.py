"""
MTAN-Lite Inference Script - Run on any image

Loads trained checkpoint, runs inference on a single image,
and saves depth, segmentation, and surface normal outputs.

Usage:
    # Cityscapes checkpoint on any outdoor image
    python infer.py --image path/to/photo.jpg --checkpoint checkpoints/cityscapes/best.pth --dataset cityscapes

    # NYUv2 checkpoint on any indoor image
    python infer.py --image path/to/photo.jpg --checkpoint checkpoints/nyuv2/best.pth --dataset nyuv2

    # Custom output directory
    python infer.py --image photo.jpg --checkpoint best.pth --dataset cityscapes --output_dir demo_outputs/

    # Run on a folder of images
    python infer.py --image_dir path/to/images/ --checkpoint best.pth --dataset cityscapes
"""

import os
import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from model import EdgeLiteMTAN


# ============================================================
# DATASET CONFIGS
# ============================================================

CONFIGS = {
    "cityscapes": {
        "num_classes": 19,
        "img_h": 512,
        "img_w": 1024,
        "class_names": [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle"
        ],
        "palette": np.array([
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
        "class_names": [
            "bed", "books", "ceiling", "chair", "floor",
            "furniture", "objects", "painting", "sofa",
            "table", "tv", "wall", "window"
        ],
        "palette": np.array([
            [0, 0, 128], [128, 64, 0], [200, 200, 200],
            [255, 128, 0], [128, 64, 128], [180, 120, 60],
            [255, 0, 255], [255, 255, 0], [0, 128, 128],
            [128, 128, 0], [0, 255, 255], [70, 130, 180],
            [220, 20, 60],
        ], dtype=np.uint8),
    },
}

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(checkpoint_path, dataset="cityscapes", device="cuda"):
    """Load MTAN-Lite model from checkpoint."""
    config = CONFIGS[dataset]
    model = EdgeLiteMTAN(
        num_classes=config["num_classes"],
        pretrained_encoder=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Dataset: {dataset} ({config['num_classes']} classes)")
    print(f"  Input resolution: {config['img_h']}x{config['img_w']}")
    return model, config


def preprocess_image(image_path, config, device="cuda"):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)

    # Resize to model input
    img_resized = img.resize((config["img_w"], config["img_h"]), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # To tensor and normalize
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    tensor = (tensor - MEAN) / STD
    tensor = tensor.to(device)

    return tensor, img_resized, original_size


def colorize_seg(seg_map, palette):
    """Convert class index map to RGB."""
    h, w = seg_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        color[seg_map == cls_id] = palette[cls_id]
    return color


def colorize_depth(depth_map, cmap="magma_r"):
    """Convert depth map to colored visualization."""
    # Normalize to 0-1
    d = depth_map.copy()
    valid = d > 0
    if valid.any():
        d_min, d_max = d[valid].min(), d[valid].max()
        d = (d - d_min) / (d_max - d_min + 1e-8)
    cm = plt.get_cmap(cmap)
    colored = (cm(d)[:, :, :3] * 255).astype(np.uint8)
    return colored


def colorize_normals(normal_map):
    """Convert normal vectors [-1,1] to RGB [0,255]."""
    vis = ((normal_map + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return vis


@torch.no_grad()
def run_inference(model, tensor, config):
    """Run model inference and return numpy outputs."""
    predictions = model(tensor)

    # Depth: squeeze and convert
    depth = predictions["depth"].squeeze().cpu().numpy()

    # Segmentation: argmax
    seg = predictions["seg"].argmax(dim=1).squeeze().cpu().numpy()

    # Normals: permute to HxWx3
    normal = predictions["normal"].squeeze().cpu().permute(1, 2, 0).numpy()

    return depth, seg, normal


def save_outputs(image_path, img_resized, depth, seg, normal, config, output_dir):
    """Save individual outputs and a combined visualization."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    img_np = np.array(img_resized)
    palette = config["palette"]
    class_names = config["class_names"]

    # Colorize outputs
    depth_vis = colorize_depth(depth)
    seg_vis = colorize_seg(seg, palette)
    normal_vis = colorize_normals(normal)

    # Save individual images
    Image.fromarray(depth_vis).save(os.path.join(output_dir, f"{basename}_depth.png"))
    Image.fromarray(seg_vis).save(os.path.join(output_dir, f"{basename}_segmentation.png"))
    Image.fromarray(normal_vis).save(os.path.join(output_dir, f"{basename}_normals.png"))

    # Combined figure with labels
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(img_np)
    axes[0].set_title("Input RGB", fontsize=14, fontweight="bold")

    axes[1].imshow(depth_vis)
    axes[1].set_title("Predicted Depth", fontsize=14, fontweight="bold")

    axes[2].imshow(seg_vis)
    axes[2].set_title("Semantic Segmentation", fontsize=14, fontweight="bold")

    axes[3].imshow(normal_vis)
    axes[3].set_title("Surface Normals", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"{basename}_combined.png")
    plt.savefig(combined_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    # Segmentation with legend
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(seg_vis)
    ax.set_title("Semantic Segmentation", fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Build legend from classes present in this image
    unique_classes = np.unique(seg)
    legend_patches = []
    for cls_id in unique_classes:
        if cls_id < len(class_names):
            color = palette[cls_id] / 255.0
            legend_patches.append(Patch(
                facecolor=color,
                label=f"{class_names[cls_id]} ({cls_id})"
            ))
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
              framealpha=0.8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_seg_legend.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    # Segmentation overlay on original image
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_np)
    ax.imshow(seg_vis, alpha=0.5)
    ax.set_title("Segmentation Overlay", fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
              framealpha=0.8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_overlay.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved: {output_dir}/{basename}_combined.png")
    print(f"  Saved: {output_dir}/{basename}_depth.png")
    print(f"  Saved: {output_dir}/{basename}_segmentation.png")
    print(f"  Saved: {output_dir}/{basename}_normals.png")
    print(f"  Saved: {output_dir}/{basename}_seg_legend.png")
    print(f"  Saved: {output_dir}/{basename}_overlay.png")


def main():
    parser = argparse.ArgumentParser(description="MTAN-Lite Inference")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to a folder of images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--dataset", type=str, default="cityscapes",
                        choices=["cityscapes", "nyuv2"],
                        help="Which config to use (affects num_classes and resolution)")
    parser.add_argument("--output_dir", type=str, default="demo_outputs",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide either --image or --image_dir")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, args.dataset, device)

    # Collect images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths.sort()

    print(f"\nProcessing {len(image_paths)} image(s)...\n")

    for img_path in image_paths:
        print(f"Image: {img_path}")

        # Preprocess
        tensor, img_resized, original_size = preprocess_image(img_path, config, device)

        # Inference
        depth, seg, normal = run_inference(model, tensor, config)

        # Save
        save_outputs(img_path, img_resized, depth, seg, normal, config, args.output_dir)
        print()

    print(f"Done! All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

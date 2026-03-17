"""
Generate Pseudo-GT Surface Normals using Omnidata v2

Fixed version: handles Cityscapes 2:1 aspect ratio properly by processing
left and right halves separately through the 384x384 model, then stitching.

Setup:
    pip install timm

Usage:
    python generate_normals.py --data_root /path/to/cityscapes
    python generate_normals.py --data_root /path/to/cityscapes --target_h 512 --target_w 1024

Output:
    cityscapes/normals_omnidata/{train,val}/city/filename_normal.npy
"""

import os
import argparse
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-GT normals with Omnidata"
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--target_h", type=int, default=256)
    parser.add_argument("--target_w", type=int, default=512)
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate even if .npy files already exist",
    )
    return parser.parse_args()


def predict_normals_tiled(model, image, device, tile_size=384, overlap=64):
    """
    Predict normals by splitting the image into overlapping square tiles,
    running each through the model, and blending the results.

    This handles Cityscapes 2:1 aspect ratio without distortion.

    Args:
        model: Omnidata DPT model (expects 384x384 input)
        image: PIL Image (any size/aspect ratio)
        device: torch device
        tile_size: model input size (384 for Omnidata)
        overlap: pixel overlap between tiles for smooth blending

    Returns:
        normals: [3, H, W] numpy array of L2-normalized surface normals
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    orig_w, orig_h = image.size

    # Resize so the shorter side = tile_size, preserving aspect ratio
    scale = tile_size / min(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    # Make sure dimensions are at least tile_size
    new_h = max(new_h, tile_size)
    new_w = max(new_w, tile_size)

    image_resized = image.resize((new_w, new_h), Image.BILINEAR)
    img_tensor = transforms.ToTensor()(image_resized)  # [3, new_h, new_w]

    # Accumulator for normals and weight map (for blending overlaps)
    normal_acc = torch.zeros(3, new_h, new_w)
    weight_acc = torch.zeros(1, new_h, new_w)

    # Generate tile positions
    stride = tile_size - overlap

    y_positions = list(range(0, new_h - tile_size + 1, stride))
    if not y_positions or y_positions[-1] + tile_size < new_h:
        y_positions.append(new_h - tile_size)

    x_positions = list(range(0, new_w - tile_size + 1, stride))
    if not x_positions or x_positions[-1] + tile_size < new_w:
        x_positions.append(new_w - tile_size)

    # Remove duplicates and sort
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    # Create blending weight (raised cosine window for smooth transitions)
    blend_1d = torch.ones(tile_size)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap)
        blend_1d[:overlap] = ramp
        blend_1d[-overlap:] = ramp.flip(0)
    blend_weight = blend_1d.unsqueeze(1) * blend_1d.unsqueeze(0)  # [tile, tile]
    blend_weight = blend_weight.unsqueeze(0)  # [1, tile, tile]

    # Process each tile
    for y in y_positions:
        for x in x_positions:
            tile = img_tensor[:, y : y + tile_size, x : x + tile_size]  # [3, 384, 384]
            tile_norm = normalize(tile).unsqueeze(0).to(device)  # [1, 3, 384, 384]

            with torch.no_grad():
                pred = model(tile_norm)  # [1, 3, 384, 384]
                pred = F.normalize(pred, p=2, dim=1)

            pred_cpu = pred.squeeze(0).cpu()  # [3, 384, 384]

            normal_acc[:, y : y + tile_size, x : x + tile_size] += (
                pred_cpu * blend_weight
            )
            weight_acc[:, y : y + tile_size, x : x + tile_size] += blend_weight

    # Normalize by weight
    weight_acc = torch.clamp(weight_acc, min=1e-6)
    normals = normal_acc / weight_acc  # [3, new_h, new_w]

    # L2 normalize the final result
    normals = F.normalize(normals.unsqueeze(0), p=2, dim=1).squeeze(0)

    return normals.numpy().astype(np.float32)


def main():
    args = parse_args()
    target_size = (args.target_h, args.target_w)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Omnidata normal model
    print("Loading Omnidata surface normal model...")
    model = torch.hub.load("alexsax/omnidata_models", "surface_normal_dpt_hybrid_384")
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    for split in ["train", "val"]:
        img_dir = os.path.join(args.data_root, "leftImg8bit", split)
        out_dir = os.path.join(args.data_root, "normals_omnidata", split)

        images = sorted(glob.glob(os.path.join(img_dir, "*", "*.png")))
        print(f"\nProcessing {len(images)} images for {split}...")

        skipped = 0
        for img_path in tqdm(images, desc=split):
            parts = img_path.split(os.sep)
            city = parts[-2]
            fname = parts[-1].replace("_leftImg8bit.png", "_normal.npy")

            city_out_dir = os.path.join(out_dir, city)
            os.makedirs(city_out_dir, exist_ok=True)
            out_path = os.path.join(city_out_dir, fname)

            # Skip if already exists (unless --force)
            if os.path.exists(out_path) and not args.force:
                skipped += 1
                continue

            # Load image
            image = Image.open(img_path).convert("RGB")

            # Predict normals with tiling (handles any aspect ratio)
            normals = predict_normals_tiled(
                model, image, device, tile_size=384, overlap=64
            )

            # Resize to target resolution
            normals_tensor = torch.from_numpy(normals).unsqueeze(0)
            normals_tensor = F.interpolate(
                normals_tensor, size=target_size, mode="bilinear", align_corners=False
            )
            normals_tensor = F.normalize(normals_tensor, p=2, dim=1)

            normal_np = normals_tensor.squeeze(0).numpy().astype(np.float32)
            np.save(out_path, normal_np)

        if skipped > 0:
            print(f"  Skipped {skipped} existing files (use --force to regenerate)")

    print(
        f"\nDone! Normals saved to: {os.path.join(args.data_root, 'normals_omnidata')}"
    )
    print(f"\nRetrain with:")
    print(
        f"  python train.py --data_root {args.data_root} --epochs 80 --batch_size 2 "
        f"--img_h {args.target_h} --img_w {args.target_w} --w_seg 4.0 --w_normal 0.5"
    )


if __name__ == "__main__":
    main()

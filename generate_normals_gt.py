"""
Generate Surface Normals from Cityscapes Disparity Maps

Uses DSINE's depth-to-normal pipeline (cross-product method) following
the approach in DSINE/notes/depth_to_normal.ipynb (Option 1).

Pipeline:
    1. Read 16-bit disparity PNG
    2. Convert: d = (p - 1) / 256 (official Cityscapes formula)
    3. Compute depth: depth = baseline * fx / d
    4. Build intrinsics matrix, compute inverse
    5. Unproject depth to 3D camera coordinates
    6. Compute surface normals via cross-product (DSINE d2n_tblr)
    7. Save as .npy [3, H, W] float32

Requirements:
    git clone https://github.com/baegwangbin/DSINE.git
    (no weights or extra dependencies needed — only math utilities)

Usage:
    python generate_normals.py --data_root /path/to/cityscapes --dsine_path /path/to/DSINE
    python generate_normals.py --data_root /path/to/cityscapes --dsine_path /path/to/DSINE --target_h 512 --target_w 1024
"""

import os
import sys
import argparse
import glob
import json
import importlib.util

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F


def load_module(name, filepath):
    """Load a Python module directly from file path, bypassing sys.path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate normals from Cityscapes disparity"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to Cityscapes root"
    )
    parser.add_argument(
        "--dsine_path", type=str, required=True, help="Path to cloned DSINE repository"
    )
    parser.add_argument("--target_h", type=int, default=512)
    parser.add_argument("--target_w", type=int, default=1024)
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Neighborhood size for cross-product (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate even if .npy files already exist",
    )
    return parser.parse_args()


def load_camera_intrinsics(data_root, split, city, frame_name):
    """
    Load per-frame camera intrinsics from Cityscapes camera JSON.
    Falls back to default values if JSON not available.
    """
    camera_path = os.path.join(
        data_root, "camera", split, city, frame_name + "_camera.json"
    )

    if os.path.exists(camera_path):
        with open(camera_path, "r") as f:
            calib = json.load(f)

        intrinsic = calib.get("intrinsic", {})
        extrinsic = calib.get("extrinsic", {})

        fx = intrinsic.get("fx", 2262.52)
        fy = intrinsic.get("fy", 2262.52)
        cx = intrinsic.get("u0", 1024.0)
        cy = intrinsic.get("v0", 512.0)
        baseline = extrinsic.get("baseline", 0.209313)

        return baseline, fx, fy, cx, cy
    else:
        return 0.209313, 2262.52, 2262.52, 1024.0, 512.0


def build_intrinsics_matrix(fx, fy, cx, cy):
    """Build 3x3 camera intrinsics matrix."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32)


def scale_intrinsics(intrins, orig_h, orig_w, new_h, new_w):
    """Scale intrinsics when image is resized."""
    sx = new_w / orig_w
    sy = new_h / orig_h
    scaled = intrins.copy()
    scaled[0, 0] *= sx  # fx
    scaled[0, 2] *= sx  # cx
    scaled[1, 1] *= sy  # fy
    scaled[1, 2] *= sy  # cy
    return scaled


def main():
    args = parse_args()
    target_size = (args.target_h, args.target_w)

    # Load DSINE utilities directly by file path (avoids utils/ naming conflict)
    try:
        projection = load_module(
            "dsine_projection",
            os.path.join(args.dsine_path, "utils", "projection.py"),
        )
        cross_module = load_module(
            "dsine_cross",
            os.path.join(args.dsine_path, "utils", "d2n", "cross.py"),
        )

        intrins_to_intrins_inv = projection.intrins_to_intrins_inv
        get_cam_coords = projection.get_cam_coords
        d2n_tblr = cross_module.d2n_tblr

        print("DSINE utilities loaded successfully!")
    except Exception as e:
        print(f"ERROR: Cannot load DSINE utilities from {args.dsine_path}")
        print(f"  {e}")
        print(f"")
        print(
            f"  Make sure you cloned: git clone https://github.com/baegwangbin/DSINE.git"
        )
        print(f"  Expected files:")
        print(f"    {os.path.join(args.dsine_path, 'utils', 'projection.py')}")
        print(f"    {os.path.join(args.dsine_path, 'utils', 'd2n', 'cross.py')}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target resolution: {args.target_h}x{args.target_w}")
    print(f"Cross-product neighborhood k={args.k}")

    for split in ["train", "val"]:
        img_dir = os.path.join(args.data_root, "leftImg8bit", split)
        disp_dir = os.path.join(args.data_root, "disparity", split)
        out_dir = os.path.join(args.data_root, "normals", split)

        images = sorted(glob.glob(os.path.join(img_dir, "*", "*.png")))
        print(f"\nProcessing {len(images)} images for {split}...")

        skipped = 0
        errors = 0

        for img_path in tqdm(images, desc=split):
            parts = img_path.split(os.sep)
            city = parts[-2]
            fname = parts[-1]
            frame_name = fname.replace("_leftImg8bit.png", "")

            city_out_dir = os.path.join(out_dir, city)
            os.makedirs(city_out_dir, exist_ok=True)
            out_path = os.path.join(city_out_dir, frame_name + "_normal.npy")

            if os.path.exists(out_path) and not args.force:
                skipped += 1
                continue

            try:
                # --- Step 1: Load disparity (16-bit PNG) ---
                disp_path = os.path.join(disp_dir, city, frame_name + "_disparity.png")

                if not os.path.exists(disp_path):
                    errors += 1
                    continue

                raw_disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(
                    np.float32
                )
                orig_h, orig_w = raw_disp.shape

                # --- Step 2: Convert disparity (official formula) ---
                # d = (p - 1) / 256 for p > 0, p = 0 is invalid
                valid_mask = raw_disp > 0
                disp = np.zeros_like(raw_disp)
                disp[valid_mask] = (raw_disp[valid_mask] - 1.0) / 256.0

                # --- Step 3: Compute depth ---
                baseline, fx, fy, cx, cy = load_camera_intrinsics(
                    args.data_root, split, city, frame_name
                )

                depth = np.zeros_like(disp)
                valid_disp = valid_mask & (disp > 0)
                depth[valid_disp] = (baseline * fx) / disp[valid_disp]

                # Clamp to reasonable range
                depth = np.clip(depth, 0, 300.0)

                # Set invalid pixels to 0
                depth[~valid_disp] = 0.0

                # --- Step 4: Resize depth to target resolution ---
                depth_resized = cv2.resize(
                    depth,
                    (target_size[1], target_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                valid_resized = cv2.resize(
                    valid_disp.astype(np.uint8),
                    (target_size[1], target_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                # Zero out invalid after resize
                depth_resized[~valid_resized] = 0.0

                # --- Step 5: Build intrinsics and scale ---
                intrins = build_intrinsics_matrix(fx, fy, cx, cy)
                intrins_scaled = scale_intrinsics(
                    intrins, orig_h, orig_w, target_size[0], target_size[1]
                )

                # Compute inverse intrinsics (DSINE utility)
                intrins_inv = intrins_to_intrins_inv(intrins_scaled)

                # --- Step 6: Convert to tensors ---
                # depth: (1, 1, H, W)
                depth_t = torch.from_numpy(depth_resized).float()
                depth_t = depth_t.unsqueeze(0).unsqueeze(0).to(device)

                # intrins_inv: (1, 3, 3)
                intrins_inv_t = torch.from_numpy(intrins_inv).float()
                intrins_inv_t = intrins_inv_t.unsqueeze(0).to(device)

                # --- Step 7: Unproject to 3D camera coordinates ---
                points = get_cam_coords(intrins_inv_t, depth_t)

                # --- Step 8: Compute normals via cross-product ---
                # Option 1 from DSINE notebook — fast and effective
                with torch.no_grad():
                    normal, normal_valid = d2n_tblr(
                        points, k=args.k, d_min=1e-3, d_max=1000.0
                    )

                # --- Step 9: Save ---
                # normal: (1, 3, H, W), values in [-1, 1]
                normal = normal * normal_valid.float()
                normal_np = normal.squeeze(0).cpu().numpy().astype(np.float32)

                np.save(out_path, normal_np)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n  Error on {frame_name}: {e}")
                elif errors == 6:
                    print(f"\n  Suppressing further error messages...")

        print(
            f"  Completed: {len(images) - skipped - errors} generated, "
            f"{skipped} skipped, {errors} errors"
        )

    print(f"\nDone! Normals saved to: {os.path.join(args.data_root, 'normals')}")
    print(f"\nRetrain with:")
    print(
        f"  python train.py --data_root {args.data_root} --epochs 80 --batch_size 2 "
        f"--img_h {args.target_h} --img_w {args.target_w} --w_seg 4.0 --w_normal 0.5"
    )


if __name__ == "__main__":
    main()

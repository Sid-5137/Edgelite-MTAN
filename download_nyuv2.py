#!/usr/bin/env python3
"""
Download and prepare NYUv2 dataset for MTAN-Lite training.

Uses the preprocessed NYUv2 data from the MTAN repository (Shikun Liu et al.)
which provides 13-class segmentation, true depth, and surface normals.

Usage:
    python download_nyuv2.py --output nyuv2/

This will create:
    nyuv2/
    ├── train/
    │   ├── image/   (795 .npy files)
    │   ├── depth/   (795 .npy files)
    │   ├── label/   (795 .npy files)
    │   └── normal/  (795 .npy files)
    └── val/
        ├── image/   (654 .npy files)
        ├── depth/   (654 .npy files)
        ├── label/   (654 .npy files)
        └── normal/  (654 .npy files)
"""

import os
import argparse
import numpy as np

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False


# MTAN's preprocessed NYUv2 (hosted on multiple mirrors)
DOWNLOAD_URLS = {
    # Original MTAN dropbox link
    "dropbox": "https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=1",
}


def download_and_extract(output_dir):
    """Download MTAN's preprocessed NYUv2 data."""
    os.makedirs(output_dir, exist_ok=True)
    
    zip_path = os.path.join(output_dir, "nyuv2_raw.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading NYUv2 preprocessed data from MTAN repository...")
        print("If the automatic download fails, manually download from:")
        print("  https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa")
        print()
        
        import subprocess
        result = subprocess.run(
            ["wget", "-O", zip_path, DOWNLOAD_URLS["dropbox"]],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"wget failed. Try manual download.")
            print(f"Place the downloaded zip at: {zip_path}")
            return False
    
    print(f"Extracting to {output_dir}...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    
    print("Done extracting.")
    return True


def convert_pt_to_npy(output_dir):
    """
    If data is in .pt format (single large tensors), 
    split into individual .npy files per sample.
    """
    import torch
    
    for split in ["train", "val"]:
        pt_path = os.path.join(output_dir, f"nyuv2_{split}.pt")
        if not os.path.exists(pt_path):
            # Try alternative naming
            pt_path = os.path.join(output_dir, f"{split}_data.pt")
        if not os.path.exists(pt_path):
            continue
        
        print(f"Converting {split} .pt to individual .npy files...")
        data = torch.load(pt_path, map_location="cpu")
        
        # Create directories
        for subdir in ["image", "depth", "label", "normal"]:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
        
        n_samples = data["image"].shape[0]
        for i in range(n_samples):
            idx_str = f"{i:04d}"
            np.save(
                os.path.join(output_dir, split, "image", f"{idx_str}.npy"),
                data["image"][i].numpy()
            )
            np.save(
                os.path.join(output_dir, split, "depth", f"{idx_str}.npy"),
                data["depth"][i].numpy()
            )
            np.save(
                os.path.join(output_dir, split, "label", f"{idx_str}.npy"),
                data["label"][i].numpy()
            )
            np.save(
                os.path.join(output_dir, split, "normal", f"{idx_str}.npy"),
                data["normal"][i].numpy()
            )
        
        print(f"  Saved {n_samples} samples for {split}")


def verify_dataset(output_dir):
    """Verify that the dataset is properly structured."""
    print("\nVerifying dataset structure...")
    
    for split in ["train", "val"]:
        for task in ["image", "depth", "label", "normal"]:
            task_dir = os.path.join(output_dir, split, task)
            if not os.path.isdir(task_dir):
                print(f"  MISSING: {task_dir}")
                continue
            
            files = [f for f in os.listdir(task_dir) if f.endswith(".npy")]
            if len(files) == 0:
                print(f"  EMPTY: {task_dir}")
                continue
            
            # Check first file shape
            sample = np.load(os.path.join(task_dir, sorted(files)[0]))
            print(f"  {split}/{task}: {len(files)} files, shape={sample.shape}, dtype={sample.dtype}")
    
    # Expected counts
    train_dir = os.path.join(output_dir, "train", "image")
    val_dir = os.path.join(output_dir, "val", "image")
    
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        n_train = len([f for f in os.listdir(train_dir) if f.endswith(".npy")])
        n_val = len([f for f in os.listdir(val_dir) if f.endswith(".npy")])
        print(f"\nTotal: {n_train} train, {n_val} val")
        
        if n_train == 795 and n_val == 654:
            print("Dataset verified OK!")
            return True
        else:
            print(f"WARNING: Expected 795 train, 654 val")
    
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NYUv2 for MTAN-Lite")
    parser.add_argument("--output", type=str, default="nyuv2",
                       help="Output directory")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify existing data")
    parser.add_argument("--convert_pt", action="store_true",
                       help="Convert .pt files to individual .npy files")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.output)
    elif args.convert_pt:
        convert_pt_to_npy(args.output)
        verify_dataset(args.output)
    else:
        success = download_and_extract(args.output)
        if success:
            # Check if we need to convert
            convert_pt_to_npy(args.output)
            verify_dataset(args.output)

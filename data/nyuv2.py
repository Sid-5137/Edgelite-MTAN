"""
NYUv2 Dataset for Multi-Task Learning

Uses the preprocessed NYUv2 data from the MTAN repository:
    https://github.com/lorenmt/mtan

Expected structure:
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

Tasks:
    - 13-class semantic segmentation
    - True depth from Kinect sensor
    - Surface normals (ground truth)
"""

import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image, ImageFilter


# NYUv2 13-class names for reference
NYU_CLASS_NAMES = [
    "bed", "books", "ceiling", "chair", "floor",
    "furniture", "objects", "painting", "sofa", "table",
    "tv", "wall", "window"
]

# Category groups for per-category normal analysis
CATEGORY_GROUPS = {
    "floor": [4],
    "wall": [11],
    "furniture": [0, 3, 5, 8, 9],    # bed, chair, furniture, sofa, table
    "structure": [2, 12],              # ceiling, window
    "objects": [1, 6, 7, 10],          # books, objects, painting, tv
}


def _scan_labels(root, split, file_ids, num_classes=13):
    """Scan all label files to find actual class distribution."""
    label_dir = os.path.join(root, split, "label")
    all_unique = set()
    for fid in file_ids:
        label = np.load(os.path.join(label_dir, fid + ".npy"))
        all_unique.update(np.unique(label).tolist())
    
    all_unique_sorted = sorted(all_unique)
    valid_classes = [int(v) for v in all_unique_sorted if 0 <= v < num_classes]
    ignored_vals = [v for v in all_unique_sorted if v < 0 or v >= num_classes]
    
    print(f"[NYUv2 label scan] {split}: unique values = {all_unique_sorted}")
    print(f"  Valid classes (0-{num_classes-1}): {valid_classes} ({len(valid_classes)}/{num_classes})")
    if ignored_vals:
        print(f"  Ignored values: {ignored_vals} (mapped to 255)")
    missing = [c for c in range(num_classes) if c not in valid_classes]
    if missing:
        print(f"  WARNING: Classes never seen: {missing}")
    
    return all_unique_sorted, valid_classes


class NYUv2MultiTask(Dataset):
    """
    NYUv2 dataset for multi-task learning.
    
    Returns same interface as CityscapesMultiTask:
        image: [3, H, W] normalized tensor
        targets: dict with 'depth', 'seg', 'normal', 'valid_depth'
    
    Augmentation (train only):
        1. Random scale jittering (0.5x - 2.0x)
        2. Random crop to target size
        3. Random horizontal flip
        4. Color jitter
        5. Random Gaussian blur
    """
    
    def __init__(
        self,
        root,
        split="train",
        img_size=(288, 384),
        augment=True,
        scale_range=(0.5, 2.0),
        gaussian_sigma=0.5,
    ):
        self.root = root
        self.split = split
        self.img_size = img_size  # (H, W)
        self.augment = augment and (split == "train")
        self.scale_range = scale_range
        self.gaussian_sigma = gaussian_sigma
        
        # Find all files
        img_dir = os.path.join(root, split, "image")
        if not os.path.isdir(img_dir):
            raise RuntimeError(
                f"NYUv2 data not found at {img_dir}. "
                f"Download from: https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa"
            )
        
        self.file_ids = sorted([
            f.replace(".npy", "") 
            for f in os.listdir(img_dir) 
            if f.endswith(".npy")
        ])
        
        if len(self.file_ids) == 0:
            raise RuntimeError(f"No .npy files found in {img_dir}")
        
        print(f"[NYUv2MultiTask] Found {len(self.file_ids)} samples for {split}")
        
        # Scan labels to verify all 13 classes present
        _scan_labels(root, split, self.file_ids, num_classes=13)
        
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
    
    def __len__(self):
        return len(self.file_ids)
    
    def _load_sample(self, idx):
        """Load and normalize all arrays to consistent shapes."""
        fid = self.file_ids[idx]
        
        image = np.load(os.path.join(self.root, self.split, "image", fid + ".npy"))
        depth = np.load(os.path.join(self.root, self.split, "depth", fid + ".npy"))
        label = np.load(os.path.join(self.root, self.split, "label", fid + ".npy"))
        normal = np.load(os.path.join(self.root, self.split, "normal", fid + ".npy"))
        
        # === Normalize image to (C, H, W) and get HWC for PIL ===
        if image.ndim == 3 and image.shape[0] == 3:
            image_hwc = np.transpose(image, (1, 2, 0))
        elif image.ndim == 3 and image.shape[2] == 3:
            image_hwc = image
            image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        img_h, img_w = image.shape[1], image.shape[2]
        
        # === Depth to (1, H, W) ===
        if depth.ndim == 2:
            depth = depth[np.newaxis, ...]
        elif depth.ndim == 3 and depth.shape[2] == 1:
            depth = np.transpose(depth, (2, 0, 1))
        if depth.shape[1] == img_w and depth.shape[2] == img_h and img_h != img_w:
            depth = np.transpose(depth, (0, 2, 1))
        
        # === Label to (H, W) ===
        if label.shape[0] == img_w and label.shape[1] == img_h and img_h != img_w:
            label = label.T
        
        # === Normal to (3, H, W) ===
        if normal.ndim == 3 and normal.shape[2] == 3:
            normal = np.transpose(normal, (2, 0, 1))
        if normal.shape[1] == img_w and normal.shape[2] == img_h and img_h != img_w:
            normal = np.transpose(normal, (0, 2, 1))
        
        # Assertions
        assert depth.shape == (1, img_h, img_w), f"Depth {depth.shape} != (1,{img_h},{img_w})"
        assert label.shape == (img_h, img_w), f"Label {label.shape} != ({img_h},{img_w})"
        assert normal.shape == (3, img_h, img_w), f"Normal {normal.shape} != (3,{img_h},{img_w})"
        
        # === Fix label: map -1 (and any out-of-range) to 255 ===
        label = label.astype(np.int64)
        label[label == -1] = 255
        label[(label < 0) | (label > 12)] = 255
        
        # Valid depth mask
        valid_depth = (depth > 0).astype(np.float32)
        
        # Normalize depth to [0, 1]
        max_depth = 10.0
        depth = np.clip(depth / max_depth, 0, 1).astype(np.float32)
        
        # Convert image to PIL
        if image_hwc.max() <= 1.0:
            image_pil = Image.fromarray((image_hwc * 255).astype(np.uint8))
        else:
            image_pil = Image.fromarray(image_hwc.astype(np.uint8))
        
        # Return depth and valid_depth as 2D (H, W) for easier augmentation
        return image_pil, label, depth[0], normal, valid_depth[0]
    
    def _random_scale_crop(self, image, seg, depth, normals, valid_depth):
        """
        Random scale jittering + random crop.
        Ported from CityscapesMultiTask — crucial for segmentation mIoU.
        
        Args:
            image: PIL Image
            seg: (H, W) numpy int64
            depth: (H, W) numpy float32
            normals: (3, H, W) numpy float32
            valid_depth: (H, W) numpy float32
        """
        H, W = self.img_size
        
        # Random scale
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_h = int(H * scale)
        new_w = int(W * scale)
        
        # Resize all inputs
        image = image.resize((new_w, new_h), Image.BILINEAR)
        seg = np.array(Image.fromarray(seg.astype(np.int32)).resize(
            (new_w, new_h), Image.NEAREST))
        depth = np.array(Image.fromarray(depth).resize(
            (new_w, new_h), Image.BILINEAR))
        valid_depth = np.array(Image.fromarray(valid_depth.astype(np.uint8)).resize(
            (new_w, new_h), Image.NEAREST)).astype(np.float32)
        
        # Resize normals (3, H, W) channel by channel
        n_resized = []
        for c in range(3):
            nc = np.array(Image.fromarray(normals[c]).resize(
                (new_w, new_h), Image.BILINEAR))
            n_resized.append(nc)
        normals = np.stack(n_resized, axis=0)
        # Re-normalize after interpolation
        mag = np.sqrt(np.sum(normals**2, axis=0, keepdims=True))
        normals = normals / np.maximum(mag, 1e-8)
        
        # Pad if scaled smaller than target
        cur_h, cur_w = new_h, new_w
        if cur_h < H or cur_w < W:
            pad_h = max(H - cur_h, 0)
            pad_w = max(W - cur_w, 0)
            
            image = np.array(image)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            image = Image.fromarray(image)
            
            seg = np.pad(seg, ((0, pad_h), (0, pad_w)),
                        mode="constant", constant_values=255)
            depth = np.pad(depth, ((0, pad_h), (0, pad_w)),
                          mode="constant", constant_values=0)
            valid_depth = np.pad(valid_depth, ((0, pad_h), (0, pad_w)),
                                mode="constant", constant_values=0)
            normals = np.pad(normals, ((0, 0), (0, pad_h), (0, pad_w)),
                            mode="constant", constant_values=0)
            
            cur_h = max(cur_h, H)
            cur_w = max(cur_w, W)
        
        # Random crop
        y = random.randint(0, cur_h - H)
        x = random.randint(0, cur_w - W)
        
        image = np.array(image)[y:y+H, x:x+W]
        image = Image.fromarray(image)
        seg = seg[y:y+H, x:x+W]
        depth = depth[y:y+H, x:x+W]
        valid_depth = valid_depth[y:y+H, x:x+W]
        normals = normals[:, y:y+H, x:x+W]
        
        return image, seg, depth, normals, valid_depth
    
    def __getitem__(self, idx):
        image_pil, seg, depth, normals, valid_depth = self._load_sample(idx)
        
        # Resize to target size if needed
        h, w = depth.shape[0], depth.shape[1]
        tH, tW = self.img_size
        if h != tH or w != tW:
            image_pil = image_pil.resize((tW, tH), Image.BILINEAR)
            seg = np.array(Image.fromarray(seg.astype(np.int32)).resize(
                (tW, tH), Image.NEAREST))
            depth = np.array(Image.fromarray(depth).resize(
                (tW, tH), Image.BILINEAR))
            valid_depth = np.array(Image.fromarray(valid_depth.astype(np.uint8)).resize(
                (tW, tH), Image.NEAREST)).astype(np.float32)
            n_resized = []
            for c in range(3):
                nc = np.array(Image.fromarray(normals[c]).resize(
                    (tW, tH), Image.BILINEAR))
                n_resized.append(nc)
            normals = np.stack(n_resized, axis=0)
            mag = np.sqrt(np.sum(normals**2, axis=0, keepdims=True))
            normals = normals / np.maximum(mag, 1e-8)
        
        # Augmentation (train only)
        if self.augment:
            # 1. Random scale + crop (biggest impact on mIoU)
            image_pil, seg, depth, normals, valid_depth = self._random_scale_crop(
                image_pil, seg, depth, normals, valid_depth
            )
            
            # 2. Random horizontal flip
            if random.random() > 0.5:
                image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
                seg = np.fliplr(seg).copy()
                depth = np.fliplr(depth).copy()
                normals = np.flip(normals, axis=2).copy()
                normals[0] = -normals[0]  # flip x-component
                valid_depth = np.fliplr(valid_depth).copy()
            
            # 3. Color jitter
            image_pil = self.color_jitter(image_pil)
            
            # 4. Random Gaussian blur
            if random.random() > 0.5:
                sigma = random.uniform(0.3, 1.5)
                image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
            # Validation: light Gaussian smoothing
            if self.gaussian_sigma > 0:
                image_pil = image_pil.filter(
                    ImageFilter.GaussianBlur(radius=self.gaussian_sigma))
        
        # Convert to tensors
        image_tensor = TF.to_tensor(image_pil)
        image_tensor = self.normalize(image_tensor)
        
        depth_tensor = torch.from_numpy(depth.copy()).unsqueeze(0).float()
        seg_tensor = torch.from_numpy(seg.copy()).long()
        normal_tensor = torch.from_numpy(normals.copy()).float()
        valid_tensor = torch.from_numpy(valid_depth.copy()).unsqueeze(0).float()
        
        targets = {
            "depth": depth_tensor,
            "seg": seg_tensor,
            "normal": normal_tensor,
            "valid_depth": valid_tensor,
        }
        
        return image_tensor, targets


def get_dataloaders(data_root, batch_size=4, num_workers=4, img_size=(288, 384)):
    """Drop-in replacement for cityscapes.get_dataloaders"""
    train_dataset = NYUv2MultiTask(
        root=data_root, split="train", img_size=img_size, augment=True
    )
    val_dataset = NYUv2MultiTask(
        root=data_root, split="val", img_size=img_size, augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


__all__ = [
    "NYUv2MultiTask",
    "get_dataloaders",
    "CATEGORY_GROUPS",
    "NYU_CLASS_NAMES",
]

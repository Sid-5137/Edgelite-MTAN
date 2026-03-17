"""
Cityscapes Dataset for Multi-Task Learning
Enhanced augmentation pipeline for stronger segmentation performance.

Key improvements over previous version:
- Random scale jittering (0.5x - 2.0x) with random crop
- Random Gaussian blur
- Proper multi-scale training support
- Class frequency weights for balanced CE loss
"""

import os
import glob
import numpy as np
from PIL import Image, ImageFilter
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# Cityscapes label mapping: 34 classes -> 19 training classes + ignore
CITYSCAPES_LABEL_MAP = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,  # road
    8: 1,  # sidewalk
    9: 255,
    10: 255,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,  # pole
    18: 255,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255,
    30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 255,
}

# 5-class grouping for radar chart analysis
CATEGORY_GROUPS = {
    "roadway": [0],
    "walkway": [1],
    "building": [2, 3, 4],
    "vegetation": [8, 9],
    "background": [5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18],
}

# Cityscapes class frequencies (approximate, from training set)
# Used for inverse-frequency class weighting in cross-entropy
# Order: road, sidewalk, building, wall, fence, pole, traffic_light,
#        traffic_sign, vegetation, terrain, sky, person, rider, car,
#        truck, bus, train, motorcycle, bicycle
CITYSCAPES_CLASS_WEIGHTS = torch.FloatTensor(
    [
        0.8373,
        0.9180,
        0.8660,
        1.0345,
        1.0166,
        0.9969,
        0.9972,
        1.0110,
        0.9093,
        0.9733,
        0.9469,
        1.0144,
        0.9828,
        0.9468,
        1.0082,
        0.9902,
        1.0207,
        1.0175,
        1.0039,
    ]
)

# More aggressive inverse-frequency weights for better rare-class mIoU
CITYSCAPES_CLASS_WEIGHTS_STRONG = torch.FloatTensor(
    [
        0.05,  # road (very common)
        0.15,  # sidewalk
        0.10,  # building
        0.80,  # wall (rare)
        0.70,  # fence
        1.50,  # pole (thin, rare)
        2.00,  # traffic light (tiny)
        1.80,  # traffic sign (small)
        0.10,  # vegetation
        0.30,  # terrain
        0.10,  # sky
        1.20,  # person
        2.50,  # rider (rare)
        0.20,  # car
        1.50,  # truck (rare)
        1.80,  # bus (rare)
        3.00,  # train (very rare)
        2.50,  # motorcycle (rare)
        1.50,  # bicycle
    ]
)


def disparity_to_depth(disparity, baseline=0.209313, focal_length=2262.52):
    """Convert Cityscapes disparity to metric depth."""
    valid = disparity > 0
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = (baseline * focal_length) / disparity[valid]
    return depth, valid


class CityscapesMultiTask(Dataset):
    """
    Cityscapes dataset for multi-task learning with enhanced augmentation.

    Augmentation pipeline (train only):
        1. Random scale jittering (0.5x - 2.0x)
        2. Random crop to target size
        3. Random horizontal flip
        4. Color jitter (brightness, contrast, saturation, hue)
        5. Random Gaussian blur
    """

    def __init__(
        self,
        root,
        split="train",
        img_size=(256, 512),
        augment=True,
        gaussian_sigma=0.5,
        scale_range=(0.5, 2.0),
    ):
        self.root = root
        self.split = split
        self.img_size = img_size  # (H, W)
        self.augment = augment and (split == "train")
        self.gaussian_sigma = gaussian_sigma
        self.scale_range = scale_range

        # Find all image files
        img_dir = os.path.join(root, "leftImg8bit", split)
        self.images = sorted(glob.glob(os.path.join(img_dir, "*", "*_leftImg8bit.png")))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {img_dir}.")

        # Check Omnidata normals
        normal_dir = os.path.join(root, "normals_omnidata", split)
        self.has_omnidata_normals = os.path.isdir(normal_dir)

        if self.has_omnidata_normals:
            normal_files = glob.glob(os.path.join(normal_dir, "*", "*.npy"))
            print(
                f"[CityscapesMultiTask] Found {len(self.images)} images, "
                f"{len(normal_files)} Omnidata normals for {split}"
            )
            if len(normal_files) == 0:
                self.has_omnidata_normals = False
                print(
                    f"  WARNING: normals_omnidata/ empty. Using depth-derived normals."
                )
        else:
            print(f"[CityscapesMultiTask] Found {len(self.images)} images for {split}")
            print(
                f"  Run 'python generate_normals.py --data_root {root}' for Omnidata normals."
            )

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Color jitter transform
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

    def _get_paths(self, img_path):
        parts = img_path.split(os.sep)
        city = parts[-2]
        fname = parts[-1]
        base = fname.replace("_leftImg8bit.png", "")

        seg_path = os.path.join(
            self.root, "gtFine", self.split, city, base + "_gtFine_labelIds.png"
        )
        disp_path = os.path.join(
            self.root, "disparity", self.split, city, base + "_disparity.png"
        )
        normal_path = os.path.join(
            self.root, "normals_omnidata", self.split, city, base + "_normal.npy"
        )

        return seg_path, disp_path, normal_path

    def __len__(self):
        return len(self.images)

    def _load_normal(self, normal_path, depth, img_size):
        """Load normals from Omnidata .npy or compute from depth as fallback."""
        if self.has_omnidata_normals and os.path.exists(normal_path):
            normals = np.load(normal_path).astype(np.float32)
            if normals.shape[1] != img_size[0] or normals.shape[2] != img_size[1]:
                normals_t = torch.from_numpy(normals).unsqueeze(0)
                normals_t = torch.nn.functional.interpolate(
                    normals_t, size=img_size, mode="bilinear", align_corners=False
                )
                normals_t = torch.nn.functional.normalize(normals_t, p=2, dim=1)
                normals = normals_t.squeeze(0).numpy()
            return normals
        else:
            # Fallback: compute from depth
            dz_dx = np.zeros_like(depth)
            dz_dy = np.zeros_like(depth)
            dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2.0
            dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2.0
            dz_dx[:, 0] = depth[:, 1] - depth[:, 0]
            dz_dx[:, -1] = depth[:, -1] - depth[:, -2]
            dz_dy[0, :] = depth[1, :] - depth[0, :]
            dz_dy[-1, :] = depth[-1, :] - depth[-2, :]
            normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=0)
            mag = np.sqrt(np.sum(normals**2, axis=0, keepdims=True))
            normals = normals / np.maximum(mag, 1e-8)
            return normals.astype(np.float32)

    def _random_scale_crop(self, image, seg, depth, normals, valid_depth):
        """
        Random scale jittering + random crop.
        Crucial for segmentation performance.
        """
        H, W = self.img_size

        # Random scale
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_h = int(H * scale)
        new_w = int(W * scale)

        # Resize all inputs to scaled size
        image = image.resize((new_w, new_h), Image.BILINEAR)
        seg = np.array(
            Image.fromarray(seg.astype(np.int32)).resize((new_w, new_h), Image.NEAREST)
        )

        depth_pil = Image.fromarray(depth)
        depth = np.array(depth_pil.resize((new_w, new_h), Image.BILINEAR))

        valid_pil = Image.fromarray(valid_depth.astype(np.uint8))
        valid_depth = np.array(valid_pil.resize((new_w, new_h), Image.NEAREST)).astype(
            bool
        )

        # Resize normals (3, H, W) -> need to handle channel-first
        normals_hwc = np.transpose(normals, (1, 2, 0))  # (H, W, 3)
        normals_pil_r = Image.fromarray(normals_hwc[:, :, 0]).resize(
            (new_w, new_h), Image.BILINEAR
        )
        normals_pil_g = Image.fromarray(normals_hwc[:, :, 1]).resize(
            (new_w, new_h), Image.BILINEAR
        )
        normals_pil_b = Image.fromarray(normals_hwc[:, :, 2]).resize(
            (new_w, new_h), Image.BILINEAR
        )
        normals = np.stack(
            [np.array(normals_pil_r), np.array(normals_pil_g), np.array(normals_pil_b)],
            axis=0,
        )
        # Re-normalize after interpolation
        mag = np.sqrt(np.sum(normals**2, axis=0, keepdims=True))
        normals = normals / np.maximum(mag, 1e-8)

        # Random crop (or pad if scaled smaller than target)
        cur_h, cur_w = new_h, new_w

        if cur_h < H or cur_w < W:
            # Pad if too small
            pad_h = max(H - cur_h, 0)
            pad_w = max(W - cur_w, 0)

            image = np.array(image)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            image = Image.fromarray(image)

            seg = np.pad(
                seg, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255
            )
            depth = np.pad(
                depth, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )
            valid_depth = np.pad(
                valid_depth,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=False,
            )
            normals = np.pad(
                normals,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

            cur_h = max(cur_h, H)
            cur_w = max(cur_w, W)

        # Random crop position
        y = random.randint(0, cur_h - H)
        x = random.randint(0, cur_w - W)

        image = np.array(image)[y : y + H, x : x + W]
        image = Image.fromarray(image)
        seg = seg[y : y + H, x : x + W]
        depth = depth[y : y + H, x : x + W]
        valid_depth = valid_depth[y : y + H, x : x + W]
        normals = normals[:, y : y + H, x : x + W]

        return image, seg, depth, normals, valid_depth

    def __getitem__(self, idx):
        img_path = self.images[idx]
        seg_path, disp_path, normal_path = self._get_paths(img_path)

        # Load image at ORIGINAL resolution first (for better scale augmentation)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load segmentation at original resolution
        seg = np.array(Image.open(seg_path), dtype=np.int64)

        # Map to train IDs
        seg_mapped = np.full_like(seg, 255)
        for orig_id, train_id in CITYSCAPES_LABEL_MAP.items():
            if orig_id >= 0:
                seg_mapped[seg == orig_id] = train_id

        # Load depth
        if os.path.exists(disp_path):
            disp_img = Image.open(disp_path)
            disp = np.array(disp_img, dtype=np.float32) / 256.0
            depth, valid_depth = disparity_to_depth(disp)
        else:
            depth = np.zeros((orig_h, orig_w), dtype=np.float32)
            valid_depth = np.zeros((orig_h, orig_w), dtype=bool)

        max_depth = 80.0
        depth = np.clip(depth / max_depth, 0, 1).astype(np.float32)

        # Resize everything to target size first
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        seg_mapped = np.array(
            Image.fromarray(seg_mapped.astype(np.int32)).resize(
                (self.img_size[1], self.img_size[0]), Image.NEAREST
            )
        )
        depth = np.array(
            Image.fromarray(depth).resize(
                (self.img_size[1], self.img_size[0]), Image.BILINEAR
            )
        )
        valid_depth = np.array(
            Image.fromarray(valid_depth.astype(np.uint8)).resize(
                (self.img_size[1], self.img_size[0]), Image.NEAREST
            )
        ).astype(bool)

        # Load normals
        normals = self._load_normal(normal_path, depth, self.img_size)

        # Augmentation
        if self.augment:
            # 1. Random scale + crop (most important for seg)
            image, seg_mapped, depth, normals, valid_depth = self._random_scale_crop(
                image, seg_mapped, depth, normals, valid_depth
            )

            # 2. Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                seg_mapped = np.fliplr(seg_mapped).copy()
                depth = np.fliplr(depth).copy()
                normals = np.fliplr(normals).copy()
                normals[0] = -normals[0]
                valid_depth = np.fliplr(valid_depth).copy()

            # 3. Color jitter
            image = self.color_jitter(image)

            # 4. Random Gaussian blur (helps seg generalization)
            if random.random() > 0.5:
                sigma = random.uniform(0.3, 1.5)
                image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
            # Validation: just apply light Gaussian smoothing
            if self.gaussian_sigma > 0:
                image = image.filter(
                    ImageFilter.GaussianBlur(radius=self.gaussian_sigma)
                )

        # Convert to tensors
        image = TF.to_tensor(image)
        image = self.normalize(image)

        depth = torch.from_numpy(depth).unsqueeze(0).float()
        seg_mapped = torch.from_numpy(seg_mapped.copy()).long()
        normals = torch.from_numpy(normals.copy()).float()
        valid_depth = torch.from_numpy(valid_depth.astype(np.float32)).unsqueeze(0)

        targets = {
            "depth": depth,
            "seg": seg_mapped,
            "normal": normals,
            "valid_depth": valid_depth,
        }

        return image, targets


def get_dataloaders(data_root, batch_size=2, num_workers=4, img_size=(256, 512)):
    train_dataset = CityscapesMultiTask(
        root=data_root, split="train", img_size=img_size, augment=True
    )
    val_dataset = CityscapesMultiTask(
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


def get_class_weights(mode="strong"):
    """Return class weights for weighted cross-entropy."""
    if mode == "strong":
        return CITYSCAPES_CLASS_WEIGHTS_STRONG
    else:
        return CITYSCAPES_CLASS_WEIGHTS


__all__ = [
    "CityscapesMultiTask",
    "get_dataloaders",
    "get_class_weights",
    "CATEGORY_GROUPS",
    "CITYSCAPES_CLASS_WEIGHTS_STRONG",
]

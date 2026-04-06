"""
Multi-Task Loss for EdgeLite-MTAN

Segmentation uses DiceFocal Loss:
    L_seg = 0.5 * Focal Loss + 0.5 * Dice Loss

    - Focal Loss (Lin et al., 2017): focuses on hard pixels, handles class imbalance
    - Dice Loss: optimizes region-level overlap per class, directly improves mIoU
    - Combined: pixel-level + region-level, used by nnU-Net as default
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleInvariantDepthLoss(nn.Module):
    """
    Scale-invariant log loss for monocular depth estimation (Eigen et al., 2014).
    Allows global scale shift while penalizing per-pixel relative errors.
    """

    def __init__(self, si_lambda=0.5):
        super().__init__()
        self.si_lambda = si_lambda

    def forward(self, pred, target, valid_mask):
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        target = torch.clamp(target, min=eps)

        diff = torch.log(pred) - torch.log(target)
        diff = diff * valid_mask

        n = torch.sum(valid_mask) + eps
        loss = torch.sum(diff**2) / n - self.si_lambda * (torch.sum(diff) ** 2) / (n**2)
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard examples (boundaries, small objects).
    """

    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        B, C, H, W = logits.shape

        valid_mask = targets != self.ignore_index

        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets_onehot = targets_safe.unsqueeze(1)

        log_pt = log_probs.gather(1, targets_onehot).squeeze(1)
        pt = probs.gather(1, targets_onehot).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets_safe]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_pt
        loss = loss * valid_mask.float()

        num_valid = valid_mask.sum().float() + 1e-6
        loss = loss.sum() / num_valid

        return loss


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for segmentation.

    Optimizes per-class region overlap (Dice coefficient), which directly
    corresponds to mIoU. Handles class imbalance inherently since each
    class contributes equally to the final loss.

    Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
    Loss = 1 - mean(Dice per class)
    """

    def __init__(self, num_classes=19, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] raw class scores
            targets: [B, H, W] ground truth class indices
        """
        B, C, H, W = logits.shape

        valid_mask = targets != self.ignore_index

        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # One-hot encode targets: [B, H, W] -> [B, C, H, W]
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets_safe.unsqueeze(1), 1.0)

        # Apply valid mask — zero out ignored pixels in both pred and gt
        valid_mask_4d = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        probs = probs * valid_mask_4d
        targets_onehot = targets_onehot * valid_mask_4d

        # Flatten spatial dims: [B, C, H*W]
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets_onehot.view(B, C, -1)

        # Per-class Dice: 2*intersection / (pred_sum + gt_sum)
        intersection = (probs_flat * targets_flat).sum(dim=2)  # [B, C]
        pred_sum = probs_flat.sum(dim=2)
        gt_sum = targets_flat.sum(dim=2)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            pred_sum + gt_sum + self.smooth
        )

        # Only average over classes that exist in the batch
        class_exists = (gt_sum > 0).float()  # [B, C]
        num_classes_present = class_exists.sum(dim=1).clamp(min=1)  # [B]

        dice_score = (dice_per_class * class_exists).sum(
            dim=1
        ) / num_classes_present  # [B]

        loss = 1.0 - dice_score.mean()

        return loss


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss (DiceFocal).

    L_seg = dice_weight * DiceLoss + focal_weight * FocalLoss

    - Dice: region-level, directly optimizes per-class overlap
    - Focal: pixel-level, focuses on hard examples and rare classes
    - Used by nnU-Net as their default segmentation loss
    """

    def __init__(
        self,
        num_classes=19,
        gamma=2.0,
        alpha=None,
        ignore_index=255,
        dice_weight=0.5,
        focal_weight=0.5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)

    def forward(self, logits, targets):
        l_dice = self.dice_loss(logits, targets)
        l_focal = self.focal_loss(logits, targets)
        return self.dice_weight * l_dice + self.focal_weight * l_focal


class CosineSimilarityNormalLoss(nn.Module):
    """
    Cosine similarity loss for surface normal estimation.
    Standard across DSINE, Omnidata, MTAN, Eigen & Fergus.

    L = 1 - mean(pred · gt)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid_mask=None):
        cos_sim = torch.sum(pred * target, dim=1, keepdim=True)

        if valid_mask is not None:
            cos_sim = cos_sim * valid_mask
            n = torch.sum(valid_mask) + 1e-6
            loss = 1.0 - torch.sum(cos_sim) / n
        else:
            loss = 1.0 - torch.mean(cos_sim)
        return loss


# Cityscapes inverse-frequency alpha weights for focal loss (19 classes)
CITYSCAPES_FOCAL_ALPHA = torch.FloatTensor(
    [
        0.3,  # road (very common)
        0.5,  # sidewalk
        0.4,  # building
        1.5,  # wall (rare)
        1.3,  # fence
        2.0,  # pole (thin, rare)
        2.5,  # traffic light (tiny)
        2.0,  # traffic sign (small)
        0.4,  # vegetation
        0.8,  # terrain
        0.4,  # sky
        1.5,  # person
        3.0,  # rider (rare)
        0.5,  # car
        2.0,  # truck (rare)
        2.5,  # bus (rare)
        3.5,  # train (very rare)
        3.0,  # motorcycle (rare)
        2.0,  # bicycle
    ]
)

# NYUv2 inverse-frequency alpha weights for focal loss (13 classes)
NYUV2_FOCAL_ALPHA = torch.FloatTensor(
    [
        1.5,  # bed
        2.0,  # books (small)
        0.5,  # ceiling (common)
        1.0,  # chair
        0.4,  # floor (very common)
        1.0,  # furniture
        1.5,  # objects
        2.0,  # painting (small)
        1.2,  # sofa
        1.0,  # table
        2.5,  # tv (small)
        0.3,  # wall (very common)
        1.5,  # window
    ]
)

# Registry for dataset-specific alpha weights
FOCAL_ALPHA_REGISTRY = {
    19: CITYSCAPES_FOCAL_ALPHA,
    13: NYUV2_FOCAL_ALPHA,
}


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss.

    L = w_depth * ScaleInvariantDepth + w_seg * DiceFocal + w_normal * CosineSimilarity
    """

    def __init__(
        self,
        num_classes=19,
        mode="fixed",
        weight_depth=1.0,
        weight_seg=2.0,
        weight_normal=1.0,
        use_focal=True,
        focal_gamma=2.0,
        class_weights=None,
    ):
        super().__init__()

        self.mode = mode
        self.depth_loss = ScaleInvariantDepthLoss(si_lambda=0.5)
        self.normal_loss = CosineSimilarityNormalLoss()

        # Segmentation loss
        if use_focal:
            if class_weights is not None:
                alpha = class_weights
            elif num_classes in FOCAL_ALPHA_REGISTRY:
                alpha = FOCAL_ALPHA_REGISTRY[num_classes]
            else:
                # Uniform weights for unknown class counts
                alpha = torch.ones(num_classes)
            self.seg_loss = DiceFocalLoss(
                num_classes=num_classes,
                gamma=focal_gamma,
                alpha=alpha,
                ignore_index=255,
                dice_weight=0.5,
                focal_weight=0.5,
            )
        else:
            if class_weights is not None:
                self.seg_loss = nn.CrossEntropyLoss(
                    weight=class_weights, ignore_index=255
                )
            else:
                self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)

        if mode == "fixed":
            self.weight_depth = weight_depth
            self.weight_seg = weight_seg
            self.weight_normal = weight_normal
        elif mode == "uncertainty":
            self.log_var_depth = nn.Parameter(torch.zeros(1))
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_normal = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, predictions, targets):
        l_depth = self.depth_loss(
            predictions["depth"], targets["depth"], targets["valid_depth"]
        )
        l_seg = self.seg_loss(predictions["seg"], targets["seg"])
        l_normal = self.normal_loss(
            predictions["normal"], targets["normal"], targets["valid_depth"]
        )

        if self.mode == "fixed":
            total = (
                self.weight_depth * l_depth
                + self.weight_seg * l_seg
                + self.weight_normal * l_normal
            )

            loss_dict = {
                "loss_total": total.item(),
                "loss_depth": l_depth.item(),
                "loss_seg": l_seg.item(),
                "loss_normal": l_normal.item(),
                "weight_depth": self.weight_depth,
                "weight_seg": self.weight_seg,
                "weight_normal": self.weight_normal,
            }
        else:
            w_depth = 0.5 * torch.exp(-self.log_var_depth)
            w_seg = torch.exp(-self.log_var_seg)
            w_normal = 0.5 * torch.exp(-self.log_var_normal)

            total = (
                w_depth * l_depth
                + 0.5 * self.log_var_depth
                + w_seg * l_seg
                + 0.5 * self.log_var_seg
                + w_normal * l_normal
                + 0.5 * self.log_var_normal
            )

            loss_dict = {
                "loss_total": total.item(),
                "loss_depth": l_depth.item(),
                "loss_seg": l_seg.item(),
                "loss_normal": l_normal.item(),
                "weight_depth": w_depth.item(),
                "weight_seg": w_seg.item(),
                "weight_normal": w_normal.item(),
            }

        return total, loss_dict


__all__ = [
    "MultiTaskLoss",
    "FocalLoss",
    "DiceLoss",
    "DiceFocalLoss",
    "CITYSCAPES_FOCAL_ALPHA",
    "NYUV2_FOCAL_ALPHA",
    "FOCAL_ALPHA_REGISTRY",
]

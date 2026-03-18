"""
Evaluation Metrics for EdgeLite-MTAN

Depth:   RMSE, AbsRel, delta1 threshold accuracy
Seg:     mIoU (global accumulation across all batches)
Normals: MAE, MedAE, WAE (depth-weighted), PFE (plane fitting error)
         + per-category breakdown for radar charts
"""

import numpy as np
import torch


# ============================================================
# DEPTH METRICS
# ============================================================


def depth_metrics(pred, target, valid_mask, max_depth=80.0):
    """
    Compute depth evaluation metrics.

    Args:
        pred: [B, 1, H, W] predicted depth (0-1 range, normalized)
        target: [B, 1, H, W] ground truth depth (0-1 range)
        valid_mask: [B, 1, H, W] valid pixel mask

    Returns:
        dict with 'rmse', 'absrel', 'delta1'
    """
    pred = pred.detach().cpu().numpy() * max_depth
    target = target.detach().cpu().numpy() * max_depth
    valid = valid_mask.detach().cpu().numpy().astype(bool)

    pred_valid = pred[valid]
    target_valid = target[valid]

    if len(pred_valid) == 0:
        return {"rmse": 0.0, "absrel": 0.0, "delta1": 0.0}

    target_valid = np.maximum(target_valid, 1e-6)
    pred_valid = np.maximum(pred_valid, 1e-6)

    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    absrel = np.mean(np.abs(pred_valid - target_valid) / target_valid)
    ratio = np.maximum(pred_valid / target_valid, target_valid / pred_valid)
    delta1 = np.mean(ratio < 1.25)

    return {"rmse": rmse, "absrel": absrel, "delta1": delta1}


# ============================================================
# SEGMENTATION METRICS
# ============================================================


def segmentation_metrics(pred, target, num_classes=19):
    """
    Compute per-batch intersection and union for global mIoU.

    Returns raw intersection/union counts — MetricAccumulator sums
    them globally and computes mIoU once at epoch end.

    Args:
        pred: [B, C, H, W] class logits
        target: [B, H, W] ground truth labels (0 to C-1, 255=ignore)

    Returns:
        dict with 'intersection' (19,), 'union' (19,)
    """
    pred_labels = pred.argmax(dim=1).detach().cpu()  # [B, H, W]
    target_cpu = target.detach().cpu()
    valid = target_cpu != 255

    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for cls in range(num_classes):
        pred_cls = (pred_labels == cls) & valid
        target_cls = (target_cpu == cls) & valid

        intersection[cls] = (pred_cls & target_cls).sum().float()
        union[cls] = (pred_cls | target_cls).sum().float()

    return {
        "intersection": intersection.numpy(),
        "union": union.numpy(),
    }


# ============================================================
# SURFACE NORMAL METRICS
# ============================================================


def angular_error(pred, target, valid_mask=None):
    """
    Compute per-pixel angular error between predicted and GT normals.
    """
    cos_sim = torch.sum(pred * target, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.acos(cos_sim) * (180.0 / np.pi)
    return angles


def normal_metrics(
    pred, target, valid_mask, depth=None, seg_labels=None, category_groups=None
):
    """
    Compute surface normal evaluation metrics.
    """
    angles = angular_error(pred, target)
    valid = valid_mask.squeeze(1).detach().cpu().numpy().astype(bool)
    angles_np = angles.detach().cpu().numpy()

    valid_angles = angles_np[valid]

    results = {}

    if len(valid_angles) == 0:
        results["mae"] = 0.0
        results["medae"] = 0.0
        results["wae"] = 0.0
        results["pfe"] = 0.0
        return results

    results["mae"] = float(np.mean(valid_angles))
    results["medae"] = float(np.median(valid_angles))

    if depth is not None:
        depth_np = depth.squeeze(1).detach().cpu().numpy()
        weights = np.zeros_like(depth_np)
        depth_valid = depth_np > 1e-6
        weights[depth_valid] = 1.0 / depth_np[depth_valid]
        weights = weights * valid.astype(np.float32)

        total_weight = np.sum(weights)
        if total_weight > 0:
            results["wae"] = float(np.sum(angles_np * weights) / total_weight)
        else:
            results["wae"] = results["mae"]
    else:
        results["wae"] = results["mae"]

    flat_threshold = 15.0
    flat_mask = valid_angles < flat_threshold
    if np.sum(flat_mask) > 10:
        results["pfe"] = float(np.std(valid_angles[flat_mask]) / 100.0)
    else:
        results["pfe"] = 0.0

    if seg_labels is not None and category_groups is not None:
        seg_np = seg_labels.detach().cpu().numpy()

        results["per_category_mae"] = {}
        results["per_category_wae"] = {}

        for cat_name, label_ids in category_groups.items():
            cat_mask = np.zeros_like(valid, dtype=bool)
            for lid in label_ids:
                cat_mask |= seg_np == lid

            combined_mask = cat_mask & valid
            cat_angles = angles_np[combined_mask]

            if len(cat_angles) > 0:
                results["per_category_mae"][cat_name] = float(np.mean(cat_angles))

                if depth is not None:
                    cat_weights = weights[combined_mask]
                    w_sum = np.sum(cat_weights)
                    if w_sum > 0:
                        results["per_category_wae"][cat_name] = float(
                            np.sum(cat_angles * cat_weights[: len(cat_angles)]) / w_sum
                        )
                    else:
                        results["per_category_wae"][cat_name] = results[
                            "per_category_mae"
                        ][cat_name]
                else:
                    results["per_category_wae"][cat_name] = results["per_category_mae"][
                        cat_name
                    ]
            else:
                results["per_category_mae"][cat_name] = 0.0
                results["per_category_wae"][cat_name] = 0.0

    return results


class MetricAccumulator:
    """
    Accumulate metrics over batches and compute epoch-level results.

    Key fix: mIoU is computed from globally accumulated intersection/union,
    not averaged from per-batch mIoU values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.depth_metrics = {"rmse": [], "absrel": [], "delta1": []}
        self.normal_metrics = {"mae": [], "medae": [], "wae": [], "pfe": []}
        self.per_category_mae = {}
        self.per_category_wae = {}

        # Global accumulation for segmentation
        self.seg_intersection = np.zeros(19)
        self.seg_union = np.zeros(19)

    def update(self, d_metrics, s_metrics, n_metrics):
        """Add batch metrics."""
        for k, v in d_metrics.items():
            if k in self.depth_metrics:
                self.depth_metrics[k].append(v)

        # Accumulate intersection/union globally
        if "intersection" in s_metrics:
            self.seg_intersection += s_metrics["intersection"]
            self.seg_union += s_metrics["union"]

        for k, v in n_metrics.items():
            if k in self.normal_metrics:
                self.normal_metrics[k].append(v)

        if "per_category_mae" in n_metrics:
            for cat, val in n_metrics["per_category_mae"].items():
                if cat not in self.per_category_mae:
                    self.per_category_mae[cat] = []
                self.per_category_mae[cat].append(val)

        if "per_category_wae" in n_metrics:
            for cat, val in n_metrics["per_category_wae"].items():
                if cat not in self.per_category_wae:
                    self.per_category_wae[cat] = []
                self.per_category_wae[cat].append(val)

    def compute(self):
        """Compute epoch averages."""
        results = {}

        for k, v in self.depth_metrics.items():
            results[f"depth/{k}"] = np.mean(v) if v else 0.0

        # Compute global mIoU from accumulated intersection/union
        iou_per_class = []
        for c in range(19):
            if self.seg_union[c] > 0:
                iou_per_class.append(self.seg_intersection[c] / self.seg_union[c])
        results["seg/miou"] = np.mean(iou_per_class) if iou_per_class else 0.0

        for k, v in self.normal_metrics.items():
            results[f"normal/{k}"] = np.mean(v) if v else 0.0

        results["per_category_mae"] = {
            cat: np.mean(vals) for cat, vals in self.per_category_mae.items() if vals
        }
        results["per_category_wae"] = {
            cat: np.mean(vals) for cat, vals in self.per_category_wae.items() if vals
        }

        return results

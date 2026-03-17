"""
Evaluation Metrics for EdgeLite-MTAN

Depth:   RMSE, AbsRel, delta1 threshold accuracy
Seg:     mIoU
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
        return {'rmse': 0.0, 'absrel': 0.0, 'delta1': 0.0}
    
    # Avoid division by zero
    target_valid = np.maximum(target_valid, 1e-6)
    pred_valid = np.maximum(pred_valid, 1e-6)
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    
    # AbsRel
    absrel = np.mean(np.abs(pred_valid - target_valid) / target_valid)
    
    # Delta1: % of pixels where max(pred/gt, gt/pred) < 1.25
    ratio = np.maximum(pred_valid / target_valid, target_valid / pred_valid)
    delta1 = np.mean(ratio < 1.25)
    
    return {'rmse': rmse, 'absrel': absrel, 'delta1': delta1}


# ============================================================
# SEGMENTATION METRICS
# ============================================================

def segmentation_metrics(pred, target, num_classes=19):
    """
    Compute mIoU for semantic segmentation.
    
    Args:
        pred: [B, C, H, W] class logits
        target: [B, H, W] ground truth labels (0 to C-1, 255=ignore)
        
    Returns:
        dict with 'miou', 'iou_per_class'
    """
    pred_labels = pred.argmax(dim=1).detach().cpu().numpy()  # [B, H, W]
    target_np = target.detach().cpu().numpy()
    
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred_labels == cls)
        target_cls = (target_np == cls)
        valid = (target_np != 255)
        
        pred_cls = pred_cls & valid
        target_cls = target_cls & valid
        
        intersection = np.sum(pred_cls & target_cls)
        union = np.sum(pred_cls | target_cls)
        
        if union > 0:
            iou_per_class.append(intersection / union)
    
    miou = np.mean(iou_per_class) if iou_per_class else 0.0
    
    return {'miou': miou, 'iou_per_class': iou_per_class}


# ============================================================
# SURFACE NORMAL METRICS
# ============================================================

def angular_error(pred, target, valid_mask=None):
    """
    Compute per-pixel angular error between predicted and GT normals.
    
    Args:
        pred: [B, 3, H, W] predicted normals (L2-normalized)
        target: [B, 3, H, W] GT normals (L2-normalized)
        valid_mask: [B, 1, H, W] optional valid pixel mask
        
    Returns:
        angles: [B, H, W] angular error in degrees per pixel
    """
    # Dot product per pixel
    cos_sim = torch.sum(pred * target, dim=1)  # [B, H, W]
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.acos(cos_sim) * (180.0 / np.pi)  # Convert to degrees
    
    return angles


def normal_metrics(pred, target, valid_mask, depth=None, seg_labels=None, 
                   category_groups=None):
    """
    Compute surface normal evaluation metrics.
    
    Args:
        pred: [B, 3, H, W] predicted normals
        target: [B, 3, H, W] GT normals
        valid_mask: [B, 1, H, W] valid mask
        depth: [B, 1, H, W] depth for weighted angular error (optional)
        seg_labels: [B, H, W] segmentation labels for per-category (optional)
        category_groups: dict mapping category names to label lists (optional)
        
    Returns:
        dict with 'mae', 'medae', 'wae', 'pfe', and per-category breakdowns
    """
    angles = angular_error(pred, target)  # [B, H, W]
    valid = valid_mask.squeeze(1).detach().cpu().numpy().astype(bool)  # [B, H, W]
    angles_np = angles.detach().cpu().numpy()
    
    valid_angles = angles_np[valid]
    
    results = {}
    
    if len(valid_angles) == 0:
        results['mae'] = 0.0
        results['medae'] = 0.0
        results['wae'] = 0.0
        results['pfe'] = 0.0
        return results
    
    # Mean Angular Error
    results['mae'] = float(np.mean(valid_angles))
    
    # Median Angular Error (less sensitive to outliers)
    results['medae'] = float(np.median(valid_angles))
    
    # Depth-Weighted Angular Error
    if depth is not None:
        depth_np = depth.squeeze(1).detach().cpu().numpy()  # [B, H, W]
        # Weight = 1/depth (closer pixels get more weight)
        weights = np.zeros_like(depth_np)
        depth_valid = depth_np > 1e-6
        weights[depth_valid] = 1.0 / depth_np[depth_valid]
        weights = weights * valid.astype(np.float32)
        
        total_weight = np.sum(weights)
        if total_weight > 0:
            results['wae'] = float(np.sum(angles_np * weights) / total_weight)
        else:
            results['wae'] = results['mae']
    else:
        results['wae'] = results['mae']
    
    # Plane Fitting Error: std of angular error on flat surfaces
    # (lower = better planar consistency)
    flat_threshold = 15.0  # degrees -- pixels with low error are "planar"
    flat_mask = valid_angles < flat_threshold
    if np.sum(flat_mask) > 10:
        results['pfe'] = float(np.std(valid_angles[flat_mask]) / 100.0)
    else:
        results['pfe'] = 0.0
    
    # Per-category breakdown for radar charts
    if seg_labels is not None and category_groups is not None:
        seg_np = seg_labels.detach().cpu().numpy()  # [B, H, W]
        
        results['per_category_mae'] = {}
        results['per_category_wae'] = {}
        
        for cat_name, label_ids in category_groups.items():
            cat_mask = np.zeros_like(valid, dtype=bool)
            for lid in label_ids:
                cat_mask |= (seg_np == lid)
            
            combined_mask = cat_mask & valid
            cat_angles = angles_np[combined_mask]
            
            if len(cat_angles) > 0:
                results['per_category_mae'][cat_name] = float(np.mean(cat_angles))
                
                # WAE per category
                if depth is not None:
                    cat_weights = weights[combined_mask]
                    w_sum = np.sum(cat_weights)
                    if w_sum > 0:
                        results['per_category_wae'][cat_name] = float(
                            np.sum(cat_angles * cat_weights[:len(cat_angles)]) / w_sum
                        )
                    else:
                        results['per_category_wae'][cat_name] = results['per_category_mae'][cat_name]
                else:
                    results['per_category_wae'][cat_name] = results['per_category_mae'][cat_name]
            else:
                results['per_category_mae'][cat_name] = 0.0
                results['per_category_wae'][cat_name] = 0.0
    
    return results


class MetricAccumulator:
    """Accumulate metrics over batches and compute epoch-level averages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.depth_metrics = {'rmse': [], 'absrel': [], 'delta1': []}
        self.seg_metrics = {'miou': []}
        self.normal_metrics = {'mae': [], 'medae': [], 'wae': [], 'pfe': []}
        self.per_category_mae = {}
        self.per_category_wae = {}
    
    def update(self, d_metrics, s_metrics, n_metrics):
        """Add batch metrics."""
        for k, v in d_metrics.items():
            if k in self.depth_metrics:
                self.depth_metrics[k].append(v)
        
        for k, v in s_metrics.items():
            if k in self.seg_metrics:
                self.seg_metrics[k].append(v)
        
        for k, v in n_metrics.items():
            if k in self.normal_metrics:
                self.normal_metrics[k].append(v)
        
        # Per-category accumulation
        if 'per_category_mae' in n_metrics:
            for cat, val in n_metrics['per_category_mae'].items():
                if cat not in self.per_category_mae:
                    self.per_category_mae[cat] = []
                self.per_category_mae[cat].append(val)
        
        if 'per_category_wae' in n_metrics:
            for cat, val in n_metrics['per_category_wae'].items():
                if cat not in self.per_category_wae:
                    self.per_category_wae[cat] = []
                self.per_category_wae[cat].append(val)
    
    def compute(self):
        """Compute epoch averages."""
        results = {}
        
        for k, v in self.depth_metrics.items():
            results[f'depth/{k}'] = np.mean(v) if v else 0.0
        
        for k, v in self.seg_metrics.items():
            results[f'seg/{k}'] = np.mean(v) if v else 0.0
        
        for k, v in self.normal_metrics.items():
            results[f'normal/{k}'] = np.mean(v) if v else 0.0
        
        # Per-category averages
        results['per_category_mae'] = {
            cat: np.mean(vals) for cat, vals in self.per_category_mae.items() if vals
        }
        results['per_category_wae'] = {
            cat: np.mean(vals) for cat, vals in self.per_category_wae.items() if vals
        }
        
        return results

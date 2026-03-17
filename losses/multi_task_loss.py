"""
Multi-Task Loss for EdgeLite-MTAN

Segmentation uses Focal Loss (Lin et al., 2017):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Advantages over weighted CE:
    - Down-weights easy/well-classified pixels automatically
    - Focuses training on hard examples (boundaries, small objects)
    - Handles class imbalance via alpha weighting
    - gamma controls how aggressively easy examples are suppressed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleInvariantDepthLoss(nn.Module):
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
        loss = torch.sum(diff ** 2) / n - self.si_lambda * (torch.sum(diff) ** 2) / (n ** 2)
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 reduces to standard CE. gamma=2 is typical.
        alpha: Per-class weights (tensor of size num_classes) or None for uniform.
        ignore_index: Label to ignore (255 for Cityscapes unlabeled pixels).
    """
    
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] raw class scores (pre-softmax)
            targets: [B, H, W] ground truth class indices
        """
        B, C, H, W = logits.shape
        
        # Create valid mask (ignore unlabeled pixels)
        valid_mask = (targets != self.ignore_index)
        
        # Replace ignore pixels with 0 temporarily (for gather, will be masked out)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, H, W]
        probs = torch.exp(log_probs)
        
        # Gather the probability of the correct class for each pixel
        # targets_safe: [B, H, W] -> [B, 1, H, W] for gather
        targets_onehot = targets_safe.unsqueeze(1)  # [B, 1, H, W]
        
        log_pt = log_probs.gather(1, targets_onehot).squeeze(1)  # [B, H, W]
        pt = probs.gather(1, targets_onehot).squeeze(1)          # [B, H, W]
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma
        
        # Alpha weighting per class
        if self.alpha is not None:
            # Gather alpha for each pixel's target class
            alpha_t = self.alpha[targets_safe]  # [B, H, W]
            focal_weight = focal_weight * alpha_t
        
        # Focal loss: -alpha_t * (1-p_t)^gamma * log(p_t)
        loss = -focal_weight * log_pt
        
        # Apply valid mask
        loss = loss * valid_mask.float()
        
        # Average over valid pixels
        num_valid = valid_mask.sum().float() + 1e-6
        loss = loss.sum() / num_valid
        
        return loss


class CosineSimilarityNormalLoss(nn.Module):
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


# Cityscapes inverse-frequency alpha weights for focal loss
# Higher weight = rarer class gets more attention
CITYSCAPES_FOCAL_ALPHA = torch.FloatTensor([
    0.3,    # road (very common, low weight)
    0.5,    # sidewalk
    0.4,    # building
    1.5,    # wall (rare)
    1.3,    # fence
    2.0,    # pole (thin, rare)
    2.5,    # traffic light (tiny)
    2.0,    # traffic sign (small)
    0.4,    # vegetation
    0.8,    # terrain
    0.4,    # sky
    1.5,    # person
    3.0,    # rider (rare)
    0.5,    # car
    2.0,    # truck (rare)
    2.5,    # bus (rare)
    3.5,    # train (very rare)
    3.0,    # motorcycle (rare)
    2.0,    # bicycle
])


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss.
    
    Args:
        num_classes: Number of segmentation classes
        mode: 'fixed' or 'uncertainty'
        weight_depth, weight_seg, weight_normal: Task loss weights
        use_focal: Use focal loss for segmentation (default True)
        focal_gamma: Focal loss gamma parameter
        class_weights: Per-class alpha for focal loss (None = use defaults)
    """
    
    def __init__(self, num_classes=19, mode='fixed',
                 weight_depth=1.0, weight_seg=2.0, weight_normal=1.0,
                 use_focal=True, focal_gamma=2.0, class_weights=None):
        super().__init__()
        
        self.mode = mode
        self.depth_loss = ScaleInvariantDepthLoss(si_lambda=0.5)
        self.normal_loss = CosineSimilarityNormalLoss()
        
        # Segmentation loss
        if use_focal:
            alpha = class_weights if class_weights is not None else CITYSCAPES_FOCAL_ALPHA
            self.seg_loss = FocalLoss(gamma=focal_gamma, alpha=alpha, ignore_index=255)
        else:
            if class_weights is not None:
                self.seg_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
            else:
                self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        
        if mode == 'fixed':
            self.weight_depth = weight_depth
            self.weight_seg = weight_seg
            self.weight_normal = weight_normal
        elif mode == 'uncertainty':
            self.log_var_depth = nn.Parameter(torch.zeros(1))
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_normal = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, predictions, targets):
        l_depth = self.depth_loss(
            predictions['depth'], targets['depth'], targets['valid_depth'])
        l_seg = self.seg_loss(predictions['seg'], targets['seg'])
        l_normal = self.normal_loss(
            predictions['normal'], targets['normal'], targets['valid_depth'])
        
        if self.mode == 'fixed':
            total = (self.weight_depth * l_depth + 
                     self.weight_seg * l_seg + 
                     self.weight_normal * l_normal)
            
            loss_dict = {
                'loss_total': total.item(),
                'loss_depth': l_depth.item(),
                'loss_seg': l_seg.item(),
                'loss_normal': l_normal.item(),
                'weight_depth': self.weight_depth,
                'weight_seg': self.weight_seg,
                'weight_normal': self.weight_normal,
            }
        else:
            w_depth = 0.5 * torch.exp(-self.log_var_depth)
            w_seg = torch.exp(-self.log_var_seg)
            w_normal = 0.5 * torch.exp(-self.log_var_normal)
            
            total = (w_depth * l_depth + 0.5 * self.log_var_depth +
                     w_seg * l_seg + 0.5 * self.log_var_seg +
                     w_normal * l_normal + 0.5 * self.log_var_normal)
            
            loss_dict = {
                'loss_total': total.item(),
                'loss_depth': l_depth.item(),
                'loss_seg': l_seg.item(),
                'loss_normal': l_normal.item(),
                'weight_depth': w_depth.item(),
                'weight_seg': w_seg.item(),
                'weight_normal': w_normal.item(),
            }
        
        return total, loss_dict


__all__ = ['MultiTaskLoss', 'FocalLoss', 'CITYSCAPES_FOCAL_ALPHA']

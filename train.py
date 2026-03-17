"""
Training Script for EdgeLite-MTAN

Usage:
    # Best config for high mIoU
    python train.py --data_root cityscapes/ --epochs 80 --batch_size 4 --img_h 512 --img_w 1024 --w_seg 4.0 --focal_gamma 2.0

    # Lower memory alternative
    python train.py --data_root cityscapes/ --epochs 80 --batch_size 8 --w_seg 4.0 --focal_gamma 2.0

    # Disable focal loss (use plain CE)
    python train.py --data_root cityscapes/ --no_focal --epochs 80 --batch_size 8

    # Ablations
    python train.py --data_root cityscapes/ --no_attention --save_dir checkpoints/no_attn --log_dir logs/no_attn
    python train.py --data_root cityscapes/ --no_skips --save_dir checkpoints/no_skips --log_dir logs/no_skips
    python train.py --data_root cityscapes/ --w_depth 1.0 --w_seg 0.0 --w_normal 0.0 --save_dir checkpoints/depth_only --log_dir logs/depth_only
    python train.py --data_root cityscapes/ --w_depth 0.0 --w_seg 1.0 --w_normal 0.0 --save_dir checkpoints/seg_only --log_dir logs/seg_only
    python train.py --data_root cityscapes/ --w_depth 0.0 --w_seg 0.0 --w_normal 1.0 --save_dir checkpoints/normal_only --log_dir logs/normal_only
"""

import os
import argparse
import time
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model import EdgeLiteMTAN
from data.cityscapes import get_dataloaders, CATEGORY_GROUPS
from losses.multi_task_loss import MultiTaskLoss
from utils.metrics import (
    depth_metrics,
    segmentation_metrics,
    normal_metrics,
    MetricAccumulator,
)
from utils.visualization import visualize_predictions


def to_python_float(val):
    if hasattr(val, "item"):
        return val.item()
    return float(val)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EdgeLite-MTAN")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--vis_dir", type=str, default="visualizations")
    parser.add_argument("--vis_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true", default=True)

    # Loss config
    parser.add_argument(
        "--loss_mode", type=str, default="fixed", choices=["fixed", "uncertainty"]
    )
    parser.add_argument("--w_depth", type=float, default=1.0)
    parser.add_argument("--w_seg", type=float, default=4.0)
    parser.add_argument("--w_normal", type=float, default=1.0)
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma (0 = plain CE)"
    )
    parser.add_argument(
        "--no_focal",
        action="store_true",
        default=False,
        help="Disable focal loss, use plain CE",
    )

    # Ablation flags
    parser.add_argument("--no_attention", action="store_true", default=False)
    parser.add_argument("--no_skips", action="store_true", default=False)

    return parser.parse_args()


def train_one_epoch(
    model, loader, criterion, optimizer, device, grad_accum_steps, epoch
):
    model.train()
    running_losses = {}
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        loss = loss / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        for k, v in loss_dict.items():
            if k not in running_losses:
                running_losses[k] = []
            running_losses[k].append(v)

        pbar.set_postfix(
            {
                "loss": f"{loss_dict['loss_total']:.4f}",
                "depth": f"{loss_dict['loss_depth']:.4f}",
                "seg": f"{loss_dict['loss_seg']:.4f}",
                "normal": f"{loss_dict['loss_normal']:.4f}",
            }
        )

    if len(loader) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return {k: sum(v) / len(v) for k, v in running_losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, vis_dir=None, num_vis=4):
    model.eval()
    accumulator = MetricAccumulator()
    running_losses = {}
    vis_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for images, targets in pbar:
        images = images.to(device)
        targets_device = {k: v.to(device) for k, v in targets.items()}

        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets_device)

        for k, v in loss_dict.items():
            if k not in running_losses:
                running_losses[k] = []
            running_losses[k].append(v)

        d_met = depth_metrics(
            predictions["depth"], targets_device["depth"], targets_device["valid_depth"]
        )
        s_met = segmentation_metrics(predictions["seg"], targets_device["seg"])
        n_met = normal_metrics(
            predictions["normal"],
            targets_device["normal"],
            targets_device["valid_depth"],
            depth=targets_device["depth"],
            seg_labels=targets_device["seg"],
            category_groups=CATEGORY_GROUPS,
        )

        accumulator.update(d_met, s_met, n_met)

        if vis_dir and vis_count < num_vis:
            for b in range(min(images.shape[0], num_vis - vis_count)):
                save_path = os.path.join(vis_dir, f"epoch{epoch}_sample{vis_count}.png")
                pred_b = {k: v[b] for k, v in predictions.items()}
                tgt_b = {k: v[b] for k, v in targets.items()}
                visualize_predictions(images[b].cpu(), pred_b, tgt_b, save_path)
                vis_count += 1

    metrics = accumulator.compute()
    avg_losses = {k: sum(v) / len(v) for k, v in running_losses.items()}
    metrics.update(avg_losses)
    return metrics


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_attention = not args.no_attention
    use_skips = not args.no_skips
    use_focal = not args.no_focal

    variant_name = "EdgeLite-MTAN"
    if not use_attention:
        variant_name += " (no attention)"
    if not use_skips:
        variant_name += " (no skips)"
    if args.w_seg == 0 and args.w_normal == 0:
        variant_name = "Single-Task Depth"
    elif args.w_depth == 0 and args.w_normal == 0:
        variant_name = "Single-Task Segmentation"
    elif args.w_depth == 0 and args.w_seg == 0:
        variant_name = "Single-Task Normal"

    print(f"\nVariant: {variant_name}")
    print(f"  Attention: {'ON' if use_attention else 'OFF'}")
    print(f"  Skip connections: {'ON' if use_skips else 'OFF'}")
    print(f"  Resolution: {args.img_h}x{args.img_w}")
    print(
        f"  Focal loss: {'ON (gamma={})'.format(args.focal_gamma) if use_focal else 'OFF (plain CE)'}"
    )

    # Model
    model = EdgeLiteMTAN(
        num_classes=args.num_classes,
        pretrained_encoder=args.pretrained,
        use_attention=use_attention,
        use_skips=use_skips,
    ).to(device)

    counts = model.get_param_count()
    print("\nParameter counts:")
    for k, v in counts.items():
        print(f"  {k}: {v / 1e6:.3f}M")

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
    )

    # Loss
    criterion = MultiTaskLoss(
        num_classes=args.num_classes,
        mode=args.loss_mode,
        weight_depth=args.w_depth,
        weight_seg=args.w_seg,
        weight_normal=args.w_normal,
        use_focal=use_focal,
        focal_gamma=args.focal_gamma,
    ).to(device)

    print(f"\nLoss mode: {args.loss_mode}")
    print(f"  Weights: depth={args.w_depth}, seg={args.w_seg}, normal={args.w_normal}")

    # Optimizer
    if args.loss_mode == "uncertainty":
        optimizer = Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=args.lr
        )
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)

    # LR scheduler
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    start_epoch = 0
    best_metric = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", float("inf"))

    log_path = os.path.join(args.log_dir, "training_log.json")
    training_log = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_accum, epoch
        )

        do_vis = (epoch % args.vis_every == 0) or (epoch == args.epochs - 1)
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            vis_dir=args.vis_dir if do_vis else None,
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(
            f"Epoch {epoch}/{args.epochs-1}  ({elapsed:.1f}s)  LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        print(f"  Train Loss: {train_losses.get('loss_total', 0):.4f}")

        if args.w_depth > 0:
            print(
                f"  Val Depth:  RMSE={val_metrics.get('depth/rmse', 0):.3f}  "
                f"AbsRel={val_metrics.get('depth/absrel', 0):.4f}  "
                f"δ1={val_metrics.get('depth/delta1', 0):.3f}"
            )
        if args.w_seg > 0:
            print(f"  Val Seg:    mIoU={val_metrics.get('seg/miou', 0):.4f}")
        if args.w_normal > 0:
            print(
                f"  Val Normal: MAE={val_metrics.get('normal/mae', 0):.2f}°  "
                f"MedAE={val_metrics.get('normal/medae', 0):.2f}°  "
                f"WAE={val_metrics.get('normal/wae', 0):.2f}°  "
                f"PFE={val_metrics.get('normal/pfe', 0):.4f}"
            )

        if val_metrics.get("per_category_mae"):
            print(f"  Per-category MAE: ", end="")
            for cat, val in val_metrics["per_category_mae"].items():
                print(f"{cat}={val:.1f}° ", end="")
            print()

        print(f"{'='*60}\n")

        log_entry = {
            "epoch": epoch,
            "variant": variant_name,
            "lr": to_python_float(scheduler.get_last_lr()[0]),
            "train_losses": {k: to_python_float(v) for k, v in train_losses.items()},
            "val_metrics": {
                k: to_python_float(v)
                for k, v in val_metrics.items()
                if not isinstance(v, dict)
            },
            "per_category_mae": {
                k: to_python_float(v)
                for k, v in val_metrics.get("per_category_mae", {}).items()
            },
            "per_category_wae": {
                k: to_python_float(v)
                for k, v in val_metrics.get("per_category_wae", {}).items()
            },
        }
        training_log.append(log_entry)

        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        metric_parts = []
        if args.w_depth > 0:
            metric_parts.append(1 - val_metrics.get("depth/delta1", 0))
        if args.w_seg > 0:
            metric_parts.append(1 - val_metrics.get("seg/miou", 0))
        if args.w_normal > 0:
            metric_parts.append(val_metrics.get("normal/mae", 90) / 100.0)

        combined_metric = (
            sum(metric_parts) / len(metric_parts) if metric_parts else float("inf")
        )

        ckpt = {
            "epoch": epoch,
            "variant": variant_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": {
                k: (
                    to_python_float(v)
                    if not isinstance(v, dict)
                    else {kk: to_python_float(vv) for kk, vv in v.items()}
                )
                for k, v in val_metrics.items()
            },
            "best_metric": min(best_metric, combined_metric),
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))

        if combined_metric < best_metric:
            best_metric = combined_metric
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"  *** New best model! ({variant_name}) ***\n")

        if (epoch + 1) % 20 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch}.pth"))

    print(f"\nTraining complete ({variant_name}). Best metric: {best_metric:.4f}")
    print(f"Best model: {os.path.join(args.save_dir, 'best.pth')}")


if __name__ == "__main__":
    main()

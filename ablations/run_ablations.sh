#!/bin/bash
# =============================================================
# EdgeLite-MTAN Single-Task Ablation Runner
# Runs 3 single-task experiments to prove multi-task benefit
# =============================================================
# Usage from project root:
#   bash ablations/run_ablations.sh 2>&1 | tee ablations/ablation_log.txt
# =============================================================

DATA_ROOT="cityscapes/"
EPOCHS=30
BATCH=2
IMG_H=512
IMG_W=1024
COMMON="--data_root $DATA_ROOT --epochs $EPOCHS --batch_size $BATCH --img_h $IMG_H --img_w $IMG_W"

echo "============================================="
echo "Starting EdgeLite-MTAN Single-Task Ablations"
echo "Time: $(date)"
echo "============================================="

# --- Ablation 1: Depth Only ---
echo ""
echo "[1/3] Depth Only — Started at $(date)"
python train.py $COMMON \
    --w_depth 1.0 --w_seg 0.0 --w_normal 0.0 \
    --save_dir checkpoints/ablation_depth_only \
    --log_dir logs/ablation_depth_only \
    --vis_dir visualizations/ablation_depth_only
echo "[1/3] Depth Only — Finished at $(date)"

# --- Ablation 2: Seg Only ---
echo ""
echo "[2/3] Seg Only — Started at $(date)"
python train.py $COMMON \
    --w_depth 0.0 --w_seg 4.0 --w_normal 0.0 \
    --save_dir checkpoints/ablation_seg_only \
    --log_dir logs/ablation_seg_only \
    --vis_dir visualizations/ablation_seg_only
echo "[2/3] Seg Only — Finished at $(date)"

# --- Ablation 3: Normal Only ---
echo ""
echo "[3/3] Normal Only — Started at $(date)"
python train.py $COMMON \
    --w_depth 0.0 --w_seg 0.0 --w_normal 1.0 \
    --save_dir checkpoints/ablation_normal_only \
    --log_dir logs/ablation_normal_only \
    --vis_dir visualizations/ablation_normal_only
echo "[3/3] Normal Only — Finished at $(date)"

echo ""
echo "============================================="
echo "All single-task ablations complete at $(date)"
echo "============================================="
echo ""
echo "Results are in:"
echo "  logs/ablation_depth_only/training_log.json"
echo "  logs/ablation_seg_only/training_log.json"
echo "  logs/ablation_normal_only/training_log.json"
echo ""
echo "Run 'python ablations/summarize_ablations.py' to see summary table"

# EdgeLite-MTAN: Lightweight Multi-Task Attention Network

## Project Structure
```
edgelite_mtan/
├── model/
│   ├── __init__.py
│   ├── encoder.py          # MobileNetV3-Large shared encoder
│   ├── attention.py        # Reduced-dimensionality task attention
│   ├── decoder.py          # Task-specific decoders
│   └── edgelite_mtan.py    # Full model assembly
├── data/
│   ├── __init__.py
│   └── cityscapes.py       # Cityscapes dataloader (depth, seg, normals)
├── losses/
│   ├── __init__.py
│   └── multi_task_loss.py  # Scale-invariant depth, CE, cosine normal loss
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics (all tasks)
│   └── visualization.py    # Qualitative result visualization
├── train.py                # Training script
├── evaluate.py             # Evaluation + radar chart generation
├── export_onnx.py          # ONNX export for deployment
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Download Cityscapes from https://www.cityscapes-dataset.com/
Set `CITYSCAPES_ROOT` environment variable or pass `--data_root` to scripts.

Expected structure:
```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
├── gtFine/           # segmentation labels
├── disparity/        # depth (disparity maps)
└── camera/           # for surface normal computation
```

## Training
```bash
python train.py --data_root /path/to/cityscapes --epochs 80 --batch_size 2 --grad_accum 2
```

## Evaluation
```bash
python evaluate.py --data_root /path/to/cityscapes --checkpoint checkpoints/best.pth
```

## ONNX Export
```bash
python export_onnx.py --checkpoint checkpoints/best.pth --output edgelite_mtan.onnx
```

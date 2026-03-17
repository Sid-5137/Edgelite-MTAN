"""
ONNX Export for EdgeLite-MTAN

Exports the model to ONNX format for TensorRT deployment on Jetson.

Usage:
    python export_onnx.py --checkpoint checkpoints/best.pth --output edgelite_mtan.onnx

For TensorRT on Jetson:
    trtexec --onnx=edgelite_mtan.onnx --saveEngine=edgelite_mtan.engine --fp16
"""

import argparse
import torch
import onnx

from model import EdgeLiteMTAN


def parse_args():
    parser = argparse.ArgumentParser(description='Export EdgeLite-MTAN to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='edgelite_mtan.onnx')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--simplify', action='store_true', default=True)
    return parser.parse_args()


class EdgeLiteMTANExport(torch.nn.Module):
    """Wrapper that returns a tuple instead of dict for ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out['depth'], out['seg'], out['normal']


def main():
    args = parse_args()
    
    # Load model
    model = EdgeLiteMTAN(num_classes=args.num_classes, pretrained_encoder=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    export_model = EdgeLiteMTANExport(model)
    
    # Dummy input
    dummy = torch.randn(1, 3, args.img_h, args.img_w)
    
    # Export
    print(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        export_model, dummy, args.output,
        opset_version=args.opset,
        input_names=['input'],
        output_names=['depth', 'segmentation', 'normals'],
        dynamic_axes={
            'input': {0: 'batch'},
            'depth': {0: 'batch'},
            'segmentation': {0: 'batch'},
            'normals': {0: 'batch'},
        },
    )
    
    # Verify
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to: {args.output}")
    
    # Optional: simplify
    if args.simplify:
        try:
            import onnxsim
            model_sim, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_sim, args.output)
                print("Model simplified successfully")
        except ImportError:
            print("onnxsim not installed, skipping simplification")
    
    # Print model size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total / 1e6:.1f}M")
    
    print(f"\nFor TensorRT deployment on Jetson:")
    print(f"  trtexec --onnx={args.output} --saveEngine=edgelite_mtan.engine --fp16")


if __name__ == '__main__':
    main()

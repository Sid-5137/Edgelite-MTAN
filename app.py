"""
MTAN-Lite Gradio Demo

Interactive web app to run MTAN-Lite inference on any image.
Upload an image or use webcam, select dataset checkpoint, and see
depth, segmentation, and surface normal predictions in real time.

Usage:
    python app.py

    # Custom checkpoint paths
    python app.py --cityscapes_ckpt path/to/cityscapes.pth --nyuv2_ckpt path/to/nyuv2.pth

    # Public share link (for demo on another device)
    python app.py --share

    # Custom port
    python app.py --port 7861

Requirements:
    pip install gradio torch torchvision matplotlib numpy Pillow
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import gradio as gr

from model import EdgeLiteMTAN


# ============================================================
# CONFIG
# ============================================================

CONFIGS = {
    "cityscapes": {
        "num_classes": 19,
        "img_h": 512,
        "img_w": 1024,
        "class_names": [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle"
        ],
        "palette": np.array([
            [128,64,128],[244,35,232],[70,70,70],[102,102,156],
            [190,153,153],[153,153,153],[250,170,30],[220,220,0],
            [107,142,35],[152,251,152],[70,130,180],[220,20,60],
            [255,0,0],[0,0,142],[0,0,70],[0,60,100],
            [0,80,100],[0,0,230],[119,11,32],
        ], dtype=np.uint8),
    },
    "nyuv2": {
        "num_classes": 13,
        "img_h": 288,
        "img_w": 384,
        "class_names": [
            "bed", "books", "ceiling", "chair", "floor",
            "furniture", "objects", "painting", "sofa",
            "table", "tv", "wall", "window"
        ],
        "palette": np.array([
            [0,0,128],[128,64,0],[200,200,200],[255,128,0],
            [128,64,128],[180,120,60],[255,0,255],[255,255,0],
            [0,128,128],[128,128,0],[0,255,255],[70,130,180],
            [220,20,60],
        ], dtype=np.uint8),
    },
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Global model cache
_models = {}


# ============================================================
# MODEL LOADING
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(dataset, checkpoint_path):
    """Load model with caching so we don't reload every time."""
    cache_key = f"{dataset}_{checkpoint_path}"
    if cache_key in _models:
        return _models[cache_key]

    config = CONFIGS[dataset]
    device = get_device()

    model = EdgeLiteMTAN(
        num_classes=config["num_classes"],
        pretrained_encoder=False,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device).eval()
    _models[cache_key] = (model, config, device)
    print(f"Loaded {dataset} model from {checkpoint_path} on {device}")
    return model, config, device


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def colorize_seg(seg_map, palette):
    h, w = seg_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        color[seg_map == cls_id] = palette[cls_id]
    return color


def colorize_depth(depth):
    d = depth.copy()
    valid = d > 0
    if valid.any():
        d = (d - d[valid].min()) / (d[valid].max() - d[valid].min() + 1e-8)
    colored = (cm.magma_r(d)[:, :, :3] * 255).astype(np.uint8)
    return colored


def colorize_normals(normals):
    return ((normals + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)


def make_seg_overlay(img_np, seg_vis, alpha=0.5):
    overlay = img_np.copy().astype(np.float32)
    overlay = overlay * (1 - alpha) + seg_vis.astype(np.float32) * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def make_legend_image(seg_map, palette, class_names):
    """Create a legend image showing classes present in the segmentation."""
    unique = np.unique(seg_map)
    fig, ax = plt.subplots(figsize=(4, max(2, len(unique) * 0.35)))
    ax.axis("off")

    patches = []
    labels = []
    for cls_id in unique:
        if cls_id < len(class_names):
            color = palette[cls_id] / 255.0
            patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))
            # Count pixels
            count = (seg_map == cls_id).sum()
            pct = count / seg_map.size * 100
            labels.append(f"{class_names[cls_id]} ({pct:.1f}%)")

    ax.legend(patches, labels, loc="center", fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="#ccc",
              handlelength=1.5, handletextpad=0.8)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    # Convert to numpy (compatible with all matplotlib versions)
    import io
    buf_io = io.BytesIO()
    fig.savefig(buf_io, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf_io.seek(0)
    buf = np.array(Image.open(buf_io).convert("RGB"))
    plt.close(fig)
    return buf


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict(image, dataset_choice, cityscapes_ckpt, nyuv2_ckpt):
    """Main inference function called by Gradio."""
    if image is None:
        return None, None, None, None, None, "Upload an image first."

    # Pick checkpoint
    if dataset_choice == "Cityscapes (outdoor)":
        dataset = "cityscapes"
        ckpt_path = cityscapes_ckpt
    else:
        dataset = "nyuv2"
        ckpt_path = nyuv2_ckpt

    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, None, None, None, None, f"Checkpoint not found: {ckpt_path}"

    # Load model
    model, config, device = load_model(dataset, ckpt_path)

    # Preprocess
    if isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image).convert("RGB")
    else:
        img_pil = image.convert("RGB")

    img_resized = img_pil.resize((config["img_w"], config["img_h"]), Image.BILINEAR)
    img_np = np.array(img_resized)

    tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = (tensor - MEAN) / STD
    tensor = tensor.to(device)

    # Inference
    start = time.time()
    preds = model(tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    # Extract outputs
    depth = preds["depth"].squeeze().cpu().numpy()
    seg = preds["seg"].argmax(dim=1).squeeze().cpu().numpy()
    normal = preds["normal"].squeeze().cpu().permute(1, 2, 0).numpy()

    # Visualize
    depth_vis = colorize_depth(depth)
    seg_vis = colorize_seg(seg, config["palette"])
    normal_vis = colorize_normals(normal)
    overlay = make_seg_overlay(img_np, seg_vis, alpha=0.5)
    legend = make_legend_image(seg, config["palette"], config["class_names"])

    # Info string
    fps = 1.0 / elapsed if elapsed > 0 else 0
    info = (
        f"Model: MTAN-Lite ({config['num_classes']} classes) | "
        f"Params: 6.2M | "
        f"Input: {config['img_h']}x{config['img_w']} | "
        f"Inference: {elapsed*1000:.1f} ms ({fps:.1f} FPS) | "
        f"Device: {device}"
    )

    return depth_vis, seg_vis, normal_vis, overlay, legend, info


# ============================================================
# GRADIO UI
# ============================================================

def build_app(cityscapes_ckpt, nyuv2_ckpt):

    custom_css = """
    .main-header { text-align: center; margin-bottom: 8px; }
    .main-header h1 { font-size: 2em; font-weight: 700; margin: 0; }
    .main-header p { color: #6b7280; margin: 4px 0; }
    .stat-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
                 padding: 12px 16px; text-align: center; }
    .stat-box .label { font-size: 0.75em; color: #6b7280; text-transform: uppercase;
                       letter-spacing: 0.05em; }
    .stat-box .value { font-size: 1.3em; font-weight: 700; color: #1e293b; }
    """

    with gr.Blocks(title="MTAN-Lite Demo") as app:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>MTAN-Lite</h1>
            <p>Lightweight Multi-Task Attention Network for Real-Time Dense Scene Understanding</p>
        </div>
        """)

        # Stats row
        with gr.Row():
            for label, value in [
                ("Parameters", "6.2M"),
                ("Backbone", "MobileNetV3-Large"),
                ("Tasks", "Depth + Seg + Normals"),
                ("FPS (GTX 1650)", "39.6"),
                ("GPU Memory", "102 MB"),
            ]:
                gr.HTML(f'<div class="stat-box"><div class="label">{label}</div><div class="value">{value}</div></div>')

        gr.Markdown("---")

        # Controls row
        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"],
                    height=350,
                )

            with gr.Column(scale=1):
                dataset_choice = gr.Radio(
                    label="Checkpoint",
                    choices=["Cityscapes (outdoor)", "NYUv2 (indoor)"],
                    value="Cityscapes (outdoor)",
                )

                cityscapes_box = gr.Textbox(
                    label="Cityscapes checkpoint path",
                    value=cityscapes_ckpt,
                    placeholder="checkpoints/Cityscapes/best.pth",
                )
                nyuv2_box = gr.Textbox(
                    label="NYUv2 checkpoint path",
                    value=nyuv2_ckpt,
                    placeholder="checkpoints/nyuv2/best.pth",
                )

                run_btn = gr.Button("Run Inference", variant="primary", size="lg")

        # Info bar
        info_text = gr.Textbox(label="Inference Info", interactive=False)

        # Output tabs
        with gr.Tabs():
            with gr.Tab("Depth"):
                depth_out = gr.Image(label="Predicted Depth", height=400)
            with gr.Tab("Segmentation"):
                seg_out = gr.Image(label="Semantic Segmentation", height=400)
            with gr.Tab("Surface Normals"):
                normal_out = gr.Image(label="Surface Normals", height=400)
            with gr.Tab("Overlay"):
                overlay_out = gr.Image(label="Segmentation Overlay", height=400)
            with gr.Tab("Legend"):
                legend_out = gr.Image(label="Class Legend", height=400)

        # Wire up
        run_btn.click(
            fn=predict,
            inputs=[input_image, dataset_choice, cityscapes_box, nyuv2_box],
            outputs=[depth_out, seg_out, normal_out, overlay_out, legend_out, info_text],
        )

        # Also auto-run on image upload
        input_image.change(
            fn=predict,
            inputs=[input_image, dataset_choice, cityscapes_box, nyuv2_box],
            outputs=[depth_out, seg_out, normal_out, overlay_out, legend_out, info_text],
        )

        # Examples
        example_dir = "demo_images"
        if os.path.isdir(example_dir):
            examples = sorted([
                os.path.join(example_dir, f)
                for f in os.listdir(example_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])[:6]
            if examples:
                gr.Markdown("### Example Images")
                gr.Examples(
                    examples=[[ex] for ex in examples],
                    inputs=[input_image],
                )

        # Footer
        gr.Markdown("""
        ---
        **MTAN-Lite** | [GitHub](https://github.com/Sid-5137/edgelite-mtan) |
        6.2M parameters | MobileNetV3-Large backbone |
        Cityscapes (19-class) + NYUv2 (13-class)
        """)

    return app


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTAN-Lite Gradio Demo")
    parser.add_argument("--cityscapes_ckpt", type=str,
                        default="checkpoints/Cityscapes/best.pth",
                        help="Path to Cityscapes checkpoint")
    parser.add_argument("--nyuv2_ckpt", type=str,
                        default="checkpoints/nyuv2/best.pth",
                        help="Path to NYUv2 checkpoint")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio share link")
    args = parser.parse_args()

    print("=" * 60)
    print("  MTAN-Lite Gradio Demo")
    print("=" * 60)
    print(f"  Cityscapes ckpt: {args.cityscapes_ckpt}")
    print(f"  NYUv2 ckpt:      {args.nyuv2_ckpt}")
    print(f"  Device:          {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Port:            {args.port}")
    print(f"  Share:           {args.share}")
    print("=" * 60)

    app = build_app(args.cityscapes_ckpt, args.nyuv2_ckpt)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )

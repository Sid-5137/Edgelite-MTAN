"""FPS and latency benchmark for EdgeLite-MTAN on GPU."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import numpy as np
from model import EdgeLiteMTAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

model = EdgeLiteMTAN(num_classes=19, pretrained_encoder=False).to(device)
model.eval()

H, W = 512, 1024
dummy = torch.randn(1, 3, H, W).to(device)

# Warmup (important — first few runs are slow due to CUDA initialization)
print("Warming up...")
for _ in range(50):
    with torch.no_grad():
        _ = model(dummy)
torch.cuda.synchronize()

# Benchmark
num_runs = 500
print(f"Benchmarking {num_runs} forward passes at {H}x{W}...")

# Measure GPU memory
torch.cuda.reset_peak_memory_stats()

times = []
for _ in range(num_runs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

times = np.array(times)
peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

print(f"\n{'='*50}")
print(f"EdgeLite-MTAN Inference Benchmark")
print(f"Resolution: {H}x{W} | Batch size: 1")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*50}")
print(f"Mean latency:   {times.mean()*1000:.1f} ms")
print(f"Std latency:    {times.std()*1000:.1f} ms")
print(f"Median latency: {np.median(times)*1000:.1f} ms")
print(f"Min latency:    {times.min()*1000:.1f} ms")
print(f"Max latency:    {times.max()*1000:.1f} ms")
print(f"FPS (mean):     {1.0/times.mean():.1f}")
print(f"FPS (median):   {1.0/np.median(times):.1f}")
print(f"Peak GPU mem:   {peak_mem:.1f} MB")
print(f"{'='*50}")

# Also measure batch=2 (your training batch size)
dummy_b2 = torch.randn(2, 3, H, W).to(device)
times_b2 = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_b2)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times_b2.append(t1 - t0)

times_b2 = np.array(times_b2)
print(f"\nBatch=2: {1.0/times_b2.mean():.1f} FPS ({times_b2.mean()*1000:.1f} ms)")

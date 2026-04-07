#!/usr/bin/env python3
"""Latency benchmark for SwinIR inference.

Measures inference time across image sizes on the available device (CPU/CUDA).
Outputs a markdown table suitable for README.

Usage:
    python scripts/run_latency_bench.py
    python scripts/run_latency_bench.py --checkpoint outputs/.../best.pt
    python scripts/run_latency_bench.py --n-warmup 5 --n-runs 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image

from inverseops.models.swinir import SwinIRBaseline


def make_test_image(size: int) -> Image.Image:
    """Create a random grayscale test image."""
    arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def bench_single(
    model: SwinIRBaseline, image: Image.Image, n_warmup: int, n_runs: int
) -> dict:
    """Benchmark a single image size. Returns timing stats in ms."""
    # Warmup
    for _ in range(n_warmup):
        model.predict_image(image)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict_image(image)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="SwinIR latency benchmark")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Fine-tuned checkpoint path"
    )
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[128, 256, 512],
        help="Image sizes to benchmark",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Warmup: {args.n_warmup}, Runs: {args.n_runs}")
    print()

    model = SwinIRBaseline(noise_level=25, device=device)
    model.load()

    results = []
    for size in args.sizes:
        image = make_test_image(size)
        stats = bench_single(model, image, args.n_warmup, args.n_runs)
        results.append((size, stats))
        print(
            f"{size}x{size}: {stats['mean']:.1f} ms "
            f"(p50={stats['p50']:.1f}, p95={stats['p95']:.1f})"
        )

    # Print markdown table
    print()
    print("| Image Size | Mean (ms) | P50 (ms) | P95 (ms) | Device |")
    print("|-----------|----------|---------|---------|--------|")
    for size, stats in results:
        row = (
            f"| {size}x{size} | {stats['mean']:.1f} "
            f"| {stats['p50']:.1f} | {stats['p95']:.1f} "
            f"| {device} |"
        )
        print(row)


if __name__ == "__main__":
    main()

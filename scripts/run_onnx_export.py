#!/usr/bin/env python3
"""ONNX export for SwinIR model.

Time-boxed attempt: exports the model to ONNX format and benchmarks
inference latency against the PyTorch baseline.

Usage:
    python scripts/run_onnx_export.py
    python scripts/run_onnx_export.py --sigma 25 --size 256
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from inverseops.models.swinir import SwinIRBaseline


def export_onnx(
    model: SwinIRBaseline,
    output_path: Path,
    input_size: int = 128,
) -> bool:
    """Export the loaded SwinIR model to ONNX.

    Returns True on success, False on failure.
    """
    dummy = torch.randn(1, 1, input_size, input_size)
    dummy = dummy.to(model.device)

    try:
        torch.onnx.export(
            model._model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"ONNX export saved to {output_path}")
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def validate_onnx(onnx_path: Path) -> bool:
    """Validate the exported ONNX model."""
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("ONNX model validation passed")
        return True
    except Exception as e:
        print(f"ONNX validation failed: {e}")
        return False


def benchmark_onnx(
    onnx_path: Path,
    input_size: int = 128,
    n_warmup: int = 3,
    n_runs: int = 20,
) -> dict | None:
    """Benchmark ONNX Runtime inference."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping ONNX benchmark")
        return None

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    dummy = np.random.randn(
        1, 1, input_size, input_size
    ).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        session.run(None, {"input": dummy})

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"input": dummy})
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def benchmark_pytorch(
    model: SwinIRBaseline,
    input_size: int = 128,
    n_warmup: int = 3,
    n_runs: int = 20,
) -> dict:
    """Benchmark PyTorch inference for comparison."""
    from PIL import Image

    dummy_arr = np.random.randint(
        0, 256, (input_size, input_size), dtype=np.uint8
    )
    dummy_img = Image.fromarray(dummy_arr, mode="L")

    for _ in range(n_warmup):
        model.predict_image(dummy_img)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict_image(dummy_img)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ONNX export for SwinIR"
    )
    parser.add_argument("--sigma", type=int, default=25)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/onnx",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"swinir_sigma{args.sigma}.onnx"

    # Load model
    print(f"Loading SwinIR sigma={args.sigma}...")
    model = SwinIRBaseline(
        noise_level=args.sigma, device="cpu"
    )
    model.load()

    # Export
    print(f"Exporting to ONNX (input {args.size}x{args.size})...")
    success = export_onnx(model, onnx_path, args.size)
    if not success:
        print("\nONNX export failed. Documenting failure.")
        return

    # Validate
    if not validate_onnx(onnx_path):
        return

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX file size: {file_size_mb:.1f} MB")

    # Benchmark comparison
    print(f"\nBenchmarking at {args.size}x{args.size}...")

    pt_stats = benchmark_pytorch(
        model, args.size, args.n_warmup, args.n_runs
    )
    print(
        f"  PyTorch:  {pt_stats['mean']:.1f} ms "
        f"(p50={pt_stats['p50']:.1f}, p95={pt_stats['p95']:.1f})"
    )

    onnx_stats = benchmark_onnx(
        onnx_path, args.size, args.n_warmup, args.n_runs
    )
    if onnx_stats:
        print(
            f"  ONNX RT:  {onnx_stats['mean']:.1f} ms "
            f"(p50={onnx_stats['p50']:.1f}, "
            f"p95={onnx_stats['p95']:.1f})"
        )
        speedup = pt_stats["mean"] / onnx_stats["mean"]
        print(f"  Speedup:  {speedup:.2f}x")

    # Print markdown table
    print()
    print("| Backend | Mean (ms) | P50 (ms) | P95 (ms) |")
    print("|---------|----------|---------|---------|")
    print(
        f"| PyTorch FP32 | {pt_stats['mean']:.1f} "
        f"| {pt_stats['p50']:.1f} "
        f"| {pt_stats['p95']:.1f} |"
    )
    if onnx_stats:
        print(
            f"| ONNX Runtime | {onnx_stats['mean']:.1f} "
            f"| {onnx_stats['p50']:.1f} "
            f"| {onnx_stats['p95']:.1f} |"
        )


if __name__ == "__main__":
    main()

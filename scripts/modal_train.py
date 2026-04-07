#!/usr/bin/env python3
"""Modal GPU training for InverseOps SwinIR fine-tuning.

Prerequisites:
    pip install modal
    modal setup          # one-time auth

Usage:
    # Run training on T4 GPU (run from project root)
    modal run scripts/modal_train.py

    # Custom training args
    modal run scripts/modal_train.py --epochs 50 --batch-size 8

    # Download results after training
    modal volume get inverseops-vol outputs/training/checkpoints/ outputs/modal_training/checkpoints/
    modal volume get inverseops-vol outputs/training/training_summary.json outputs/modal_training/
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("inverseops-training")

# Persistent volume for training outputs only (checkpoints, summaries)
vol = modal.Volume.from_name("inverseops-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: deps + pretrained weights + data — all baked in, local-disk fast
# ---------------------------------------------------------------------------

SWINIR_WEIGHTS_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
)
WEIGHTS_DIR = "/cache/inverseops/models"
DATA_DIR = "/data/fmd"

# 1) Install Python deps + download pretrained weights
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "timm>=0.9.0",
        "numpy>=1.24",
        "pillow>=10.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "wandb>=0.15",
    )
    .run_commands(
        f"mkdir -p {WEIGHTS_DIR}",
        f"python -c \"import urllib.request; urllib.request.urlretrieve('{SWINIR_WEIGHTS_URL}', '{WEIGHTS_DIR}/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth')\"",
    )
)

# 2) Add fmd.zip and extract into image (cached after first build)
data_image = (
    base_image
    .add_local_file("data/raw/fmd.zip", remote_path="/tmp/fmd.zip", copy=True)
    .run_commands(
        f"mkdir -p {DATA_DIR}",
        f"python -c \"import zipfile; zipfile.ZipFile('/tmp/fmd.zip').extractall('{DATA_DIR}')\"",
        "rm /tmp/fmd.zip",
    )
)

# 3) Add source code on top
def _source_ignore(path: Path) -> bool:
    skip = {"data", ".git", "__pycache__", "outputs", "artifacts",
            ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    top = path.parts[0] if path.parts else ""
    return top in skip

train_image = data_image.add_local_dir(
    ".", remote_path="/app", ignore=_source_ignore,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="A100",  # 40GB; use "A100-80GB" for larger batch sizes if available
    volumes={"/vol": vol},      # only for saving outputs
    timeout=86400,              # 24 hours
)
def train(
    epochs: int = 100,
    batch_size: int = 4,
    limit_train: int = 0,
    limit_val: int = 0,
    resume: bool = False,
):
    """Run SwinIR fine-tuning on a Modal GPU."""
    import json
    import subprocess
    import sys

    import yaml

    # ------------------------------------------------------------------
    # Patch config for Modal paths (data is on local disk in the image)
    # ------------------------------------------------------------------
    with open("/app/configs/denoise_swinir.yaml") as f:
        config = yaml.safe_load(f)

    config["data"]["train_root"] = DATA_DIR
    config["data"]["val_root"] = DATA_DIR
    config["data"]["num_workers"] = 0
    config["training"]["amp"] = False              # avoid fp16 NaN on T4
    config["output_dir"] = "/vol/outputs/training"

    modal_config = Path("/tmp/modal_config.yaml")
    with open(modal_config, "w") as f:
        yaml.dump(config, f)

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    cmd = [
        sys.executable, "/app/scripts/run_training.py",
        "--config", str(modal_config),
        "--no-wandb",
        "--preload",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    if limit_train > 0:
        cmd += ["--limit-train-samples", str(limit_train)]
    if limit_val > 0:
        cmd += ["--limit-val-samples", str(limit_val)]
    if resume:
        checkpoint = Path("/vol/outputs/training/checkpoints/latest.pt")
        if checkpoint.exists():
            cmd += ["--resume", str(checkpoint)]
            print(f"Resuming from {checkpoint}")
        else:
            print("No checkpoint found — starting fresh")

    env = {
        **os.environ,
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
        "TORCH_HOME": "/vol/.cache/torch",
        "INVERSEOPS_CACHE": "/cache/inverseops",
    }

    print(f"Training: {epochs} epochs, batch_size={batch_size}, gpu=A100")
    print(f"Data: {DATA_DIR} (baked into image)")
    print("=" * 60)
    result = subprocess.run(cmd, env=env)
    print("=" * 60)

    vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {result.returncode})")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    summary_path = Path("/vol/outputs/training/training_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\nBest PSNR:  {summary.get('best_val_psnr', 0):.2f} dB")
        print(f"Best epoch: {summary.get('best_epoch', '?')}")
        print(f"Epochs:     {summary.get('epochs_completed', '?')}")
        print(f"Time:       {summary.get('train_time_minutes', 0):.1f} min")
        print(f"Device:     {summary.get('device', '?')}")
        gpu_mb = summary.get("max_gpu_memory_mb")
        if gpu_mb:
            print(f"GPU mem:    {gpu_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Evaluation: compare fine-tuned vs pretrained
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={"/vol": vol},
    timeout=3600,
)
def evaluate():
    """Run comparison evaluation: fine-tuned vs pretrained baseline."""
    import subprocess
    import sys

    microscopy_root = f"{DATA_DIR}/fmd/Confocal_FISH/gt"
    checkpoint = "/vol/outputs/training/checkpoints/best.pt"

    # Full comparison (sigma 15, 25, 50)
    cmd = [
        sys.executable, "/app/scripts/run_evaluation.py",
        "--microscopy-root", microscopy_root,
        "--checkpoint", checkpoint,
        "--model-mode", "finetuned",
        "--output-csv", "/vol/artifacts/compare_finetuned/finetuned_full_metrics.csv",
        "--baseline-csv", "/vol/artifacts/baseline/baseline_summary.csv",
        "--summary-csv", "/vol/artifacts/compare_finetuned/finetuned_summary.csv",
        "--no-wandb",
        "--allow-missing-datasets",
    ]

    env = {
        **os.environ,
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
        "INVERSEOPS_CACHE": "/cache/inverseops",
    }

    print("Running comparison evaluation (fine-tuned vs pretrained)...")
    print("=" * 60)
    result = subprocess.run(cmd, env=env)
    print("=" * 60)

    vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed (exit code {result.returncode})")

    # Print comparison summary
    compare_path = Path("/vol/artifacts/compare_finetuned/compare_summary.csv")
    if compare_path.exists():
        print(f"\n{compare_path.read_text()}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 4,
    limit_train: int = 0,
    limit_val: int = 0,
    resume: bool = False,
):
    """Train SwinIR on a Modal GPU."""
    print(f"Training: {epochs} epochs, batch_size={batch_size}")
    if limit_train > 0:
        print(f"  limit_train={limit_train}, limit_val={limit_val}")
    if resume:
        print("  Resuming from last checkpoint")
    train.remote(
        epochs=epochs,
        batch_size=batch_size,
        limit_train=limit_train,
        limit_val=limit_val,
        resume=resume,
    )

    print("\n" + "=" * 60)
    print("Download results:")
    print("  modal volume get inverseops-vol outputs/training/checkpoints/ outputs/modal_training/checkpoints/")
    print("  modal volume get inverseops-vol outputs/training/training_summary.json outputs/modal_training/")
    print("=" * 60)

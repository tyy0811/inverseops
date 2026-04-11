#!/usr/bin/env python3
"""Modal GPU training for InverseOps models.

Prerequisites:
    pip install modal
    modal setup          # one-time auth
    modal secret create wandb-api-key WANDB_API_KEY=<key>  # optional, for W&B

Usage:
    # W2S SwinIR denoising
    modal run scripts/modal_train.py --config configs/w2s_denoise_swinir.yaml

    # W2S NAFNet denoising
    modal run scripts/modal_train.py --config configs/w2s_denoise_nafnet.yaml

    # With W&B logging
    modal run scripts/modal_train.py --wandb

    # Custom training args
    modal run scripts/modal_train.py --epochs 50 --batch-size 8

    # Download results after training
    modal volume get inverseops-vol \
        outputs/training/checkpoints/ \
        outputs/modal_training/checkpoints/
    modal volume get inverseops-vol \
        outputs/training/training_summary.json \
        outputs/modal_training/
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("inverseops-training")

# Persistent volume for training outputs only (checkpoints, summaries)
vol = modal.Volume.from_name("inverseops-vol", create_if_missing=True)

# Data volume — W2S normalized .npy files downloaded via scripts/download_w2s.py
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: deps + pretrained weights (data loaded from volume at runtime)
# ---------------------------------------------------------------------------

WEIGHTS_DIR = "/cache/inverseops/models"

# All pretrained weight URLs — baked into image at build time
PRETRAINED_URLS = [
    # SwinIR denoising (sigma 15, 25, 50)
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth",
    # SwinIR SR (2x, 4x)
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
    # NAFNet SIDD denoising (mirrored to GitHub release — original is Google Drive)
    "https://github.com/tyy0811/inverseops/releases/download/pretrained-weights-v1/NAFNet-SIDD-width32.pth",
]

# 1) Install Python deps + download all pretrained weights
_download_cmds = [f"mkdir -p {WEIGHTS_DIR}"] + [
    (
        f"python -c \"import urllib.request; "
        f"urllib.request.urlretrieve('{url}', "
        f"'{WEIGHTS_DIR}/{url.split('/')[-1]}')\""
    )
    for url in PRETRAINED_URLS
]

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
        "structlog>=23.0",
        "nibabel>=5.0",
    )
    .run_commands(*_download_cmds)
)

# Add source code on top (data loaded from volume, not baked into image)
def _source_ignore(path: Path) -> bool:
    skip = {"data", ".git", "__pycache__", "outputs", "artifacts",
            ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    top = path.parts[0] if path.parts else ""
    return top in skip

train_image = base_image.add_local_dir(
    ".", remote_path="/app", ignore=_source_ignore,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="A100",
    volumes={"/vol": vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("wandb-api-key")],
    timeout=86400,              # 24 hours
)
def train(
    config_path: str = "configs/w2s_denoise_swinir.yaml",
    epochs: int = 100,
    batch_size: int = 4,
    limit_train: int = 0,
    limit_val: int = 0,
    resume: bool = False,
    wandb: bool = False,
    pretrained_checkpoint: str = "",
):
    """Run training on a Modal GPU.

    Args:
        config_path: Path to config YAML (relative to project root).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        limit_train: Limit training samples (0 = no limit).
        limit_val: Limit validation samples (0 = no limit).
        resume: Resume from last checkpoint.
        wandb: Enable W&B logging (requires wandb-api-key secret).
        pretrained_checkpoint: Path to checkpoint for transfer learning
            (loads weights only, resets optimizer/epoch). Relative to /vol/.
    """
    import json
    import subprocess
    import sys

    import yaml

    # ------------------------------------------------------------------
    # Patch config for Modal paths (data on volume, source in /app)
    # ------------------------------------------------------------------
    with open(f"/app/{config_path}") as f:
        config = yaml.safe_load(f)

    # Data paths already point to /data/w2s/... in W2S configs.
    # Rebase splits_path to the app source tree.
    if config["data"].get("splits_path"):
        config["data"]["splits_path"] = f"/app/{config['data']['splits_path']}"
    config["data"]["num_workers"] = 0
    config["output_dir"] = f"/vol/{config['output_dir']}"

    modal_config = Path("/tmp/modal_config.yaml")
    with open(modal_config, "w") as f:
        yaml.dump(config, f)

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    wandb_flag = "--wandb" if wandb else "--no-wandb"
    cmd = [
        sys.executable, "/app/scripts/run_training.py",
        "--config", str(modal_config),
        wandb_flag,
        "--preload",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    if limit_train > 0:
        cmd += ["--limit-train-samples", str(limit_train)]
    if limit_val > 0:
        cmd += ["--limit-val-samples", str(limit_val)]
    if resume:
        checkpoint = Path(config["output_dir"]) / "checkpoints" / "latest.pt"
        if checkpoint.exists():
            cmd += ["--resume", str(checkpoint)]
            print(f"Resuming from {checkpoint}")
        else:
            print("No checkpoint found — starting fresh")
    if pretrained_checkpoint:
        pt_path = f"/vol/{pretrained_checkpoint}"
        cmd += ["--pretrained-checkpoint", pt_path]
        print(f"Transfer learning from {pt_path}")

    env = {
        **os.environ,
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
        "TORCH_HOME": "/vol/.cache/torch",
        "INVERSEOPS_CACHE": "/cache/inverseops",
    }

    print(f"Config: {config_path}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, gpu=A100")
    print(f"W&B: {'enabled' if wandb else 'disabled'}")
    print(f"Data: /data volume (inverseops-data)")
    print("=" * 60)
    result = subprocess.run(cmd, env=env)
    print("=" * 60)

    vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {result.returncode})")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    summary_path = Path(config["output_dir"]) / "training_summary.json"
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
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    config: str = "configs/w2s_denoise_swinir.yaml",
    epochs: int = 100,
    batch_size: int = 4,
    limit_train: int = 0,
    limit_val: int = 0,
    resume: bool = False,
    wandb: bool = False,
    pretrained_checkpoint: str = "",
):
    """Train on a Modal GPU with any config."""
    print(f"Config: {config}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}")
    if limit_train > 0:
        print(f"  limit_train={limit_train}, limit_val={limit_val}")
    if resume:
        print("  Resuming from last checkpoint")
    if pretrained_checkpoint:
        print(f"  Transfer learning from {pretrained_checkpoint}")
    if wandb:
        print("  W&B logging enabled")
    train.remote(
        config_path=config,
        epochs=epochs,
        batch_size=batch_size,
        limit_train=limit_train,
        limit_val=limit_val,
        resume=resume,
        wandb=wandb,
        pretrained_checkpoint=pretrained_checkpoint,
    )

    print("\n" + "=" * 60)
    print("Download results:")
    print(
        "  modal volume get inverseops-vol"
        " outputs/training/checkpoints/"
        " outputs/modal_training/checkpoints/"
    )
    print(
        "  modal volume get inverseops-vol"
        " outputs/training/training_summary.json"
        " outputs/modal_training/"
    )
    print("=" * 60)

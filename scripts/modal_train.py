#!/usr/bin/env python3
"""Modal GPU training for InverseOps models.

Prerequisites:
    pip install modal
    modal setup          # one-time auth
    modal secret create wandb-api-key WANDB_API_KEY=<key>  # optional, for W&B

Usage:
    # Default SwinIR denoising
    modal run scripts/modal_train.py

    # NAFNet denoising
    modal run scripts/modal_train.py --config configs/denoise_nafnet_sigma25.yaml

    # SwinIR SR 2x
    modal run scripts/modal_train.py --config configs/sr_swinir_2x.yaml

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

# ---------------------------------------------------------------------------
# Image: deps + pretrained weights + data — all baked in, local-disk fast
# ---------------------------------------------------------------------------

WEIGHTS_DIR = "/cache/inverseops/models"
DATA_DIR = "/data/fmd"

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
    )
    .run_commands(*_download_cmds)
)

# 2) Add fmd.zip and extract into image (cached after first build)
data_image = (
    base_image
    .add_local_file("data/raw/fmd.zip", remote_path="/tmp/fmd.zip", copy=True)
    .run_commands(
        f"mkdir -p {DATA_DIR}",
        (
            f"python -c \"import zipfile; "
            f"zipfile.ZipFile('/tmp/fmd.zip')"
            f".extractall('{DATA_DIR}')\""
        ),
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
    gpu="A100",
    volumes={"/vol": vol},      # only for saving outputs
    secrets=[modal.Secret.from_name("wandb-api-key")],
    timeout=86400,              # 24 hours
)
def train(
    config_path: str = "configs/denoise_swinir.yaml",
    epochs: int = 100,
    batch_size: int = 4,
    limit_train: int = 0,
    limit_val: int = 0,
    resume: bool = False,
    wandb: bool = False,
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
    """
    import json
    import subprocess
    import sys

    import yaml

    # ------------------------------------------------------------------
    # Patch config for Modal paths (data is on local disk in the image)
    # ------------------------------------------------------------------
    with open(f"/app/{config_path}") as f:
        config = yaml.safe_load(f)

    # Rebase data paths: replace local "data/raw/fmd" prefix with Modal's DATA_DIR
    for key in ("train_root", "val_root"):
        local_path = config["data"].get(key, "")
        # Strip the local "data/raw/fmd" prefix, keep the rest (e.g., /fmd/Confocal_FISH)
        if "data/raw/fmd" in local_path:
            suffix = local_path.split("data/raw/fmd", 1)[1]
            config["data"][key] = DATA_DIR + suffix
        else:
            config["data"][key] = DATA_DIR
    config["data"]["num_workers"] = 0
    config["output_dir"] = "/vol/outputs/training"

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

    print(f"Config: {config_path}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, gpu=A100")
    print(f"W&B: {'enabled' if wandb else 'disabled'}")
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
# Generate example images: clean → noisy → denoised
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    volumes={"/vol": vol},
    timeout=1800,
)
def generate_examples(sigma: int = 50):
    """Generate example denoising images for README."""
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import torch
    from PIL import Image

    from inverseops.data.microscopy import MicroscopyDataset

    os.environ["INVERSEOPS_CACHE"] = "/cache/inverseops"

    # Load a test image
    ds = MicroscopyDataset(root_dir=DATA_DIR, split="test", seed=42)
    ds.prepare()
    clean_pil = ds.load_image(0)
    print(f"Loaded test image: {ds.image_path(0)} ({clean_pil.size})")

    # Crop to a nice region (256x256)
    w, h = clean_pil.size
    left = (w - 256) // 2
    top = (h - 256) // 2
    clean_pil = clean_pil.crop((left, top, left + 256, top + 256))

    # Add noise
    clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0
    rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma / 255.0, clean_arr.shape).astype(np.float32)
    noisy_arr = np.clip(clean_arr + noise, 0, 1)

    # Denoise with fine-tuned model
    from inverseops.models.swinir import get_trainable_swinir
    checkpoint = "/vol/outputs/training/checkpoints/best.pt"
    model = get_trainable_swinir(noise_level=25, pretrained=True, device="cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    noisy_pil = Image.fromarray((noisy_arr * 255).astype(np.uint8), mode="L")

    # Run inference
    with torch.no_grad():
        input_t = torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        output_t = model(input_t)
        denoised_arr_t = output_t.squeeze().clamp(0, 1).numpy()

    denoised_pil = Image.fromarray((denoised_arr_t * 255).astype(np.uint8), mode="L")

    # Save
    out_dir = Path("/vol/examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_pil.save(out_dir / "clean.png")
    noisy_pil.save(out_dir / f"noisy_sigma{sigma}.png")
    denoised_pil.save(out_dir / f"denoised_sigma{sigma}.png")

    # Compute PSNR
    denoised_arr = denoised_arr_t
    mse = np.mean((clean_arr - denoised_arr) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))

    mse_noisy = np.mean((clean_arr - noisy_arr) ** 2)
    psnr_noisy = 10 * np.log10(1.0 / max(mse_noisy, 1e-12))

    print(f"Noisy PSNR:    {psnr_noisy:.2f} dB")
    print(f"Denoised PSNR: {psnr:.2f} dB")
    print(f"Saved to {out_dir}")

    vol.commit()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    config: str = "configs/denoise_swinir.yaml",
    epochs: int = 100,
    batch_size: int = 4,
    limit_train: int = 0,
    limit_val: int = 0,
    resume: bool = False,
    wandb: bool = False,
):
    """Train on a Modal GPU with any config."""
    print(f"Config: {config}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}")
    if limit_train > 0:
        print(f"  limit_train={limit_train}, limit_val={limit_val}")
    if resume:
        print("  Resuming from last checkpoint")
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

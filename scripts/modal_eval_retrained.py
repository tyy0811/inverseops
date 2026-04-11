#!/usr/bin/env python3
"""Evaluate retrained SwinIR and NAFNet on W2S test split.

Run AFTER calibration check passes. Uses the same eval harness
that was verified against W2S pretrained baselines.

Usage:
    modal run scripts/modal_eval_retrained.py
"""
from __future__ import annotations
from pathlib import Path
import modal

app = modal.App("inverseops-eval-retrained")
vol = modal.Volume.from_name("inverseops-vol", create_if_missing=True)
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

WEIGHTS_DIR = "/cache/inverseops/models"
PRETRAINED_URLS = [
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth",
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
    "https://github.com/tyy0811/inverseops/releases/download/pretrained-weights-v1/NAFNet-SIDD-width32.pth",
]
_download_cmds = [f"mkdir -p {WEIGHTS_DIR}"] + [
    f"python -c \"import urllib.request; urllib.request.urlretrieve('{url}', '{WEIGHTS_DIR}/{url.split('/')[-1]}')\""
    for url in PRETRAINED_URLS
]

def _source_ignore(path: Path) -> bool:
    skip = {"data", ".git", "__pycache__", "outputs", "artifacts",
            ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    top = path.parts[0] if path.parts else ""
    return top in skip

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "timm>=0.9.0", "numpy>=1.24", "pillow>=10.0",
                 "pydantic>=2.0", "pyyaml>=6.0")
    .run_commands(*_download_cmds)
    .add_local_dir(".", remote_path="/app", ignore=_source_ignore)
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/data": data_vol},
    timeout=3600,
)
def evaluate():
    """Evaluate retrained SwinIR and NAFNet."""
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
        "INVERSEOPS_CACHE": "/cache/inverseops",
    }

    models = [
        ("swinir", "/vol/outputs/training_w2s_swinir/checkpoints/best.pt"),
        ("nafnet", "/vol/outputs/training_w2s_nafnet/checkpoints/best.pt"),
    ]

    for model_name, ckpt_path in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  ERROR: checkpoint not found at {ckpt_path}")
            continue

        cmd = [
            sys.executable, "/app/scripts/run_evaluation.py",
            "--data-root", "/data/w2s/data/normalized",
            "--splits-path", "/app/inverseops/data/splits.json",
            "--device", "cuda",
            "--checkpoint", ckpt_path,
            "--model", model_name,
        ]

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  ERROR: evaluation failed (exit code {result.returncode})")


@app.local_entrypoint()
def main():
    print("Evaluating retrained models...")
    evaluate.remote()

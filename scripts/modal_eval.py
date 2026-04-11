#!/usr/bin/env python3
"""Run evaluation on Modal where data and checkpoints live.

Usage:
    # Calibration check (W2S pretrained baselines)
    modal run scripts/modal_eval.py --calibration

    # Evaluate retrained checkpoint
    modal run scripts/modal_eval.py \
        --checkpoint /vol/outputs/training_w2s_swinir/checkpoints/best.pt \
        --model swinir
"""
from __future__ import annotations

import modal

app = modal.App("inverseops-eval")

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
    (
        f"python -c \"import urllib.request; "
        f"urllib.request.urlretrieve('{url}', "
        f"'{WEIGHTS_DIR}/{url.split('/')[-1]}')\""
    )
    for url in PRETRAINED_URLS
]

from pathlib import Path

def _source_ignore(path: Path) -> bool:
    skip = {"data", ".git", "__pycache__", "outputs", "artifacts",
            ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    top = path.parts[0] if path.parts else ""
    return top in skip

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "timm>=0.9.0",
        "numpy>=1.24",
        "pillow>=10.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
    )
    .run_commands(*_download_cmds)
    .add_local_dir(".", remote_path="/app", ignore=_source_ignore)
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/data": data_vol},
    timeout=7200,
)
def run_eval(
    calibration: bool = False,
    checkpoint: str = "",
    model: str = "swinir",
    output_csv: str = "",
):
    """Run evaluation on Modal."""
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
        "INVERSEOPS_CACHE": "/cache/inverseops",
    }

    cmd = [
        sys.executable, "/app/scripts/run_evaluation.py",
        "--data-root", "/data/w2s/data/normalized",
        "--splits-path", "/app/inverseops/data/splits.json",
        "--device", "cuda",
    ]

    if calibration:
        cmd += [
            "--calibration",
            "--calibration-dir", "/data/w2s/net_data/trained_denoisers/",
        ]
        if output_csv:
            cmd += ["--output-csv", output_csv]
    else:
        if not checkpoint:
            print("ERROR: --checkpoint required when not using --calibration")
            return
        cmd += [
            "--checkpoint", checkpoint,
            "--model", model,
        ]
        if output_csv:
            cmd += ["--output-csv", output_csv]

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed (exit code {result.returncode})")


@app.local_entrypoint()
def main(
    calibration: bool = False,
    checkpoint: str = "",
    model: str = "swinir",
    output_csv: str = "",
):
    """Run evaluation on Modal GPU."""
    if calibration:
        print("Running calibration check (W2S pretrained baselines)...")
    else:
        print(f"Evaluating {model} checkpoint: {checkpoint}")
    run_eval.remote(
        calibration=calibration,
        checkpoint=checkpoint,
        model=model,
        output_csv=output_csv,
    )

#!/usr/bin/env python3
"""Generate IXI denoising visual comparison figure for README.

Picks one test subject, extracts a central axial slice, applies Rician noise
(same sigma as training), runs the IXI-finetuned SwinIR, saves:
  - Clean reference (percentile-normalized to [0, 1])
  - Noisy input (Rician noise at sigma=0.10)
  - SwinIR denoised
  - Combined 3-panel figure

Usage:
    modal run scripts/modal_generate_ixi_figure.py
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("inverseops-figure-ixi")
vol = modal.Volume.from_name("inverseops-vol", create_if_missing=True)
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

WEIGHTS_DIR = "/cache/inverseops/models"
PRETRAINED_URLS = [
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
]
_download_cmds = [f"mkdir -p {WEIGHTS_DIR}"] + [
    (
        f'python -c "import urllib.request; '
        f"urllib.request.urlretrieve('{url}', "
        f"'{WEIGHTS_DIR}/{url.split('/')[-1]}')\""
    )
    for url in PRETRAINED_URLS
]


def _source_ignore(path: Path) -> bool:
    skip = {
        "data",
        ".git",
        "__pycache__",
        "outputs",
        "artifacts",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
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
        "matplotlib>=3.7",
        "nibabel>=5.0",
    )
    .run_commands(*_download_cmds)
    .add_local_dir(".", remote_path="/app", ignore=_source_ignore)
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/data": data_vol},
    timeout=600,
)
def generate(sigma: float = 0.10):
    """Generate IXI denoising visual comparison figure."""
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    sys.path.insert(0, "/app")
    from inverseops.data import build_dataset
    from inverseops.models import build_model
    from scripts.run_evaluation import psnr_tensor, ssim_tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build test dataset (full slices, no cropping)
    config = {
        "data": {
            "dataset": "ixi",
            "train_root": "/data/ixi/T1",
            "splits_path": "/app/inverseops/data/splits.json",
            "patch_size": 0,
            "sigma": sigma,
        },
    }
    test_dataset = build_dataset(config, split="test", training=False)
    print(f"Test dataset: {len(test_dataset)} samples")

    # Pick the slice with highest variance (most anatomical detail)
    # Sample the first 200 slices for speed
    n_sample = min(200, len(test_dataset))
    best_idx, best_var = 0, 0.0
    for i in range(0, n_sample, 2):
        s = test_dataset[i]
        v = float(s["target"].var())
        if v > best_var:
            best_idx, best_var = i, v
    print(f"Selected slice index {best_idx} (var={best_var:.4f})")

    sample = test_dataset[best_idx]
    clean = sample["target"].squeeze(0).numpy()  # (H, W)
    noisy = sample["input"].squeeze(0).numpy()
    subject_id = sample["subject_id"]
    print(f"Subject: IXI{subject_id:03d}, shape={clean.shape}")

    # Load SwinIR checkpoint
    model_config = {
        "model": {"name": "swinir", "pretrained": False},
        "task": "denoise",
    }
    model = build_model(model_config, device=device)
    ckpt_path = "/vol/outputs/training_ixi_swinir/checkpoints/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Run inference
    inp_tensor = sample["input"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(inp_tensor).squeeze().cpu().numpy()
    # Clamp to valid range
    output = np.clip(output, 0, 1)

    # Compute per-image metrics for the caption
    data_range = 1.0
    clean_t = torch.from_numpy(clean)
    noisy_t = torch.from_numpy(noisy)
    output_t = torch.from_numpy(output)

    noisy_psnr = psnr_tensor(clean_t, noisy_t, data_range=data_range)
    noisy_ssim = ssim_tensor(clean_t, noisy_t, data_range=data_range)
    denoised_psnr = psnr_tensor(clean_t, output_t, data_range=data_range)
    denoised_ssim = ssim_tensor(clean_t, output_t, data_range=data_range)

    print(f"\nNoisy:    PSNR={noisy_psnr:.2f} dB  SSIM={noisy_ssim:.4f}")
    print(f"SwinIR:   PSNR={denoised_psnr:.2f} dB  SSIM={denoised_ssim:.4f}")

    # Build combined figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    vmin, vmax = 0.0, 1.0

    panels = [
        ("Clean reference", clean),
        (f"Noisy input (sigma={sigma})\nPSNR={noisy_psnr:.1f} dB  SSIM={noisy_ssim:.3f}", noisy),
        (f"SwinIR denoised\nPSNR={denoised_psnr:.1f} dB  SSIM={denoised_ssim:.3f}", output),
    ]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"IXI Brain MRI Denoising — Subject IXI{subject_id:03d}, Rician noise sigma={sigma}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    out_dir = Path("/vol/figures/ixi")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "ixi_denoising_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nCombined figure saved to {fig_path}")

    # Save individual panels as 8-bit PNGs
    from PIL import Image

    def to_uint8(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr * 255, 0, 255).astype(np.uint8)

    for name, arr in [
        ("clean", clean),
        ("noisy", noisy),
        ("swinir_denoised", output),
    ]:
        pil = Image.fromarray(to_uint8(arr), mode="L")
        pil.save(out_dir / f"ixi_{name}.png")
        print(f"  Saved ixi_{name}.png")

    vol.commit()
    print("\nDone. Download with:")
    print("  modal volume get inverseops-vol figures/ixi figures/ixi")


@app.local_entrypoint()
def main(sigma: float = 0.10):
    print(f"Generating IXI denoising figure (sigma={sigma})...")
    generate.remote(sigma=sigma)

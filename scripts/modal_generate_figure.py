#!/usr/bin/env python3
"""Generate visual comparison figure for README.

Picks one test FoV, runs both models, saves:
  - Noisy input (avg1)
  - Clean reference (avg400)
  - SwinIR denoised
  - NAFNet denoised
  - SIM HR ground truth (2x resolution)

Usage:
    modal run scripts/modal_generate_figure.py
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("inverseops-figure")
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
def generate():
    """Generate visual comparison figure."""
    import json
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    sys.path.insert(0, "/app")
    from inverseops.models import build_model

    W2S_MEAN = 154.54
    W2S_STD = 66.03
    data_root = Path("/data/w2s/data/normalized")

    with open("/app/inverseops/data/splits.json") as f:
        splits = json.load(f)
    test_fovs = splits["w2s"]["test"]

    device = "cuda"

    # Pick a FoV with interesting structure — try a few and pick the one
    # with highest std (most structural detail)
    best_fov, best_wl, best_std = None, None, 0
    for fov in test_fovs[:5]:
        for wl in [0, 1, 2]:
            arr = np.load(data_root / "avg400" / f"{fov:03d}_{wl}.npy")
            s = arr.std()
            if s > best_std:
                best_fov, best_wl, best_std = fov, wl, s

    fov_id = best_fov
    wl = best_wl
    print(f"Selected FoV {fov_id}, wavelength {wl} (std={best_std:.4f})")

    # Load images
    noisy_npy = np.load(data_root / "avg1" / f"{fov_id:03d}_{wl}.npy").astype(
        np.float32
    )
    clean_npy = np.load(data_root / "avg400" / f"{fov_id:03d}_{wl}.npy").astype(
        np.float32
    )
    sim_npy = (
        np.load(
            data_root
            / ".."
            / ".."
            / "data"
            / "normalized"
            / ".."
            / ".."
            / "data"
            / "normalized"
            / "sim"
            / f"{fov_id:03d}_{wl}.npy"
        ).astype(np.float32)
        if (data_root / "sim" / f"{fov_id:03d}_{wl}.npy").exists()
        else None
    )

    # Try loading SIM from the right path
    sim_path = (
        data_root.parent.parent
        / "data"
        / "normalized"
        / "sim"
        / f"{fov_id:03d}_{wl}.npy"
    )
    if not sim_path.exists():
        sim_path = data_root / "sim" / f"{fov_id:03d}_{wl}.npy"
    if sim_path.exists():
        sim_npy = np.load(sim_path).astype(np.float32)
        print(f"SIM loaded: {sim_npy.shape}")
    else:
        sim_npy = None
        print(f"SIM not found at {sim_path}")

    # Denormalize to [0, 255] for display
    def to_display(npy):
        return np.clip(npy * W2S_STD + W2S_MEAN, 0, 255)

    noisy_display = to_display(noisy_npy)
    clean_display = to_display(clean_npy)
    sim_display = to_display(sim_npy) if sim_npy is not None else None

    # Run models
    noisy_tensor = (
        torch.from_numpy(noisy_npy).unsqueeze(0).unsqueeze(0).float().to(device)
    )

    results = {}
    for model_name, ckpt_path in [
        ("SwinIR", "/vol/outputs/training_w2s_swinir/checkpoints/best.pt"),
        ("NAFNet", "/vol/outputs/training_w2s_nafnet/checkpoints/best.pt"),
    ]:
        config = {
            "model": {"name": model_name.lower(), "pretrained": False},
            "task": "denoise",
        }
        model = build_model(config, device=device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            output = model(noisy_tensor).squeeze().cpu().numpy()

        results[model_name] = to_display(output)
        print(
            f"{model_name} output range: "
            f"[{results[model_name].min():.1f}, "
            f"{results[model_name].max():.1f}]"
        )

    # Crop to a 256x256 region with interesting structure
    # Find the region with highest variance in the clean image
    ps = 256
    h, w = clean_display.shape
    best_y, best_x, best_var = 0, 0, 0
    for y in range(0, h - ps, 32):
        for x in range(0, w - ps, 32):
            v = clean_display[y : y + ps, x : x + ps].var()
            if v > best_var:
                best_y, best_x, best_var = y, x, v

    cy, cx = best_y, best_x
    print(f"Crop region: ({cy}, {cx}) to ({cy + ps}, {cx + ps})")

    def crop(img, scale=1):
        return img[cy * scale : (cy + ps) * scale, cx * scale : (cx + ps) * scale]

    # Build figure
    has_sim = sim_display is not None
    n_panels = 5 if has_sim else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))

    vmin = crop(clean_display).min()
    vmax = crop(clean_display).max()

    panels = [
        ("Noisy (avg1)", crop(noisy_display)),
        ("Clean (avg400)", crop(clean_display)),
        ("SwinIR (34.31 dB)", crop(results["SwinIR"])),
        ("NAFNet (34.05 dB)", crop(results["NAFNet"])),
    ]
    if has_sim:
        panels.append(("SIM Ground Truth (2x)", crop(sim_display, scale=2)))

    for ax, (title, img) in zip(axes, panels):
        if "SIM" in title:
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"W2S Fluorescence Microscopy Denoising — FoV {fov_id}, Wavelength {wl}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    # Save
    out_dir = Path("/vol/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "w2s_denoising_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to {fig_path}")

    # Also save individual images for flexibility
    from PIL import Image

    for name, img in panels:
        clean_name = name.split("(")[0].strip().lower().replace(" ", "_")
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")
        pil.save(out_dir / f"{clean_name}.png")
        print(f"  Saved {clean_name}.png")

    vol.commit()
    print("\nDone. Download with:")
    print("  modal volume get inverseops-vol figures/ figures/")


@app.local_entrypoint()
def main():
    print("Generating visual comparison figure...")
    generate.remote()

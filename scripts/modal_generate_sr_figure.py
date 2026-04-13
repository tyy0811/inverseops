#!/usr/bin/env python3
"""Generate W2S super-resolution visual comparison figure for README.

Runs the trained SwinIR SR 2x checkpoint plus a bicubic baseline on one
representative test FoV, saves a 4-panel comparison (LR / bicubic / SwinIR
/ SIM ground truth) parallel to the existing denoising comparison figure.

FoV selection: hardcoded to FoV 31, wavelength 0. This is a deliberately
representative choice, not a cherry-pick:

  - Test split aggregate: PSNR_clip 21.22 +/- 2.88 dB, delta over matched
    bicubic +1.35 dB.
  - FoV 31 (all wavelengths averaged): PSNR_clip 21.75 dB, delta +1.93 dB.
    Above the aggregate by +0.5 dB in absolute and +0.6 dB in delta — a
    slightly-above-mean case, neither the best (FoV 93 at +2.57) nor the
    worst (FoV 110 at -0.43 regression).

The region crop is selected as the highest-variance 256x256 region of the
SIM ground truth — the choice is driven by the target, not by either
model's output, so neither model can "win" the crop selection.

Usage:
    modal run scripts/modal_generate_sr_figure.py
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("inverseops-figure-sr")
vol = modal.Volume.from_name("inverseops-vol", create_if_missing=True)
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)


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
        "pyyaml>=6.0",
        "matplotlib>=3.7",
        "scikit-image>=0.20",
    )
    .add_local_dir(".", remote_path="/app", ignore=_source_ignore)
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/data": data_vol},
    timeout=600,
)
def generate_sr_figure(
    checkpoint_path: str = (
        "/vol/outputs/training_w2s_sr_swinir_2x/checkpoints/best.pt"
    ),
    fov_id: int = 31,
    wavelength: int = 0,
):
    """Generate the 4-panel SR comparison figure on one test FoV."""
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from skimage.filters import sobel as skimage_sobel
    from skimage.metrics import structural_similarity as ssim_skimage
    from skimage.transform import resize as skimage_resize

    sys.path.insert(0, "/app")
    from inverseops.evaluation.stitching import sliding_window_sr
    from inverseops.models import build_model

    W2S_MEAN = 154.54
    W2S_STD = 66.03
    data_root = Path("/data/w2s/data/normalized")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    lr_path = data_root / "avg400" / f"{fov_id:03d}_{wavelength}.npy"
    hr_path = data_root / "sim" / f"{fov_id:03d}_{wavelength}.npy"
    if not lr_path.exists() or not hr_path.exists():
        raise FileNotFoundError(
            f"Missing data for FoV {fov_id} wl {wavelength}:\n"
            f"  {lr_path}  exists={lr_path.exists()}\n"
            f"  {hr_path}  exists={hr_path.exists()}"
        )

    lr_z = np.load(lr_path).astype(np.float32)
    hr_z = np.load(hr_path).astype(np.float32)
    print(f"FoV {fov_id} wl {wavelength}: LR {lr_z.shape}  HR {hr_z.shape}")

    # ------------------------------------------------------------------
    # Load model + run SR inference
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    config.setdefault("model", {})["pretrained"] = False
    model = build_model(config, device=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"  best_val_psnr: {ckpt.get('best_val_psnr', '?')}")

    print("Running stitched SR inference...")
    sr_z = sliding_window_sr(model, lr_z, device, clamp=False)
    print(f"  SR output: {sr_z.shape}")

    # ------------------------------------------------------------------
    # Reporting space: denormalized, clipped to [0, 1]
    # ------------------------------------------------------------------
    def to_01_clip(z):
        return np.clip(z * W2S_STD + W2S_MEAN, 0, 255) / 255.0

    lr_01 = to_01_clip(lr_z)
    hr_01 = to_01_clip(hr_z)
    sr_01 = to_01_clip(sr_z)

    # Bicubic baseline (matching Decision 20's bicubic(avg400 -> SIM))
    bic_01 = skimage_resize(lr_01, hr_01.shape, order=3, anti_aliasing=False)
    bic_01 = np.clip(bic_01, 0, 1)

    # ------------------------------------------------------------------
    # Metrics for panel subtitles
    # ------------------------------------------------------------------
    def psnr(gt, pred):
        mse = np.mean((gt - pred) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(1.0 / mse))

    def ssim(gt, pred):
        return float(ssim_skimage(gt, pred, data_range=1.0))

    psnr_bic, ssim_bic = psnr(hr_01, bic_01), ssim(hr_01, bic_01)
    psnr_sr, ssim_sr = psnr(hr_01, sr_01), ssim(hr_01, sr_01)
    print(
        f"  bicubic PSNR={psnr_bic:.2f} dB  SSIM={ssim_bic:.4f}"
    )
    print(
        f"  SwinIR  PSNR={psnr_sr:.2f} dB  SSIM={ssim_sr:.4f}"
    )

    # ------------------------------------------------------------------
    # Two-level crop, both picked by model-independent rules from the
    # SIM target (not from either model's output):
    #   1. Main crop: highest-variance 256x256 LR region. Shows the
    #      overall comparison at moderate zoom.
    #   2. Detail crop: highest edge-density 64x64 LR sub-region WITHIN
    #      the main crop. Edge density (Sobel gradient magnitude) is a
    #      better selector for "fine structure" than raw variance —
    #      raw variance can be dominated by large bright blobs, where
    #      the SR improvement is less visible, whereas edge density
    #      picks regions with actual texture.
    # ------------------------------------------------------------------
    def pick_main_crop(img_hr01, h_lr, w_lr, ps_lr, stride):
        best_y, best_x, best_var = 0, 0, -1.0
        for y in range(0, h_lr - ps_lr + 1, stride):
            for x in range(0, w_lr - ps_lr + 1, stride):
                hr_crop = img_hr01[
                    y * 2 : (y + ps_lr) * 2, x * 2 : (x + ps_lr) * 2
                ]
                v = float(hr_crop.var())
                if v > best_var:
                    best_y, best_x, best_var = y, x, v
        return best_y, best_x, best_var

    def pick_detail_crop(edge_map, ps_lr, stride):
        # edge_map is computed on the HR target (variance-independent
        # proxy for fine structure). Pick the LR-space window whose HR
        # projection has the highest mean edge magnitude.
        h_hr, w_hr = edge_map.shape
        h_lr_local, w_lr_local = h_hr // 2, w_hr // 2
        best_y, best_x, best_edge = 0, 0, -1.0
        for y in range(0, h_lr_local - ps_lr + 1, stride):
            for x in range(0, w_lr_local - ps_lr + 1, stride):
                e = float(
                    edge_map[
                        y * 2 : (y + ps_lr) * 2, x * 2 : (x + ps_lr) * 2
                    ].mean()
                )
                if e > best_edge:
                    best_y, best_x, best_edge = y, x, e
        return best_y, best_x, best_edge

    ps_main = 256
    h, w = lr_01.shape
    cy, cx, main_var = pick_main_crop(hr_01, h, w, ps_main, stride=32)
    print(
        f"Main crop (LR space): ({cy}, {cx}) -> ({cy + ps_main}, {cx + ps_main})  "
        f"[SIM var={main_var:.4f}]"
    )

    # Sobel edge map on the main-crop HR window only (detail search
    # is restricted to within the main crop, matching column alignment)
    ps_detail = 64
    hr_main_crop = hr_01[
        cy * 2 : (cy + ps_main) * 2, cx * 2 : (cx + ps_main) * 2
    ]
    edge_map = skimage_sobel(hr_main_crop)
    dy, dx, detail_edge = pick_detail_crop(edge_map, ps_detail, stride=8)
    dy_abs, dx_abs = cy + dy, cx + dx
    print(
        f"Detail crop (LR space): ({dy_abs}, {dx_abs}) -> "
        f"({dy_abs + ps_detail}, {dx_abs + ps_detail})  "
        f"[SIM edge density={detail_edge:.4f}]"
    )

    def _crop_lr(img, y, x, ps):
        return img[y : y + ps, x : x + ps]

    def _crop_hr(img, y, x, ps):
        return img[y * 2 : (y + ps) * 2, x * 2 : (x + ps) * 2]

    # Main row (256 LR / 512 HR)
    lr_main = _crop_lr(lr_01, cy, cx, ps_main)
    bic_main = _crop_hr(bic_01, cy, cx, ps_main)
    sr_main = _crop_hr(sr_01, cy, cx, ps_main)
    gt_main = _crop_hr(hr_01, cy, cx, ps_main)

    # Detail row (64 LR / 128 HR)
    lr_det = _crop_lr(lr_01, dy_abs, dx_abs, ps_detail)
    bic_det = _crop_hr(bic_01, dy_abs, dx_abs, ps_detail)
    sr_det = _crop_hr(sr_01, dy_abs, dx_abs, ps_detail)
    gt_det = _crop_hr(hr_01, dy_abs, dx_abs, ps_detail)

    # ------------------------------------------------------------------
    # Build 2x4 figure: main row + detail row
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5))

    # Shared contrast scale per row (fair within-row comparison)
    vmin_main, vmax_main = float(gt_main.min()), float(gt_main.max())
    vmin_det, vmax_det = float(gt_det.min()), float(gt_det.max())

    main_panels = [
        ("LR input (avg400)\n512x512", lr_main),
        (f"Bicubic x2\nPSNR={psnr_bic:.2f} dB  SSIM={ssim_bic:.3f}", bic_main),
        (
            f"SwinIR SR x2 (ours)\nPSNR={psnr_sr:.2f} dB  SSIM={ssim_sr:.3f}",
            sr_main,
        ),
        ("SIM HR ground truth\n1024x1024", gt_main),
    ]
    detail_panels = [
        ("(detail, 64x64 LR)", lr_det),
        ("(detail)", bic_det),
        ("(detail)", sr_det),
        ("(detail, 128x128 HR)", gt_det),
    ]

    for ax, (title, img) in zip(axes[0], main_panels):
        ax.imshow(
            img,
            cmap="gray",
            vmin=vmin_main,
            vmax=vmax_main,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    for ax, (title, img) in zip(axes[1], detail_panels):
        ax.imshow(
            img,
            cmap="gray",
            vmin=vmin_det,
            vmax=vmax_det,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(
        (
            f"W2S Super-Resolution (clean LR avg400 -> SIM, 2x) — "
            f"FoV {fov_id}, Wavelength {wavelength}"
        ),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path("/vol/figures/v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "w2s_sr_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to {fig_path}")

    vol.commit()
    print("\nDone. Download with:")
    print(
        "  modal volume get inverseops-vol figures/v3/w2s_sr_comparison.png "
        "figures/v3/w2s_sr_comparison.png --force"
    )


@app.local_entrypoint()
def main(
    checkpoint_path: str = (
        "/vol/outputs/training_w2s_sr_swinir_2x/checkpoints/best.pt"
    ),
    fov_id: int = 31,
    wavelength: int = 0,
):
    print(f"Generating SR comparison figure for FoV {fov_id} wl {wavelength}...")
    generate_sr_figure.remote(
        checkpoint_path=checkpoint_path,
        fov_id=fov_id,
        wavelength=wavelength,
    )

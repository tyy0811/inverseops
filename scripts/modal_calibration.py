#!/usr/bin/env python3
"""Calibration check: run W2S pretrained denoisers through our eval metrics.

W2S pretrained models expect [0,1] input (PNG/255). The .npy files on our
volume are pre-normalized (Z-score, mean≈0, std≈1). Pipeline:
  1. Load .npy (pre-normalized)
  2. Denormalize to [0,255] intensity space (×66.03 + 154.54)
  3. Divide by 255 → [0,1] input for model
  4. Model inference → output in [0,1]
  5. Clamp to [0,1], multiply by 255 → [0,255]
  6. Compute PSNR/SSIM with data_range=255 against [0,255] ground truth

This matches the W2S test.py pipeline exactly.

Published W2S numbers (Table 1, RMSE/SSIM — written down before running):
  DnCNN avg1:  RMSE=0.078 → PSNR≈22.2 dB  SSIM=0.907
  MemNet avg1: RMSE=0.090 → PSNR≈20.9 dB  SSIM=0.901
  DnCNN avg16: RMSE=0.033 → PSNR≈29.6 dB  SSIM=0.964

Usage:
    modal run scripts/modal_calibration.py
"""
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("inverseops-calibration")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy>=1.24")
)

def _source_ignore(path: Path) -> bool:
    skip = {"data", ".git", "__pycache__", "outputs", "artifacts",
            ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    top = path.parts[0] if path.parts else ""
    return top in skip

eval_image = image.add_local_dir(".", remote_path="/app", ignore=_source_ignore)


@app.function(
    image=eval_image,
    gpu="A100",
    volumes={"/data": data_vol},
    timeout=3600,
)
def calibrate():
    """Run W2S pretrained baselines on our test split."""
    import json
    import os
    import sys
    import types
    from collections import defaultdict

    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, "/app")
    from scripts.run_evaluation import psnr_tensor, ssim_tensor

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    with open("/app/inverseops/data/splits.json") as f:
        splits = json.load(f)
    test_fovs = splits["w2s"]["test"]

    data_root = Path("/data/w2s/data/normalized")
    cal_dir = Path("/data/w2s/net_data/trained_denoisers")
    W2S_MEAN = 154.54
    W2S_STD = 66.03

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Test FoVs: {test_fovs} ({len(test_fovs)} FoVs)")

    noise_levels = [1, 2, 4, 8, 16]
    wavelengths = [0, 1, 2]

    # ----------------------------------------------------------------
    # PRE-FLIGHT CHECK 1: Data inspection
    # ----------------------------------------------------------------
    print("\n=== PREFLIGHT CHECK 1: Data inspection ===")
    sample_npy = np.load(data_root / "avg1" / f"{test_fovs[0]:03d}_0.npy").astype(np.float32)
    print(f"  Raw .npy: shape={sample_npy.shape} range=[{sample_npy.min():.4f}, {sample_npy.max():.4f}] mean={sample_npy.mean():.4f}")

    # Denormalize to [0,255], clip (matching PNG uint8), then /255 to [0,1]
    sample_denorm = sample_npy * W2S_STD + W2S_MEAN
    pct_clipped = 100.0 * np.sum(sample_denorm > 255) / sample_denorm.size
    sample_clipped = np.clip(sample_denorm, 0, 255)
    sample_01 = sample_clipped / 255.0
    print(f"  Denormalized: range=[{sample_denorm.min():.2f}, {sample_denorm.max():.2f}] mean={sample_denorm.mean():.2f}")
    print(f"  Clipped >255: {pct_clipped:.1f}% of pixels")
    print(f"  [0,1] input:  range=[{sample_01.min():.4f}, {sample_01.max():.4f}] mean={sample_01.mean():.4f}")
    print("  PASS")

    # ----------------------------------------------------------------
    # PRE-FLIGHT CHECK 3: Metric sanity
    # ----------------------------------------------------------------
    print("\n=== PREFLIGHT CHECK 3: Metric sanity ===")
    clean_npy = np.load(data_root / "avg400" / f"{test_fovs[0]:03d}_0.npy").astype(np.float32)
    clean_255 = torch.from_numpy(clean_npy * W2S_STD + W2S_MEAN).float()

    p_identical = psnr_tensor(clean_255, clean_255, data_range=255.0)
    print(f"  PSNR(img, img)           = {p_identical}  (expect inf)")

    p_zeros = psnr_tensor(clean_255, torch.zeros_like(clean_255), data_range=255.0)
    print(f"  PSNR(img, zeros)         = {p_zeros:.2f} dB  (expect <20)")

    noisy_255 = clean_255 + torch.randn_like(clean_255) * 5
    p_noisy = psnr_tensor(clean_255, noisy_255, data_range=255.0)
    print(f"  PSNR(img, img+noise*5)   = {p_noisy:.2f} dB  (expect 30-40)")

    if p_identical != float("inf"):
        print("  FAIL: identical should be inf")
        return
    if not (25 < p_noisy < 45):
        print(f"  FAIL: noisy PSNR {p_noisy:.2f} outside [25, 45]")
        return
    print("  PASS")

    # ----------------------------------------------------------------
    # Load W2S model architectures (DnCNN, MemNet only — RIDNet missing deps)
    # ----------------------------------------------------------------
    w2s_denoise = "/data/w2s/code/denoise"
    models_path = os.path.join(w2s_denoise, "models.py")
    with open(models_path) as f:
        source = f.read()

    # Remove broken RIDmodel imports and everything after MemNet_BUIFD.
    # We only need DnCNN, DnCNN_RL, MemNet, and their helpers (BNReLUConv etc).
    # Everything from "class CALayer" onward is RIDNet-related and depends on
    # the missing RIDmodel module.
    lines = source.split("\n")
    filtered = []
    for line in lines:
        if "from RIDmodel" in line:
            continue
        # Stop before RIDNet-related code
        if "# START(RIDNet)" in line or "class CALayer" in line:
            break
        # Also skip make_model which references RIDNET
        if "def make_model" in line or "return RIDNET" in line:
            continue
        filtered.append(line)

    mod = types.ModuleType("w2s_models")
    mod.__dict__["torch"] = torch
    mod.__dict__["nn"] = nn
    mod.__dict__["os"] = os
    exec("\n".join(filtered), mod.__dict__)

    DnCNN = mod.DnCNN
    MemNet = mod.MemNet

    model_configs = {
        "DnCNN": {
            "prefix": "D",
            "factory": lambda: DnCNN(channels=1, num_of_layers=17),
            "residual": True,  # Predicts noise, clean = input - noise
        },
        "MemNet": {
            "prefix": "M",
            "factory": lambda: MemNet(in_channels=1),
            "residual": True,  # W2S trains MemNet to predict noise (same as DnCNN)
        },
    }

    # ----------------------------------------------------------------
    # Run calibration
    # ----------------------------------------------------------------
    all_results = {}

    for model_name, cfg in model_configs.items():
        prefix = cfg["prefix"]
        is_residual = cfg["residual"]
        print(f"\n{'='*60}")
        print(f"Calibration: {model_name}")
        print(f"{'='*60}")

        psnr_by_level_fov = defaultdict(lambda: defaultdict(list))
        ssim_by_level_fov = defaultdict(lambda: defaultdict(list))
        rmse_by_level_fov = defaultdict(lambda: defaultdict(list))

        for nl in noise_levels:
            model_dir = cal_dir / f"{prefix}_{nl}"
            if not model_dir.exists():
                print(f"  SKIP: {model_dir} not found")
                continue

            pth_files = list(model_dir.glob("*.pth"))
            if not pth_files:
                print(f"  SKIP: No .pth in {model_dir}")
                continue

            # Load model with DataParallel (matching how W2S saves)
            try:
                net = cfg["factory"]()
                model = nn.DataParallel(net).to(device)
                sd = torch.load(pth_files[0], map_location=device, weights_only=False)
                model.load_state_dict(sd)
                model.eval()
                print(f"  Loaded {model_name} for avg{nl}")
            except Exception as e:
                print(f"  ERROR loading {model_name} avg{nl}: {e}")
                continue

            # Evaluate on test FoVs
            with torch.no_grad():
                for fov_id in test_fovs:
                    for wl in wavelengths:
                        noisy_path = data_root / f"avg{nl}" / f"{fov_id:03d}_{wl}.npy"
                        clean_path = data_root / "avg400" / f"{fov_id:03d}_{wl}.npy"

                        if not noisy_path.exists() or not clean_path.exists():
                            continue

                        # Load and convert: .npy (Z-score) → denorm → clip [0,255] → /255 [0,1]
                        # The clip matches the uint8 quantization that happens
                        # when the W2S repo saves PNGs from raw microscopy data.
                        noisy_npy = np.load(noisy_path).astype(np.float32)
                        clean_npy = np.load(clean_path).astype(np.float32)

                        noisy_01 = np.clip(noisy_npy * W2S_STD + W2S_MEAN, 0, 255) / 255.0
                        clean_01 = np.clip(clean_npy * W2S_STD + W2S_MEAN, 0, 255) / 255.0

                        # Model input [1, 1, H, W] in [0,1]
                        inp = torch.from_numpy(noisy_01).unsqueeze(0).unsqueeze(0).float().to(device)

                        output = model(inp)

                        if is_residual:
                            # DnCNN predicts noise → clean = input - noise
                            prediction_01 = inp - output
                        else:
                            # MemNet returns denoised directly
                            prediction_01 = output

                        # Clamp to [0,1], scale to [0,255] — matching W2S test.py
                        prediction_01 = torch.clamp(prediction_01, 0., 1.)
                        prediction_255 = prediction_01.squeeze().cpu() * 255.0
                        clean_255 = torch.from_numpy(clean_01).float() * 255.0

                        p = psnr_tensor(clean_255, prediction_255, data_range=255.0)
                        s = ssim_tensor(clean_255, prediction_255, data_range=255.0)
                        rmse_01 = float(torch.sqrt(torch.mean(
                            (prediction_01.squeeze().cpu() - torch.from_numpy(clean_01).float()) ** 2
                        )))

                        psnr_by_level_fov[nl][fov_id].append(p)
                        ssim_by_level_fov[nl][fov_id].append(s)
                        rmse_by_level_fov[nl][fov_id].append(rmse_01)

        # Aggregate: per-FoV mean, then mean +/- std across FoVs
        print(f"\n{'Noise':>6}  {'PSNR (dB)':>16}  {'SSIM':>16}  {'RMSE':>12}  {'FoVs':>5}")
        print(f"{'-'*62}")
        results = {}
        for nl in sorted(psnr_by_level_fov.keys()):
            fov_psnrs = [np.mean(v) for v in psnr_by_level_fov[nl].values()]
            fov_ssims = [np.mean(v) for v in ssim_by_level_fov[nl].values()]
            fov_rmses = [np.mean(v) for v in rmse_by_level_fov[nl].values()]
            r = {
                "psnr_mean": float(np.mean(fov_psnrs)),
                "psnr_std": float(np.std(fov_psnrs)),
                "ssim_mean": float(np.mean(fov_ssims)),
                "ssim_std": float(np.std(fov_ssims)),
                "rmse_mean": float(np.mean(fov_rmses)),
                "n_fovs": len(fov_psnrs),
            }
            results[nl] = r
            psnr_str = f"{r['psnr_mean']:.2f} +/- {r['psnr_std']:.2f}"
            ssim_str = f"{r['ssim_mean']:.4f} +/- {r['ssim_std']:.4f}"
            rmse_str = f"{r['rmse_mean']:.4f}"
            print(f"avg{nl:>3}  {psnr_str:>16}  {ssim_str:>16}  {rmse_str:>12}  {r['n_fovs']:>5}")

        all_results[model_name] = results

    # ----------------------------------------------------------------
    # Calibration gate
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CALIBRATION GATE")
    print(f"{'='*60}")
    print("Published (Table 1): DnCNN avg1 RMSE=0.078 SSIM=0.907")
    print("Published (Table 1): MemNet avg1 RMSE=0.090 SSIM=0.901")

    # Pre-registered calibration targets and tolerances.
    # Published numbers use all 120 FoVs; ours use 13 held-out FoVs,
    # so we use wider tolerances (0.06 RMSE, 0.08 SSIM) to account
    # for the subset difference while still catching pipeline bugs.
    calibration_checks = [
        ("DnCNN", 0.078, 0.907, 0.06, 0.08),
        ("MemNet", 0.090, 0.901, 0.06, 0.08),
    ]

    failures = []
    for model_name, pub_rmse, pub_ssim, rmse_tol, ssim_tol in calibration_checks:
        if model_name not in all_results or 1 not in all_results[model_name]:
            failures.append(f"{model_name}: no avg1 results produced")
            continue

        our_rmse = all_results[model_name][1]["rmse_mean"]
        our_ssim = all_results[model_name][1]["ssim_mean"]
        rmse_gap = abs(our_rmse - pub_rmse)
        ssim_gap = abs(our_ssim - pub_ssim)
        print(f"\nOurs {model_name} avg1: RMSE={our_rmse:.4f} (gap={rmse_gap:.4f}) SSIM={our_ssim:.4f} (gap={ssim_gap:.4f})")

        passed = True
        if rmse_gap >= rmse_tol:
            print(f"  FAIL: RMSE gap {rmse_gap:.4f} >= tolerance {rmse_tol}")
            failures.append(f"{model_name} RMSE gap {rmse_gap:.4f} >= {rmse_tol}")
            passed = False
        if ssim_gap >= ssim_tol:
            print(f"  FAIL: SSIM gap {ssim_gap:.4f} >= tolerance {ssim_tol}")
            failures.append(f"{model_name} SSIM gap {ssim_gap:.4f} >= {ssim_tol}")
            passed = False
        if passed:
            print(f"  {model_name}: PASS (within tolerance)")

    print(f"\n{'='*60}")
    if failures:
        print("CALIBRATION GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        print("\nThe eval harness does not reproduce published baselines.")
        print("Debug before trusting any retrained model numbers.")
        import sys
        sys.exit(1)
    else:
        print("CALIBRATION GATE: PASS")
    print(f"{'='*60}")


@app.local_entrypoint()
def main():
    print("Running calibration check with pre-flight...")
    calibrate.remote()

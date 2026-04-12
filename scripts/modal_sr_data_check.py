#!/usr/bin/env python3
"""DIAGNOSTIC-ONLY: check SIM vs widefield normalization on the Modal volume.

Run once during the SR calibration investigation (Decision 19) to
verify that SIM .npy files use the same Z-score normalization as
widefield .npy files. If they used different normalization constants,
denormalizing SIM with the widefield mean/std would silently place the
ground truth in the wrong intensity space — a classic data-provenance
bug. This script prints raw and denormalized statistics for avg1,
avg16, avg400, and sim data at one FoV so the comparison is auditable.

Outcome: SIM and widefield use the same Z-score constants; pipeline
denormalization with `* 66.03 + 154.54` is correct for both.

Usage:
    modal run scripts/modal_sr_data_check.py
"""

from __future__ import annotations

import modal

app = modal.App("inverseops-sr-data-check")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.24")


@app.function(image=image, volumes={"/data": data_vol}, timeout=120)
def check_sr_data():
    """Compare SIM and widefield .npy normalization."""
    import numpy as np
    from pathlib import Path

    data_root = Path("/data/w2s/data/normalized")
    W2S_MEAN = 154.54
    W2S_STD = 66.03

    # Pick a test FoV
    fov_id = 1
    wl = 0

    print("=" * 60)
    print("SR DATA NORMALIZATION CHECK")
    print("=" * 60)

    # Check all data levels including SIM
    levels = ["avg1", "avg16", "avg400", "sim"]
    for level in levels:
        path = data_root / level / f"{fov_id:03d}_{wl}.npy"
        if not path.exists():
            print(f"\n{level}: NOT FOUND at {path}")
            continue

        arr = np.load(path).astype(np.float32)
        denorm = arr * W2S_STD + W2S_MEAN

        print(f"\n--- {level} (shape={arr.shape}) ---")
        print(f"  Raw .npy:     min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}")
        print(f"  Denormalized: min={denorm.min():.2f}  max={denorm.max():.2f}  "
              f"mean={denorm.mean():.2f}  std={denorm.std():.2f}")
        print(f"  % outside [0,255] after denorm: "
              f"{100.0 * np.sum((denorm < 0) | (denorm > 255)) / denorm.size:.2f}%")

    # Check shape compatibility for 2x SR
    lr_path = data_root / "avg400" / f"{fov_id:03d}_{wl}.npy"
    hr_path = data_root / "sim" / f"{fov_id:03d}_{wl}.npy"
    if lr_path.exists() and hr_path.exists():
        lr = np.load(lr_path)
        hr = np.load(hr_path)
        print(f"\n--- Shape check ---")
        print(f"  LR (avg400): {lr.shape}")
        print(f"  HR (sim):    {hr.shape}")
        print(f"  Scale ratio: {hr.shape[0] / lr.shape[0]:.1f}x x {hr.shape[1] / lr.shape[1]:.1f}x")

    # Also check if SIM normalization constants differ
    # by computing the mean/std across a few SIM images
    print(f"\n--- SIM normalization constants (sample of 10 images) ---")
    sim_dir = data_root / "sim"
    sim_files = sorted(sim_dir.glob("*.npy"))[:10]
    all_means = []
    all_stds = []
    for sf in sim_files:
        s = np.load(sf).astype(np.float32)
        # If denormalized with widefield constants, check what the raw mean would be
        # in intensity space
        s_denorm = s * W2S_STD + W2S_MEAN
        all_means.append(s_denorm.mean())
        all_stds.append(s_denorm.std())

    print(f"  Denorm'd mean across 10 SIM images: {np.mean(all_means):.2f} "
          f"(widefield ref: {W2S_MEAN:.2f})")
    print(f"  Denorm'd std across 10 SIM images:  {np.mean(all_stds):.2f} "
          f"(widefield ref: {W2S_STD:.2f})")

    # If SIM was normalized with DIFFERENT constants, the denormalized mean
    # would NOT be ~154.54. Flag this.
    sim_denorm_mean = np.mean(all_means)
    if abs(sim_denorm_mean - W2S_MEAN) > 30:
        print(f"\n  *** WARNING: SIM denormalized mean ({sim_denorm_mean:.2f}) is far "
              f"from widefield mean ({W2S_MEAN:.2f})")
        print(f"  *** SIM likely uses DIFFERENT normalization constants!")
        print(f"  *** Check W2S supplementary Section 3 for SIM-specific constants.")
    else:
        print(f"\n  SIM denormalized mean is close to widefield mean — "
              f"likely same normalization.")


@app.local_entrypoint()
def main():
    check_sr_data.remote()

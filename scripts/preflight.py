#!/usr/bin/env python3
"""Training readiness gate — run before every Modal GPU launch.

Verifies data properties, normalization, metrics, and (optionally)
a reference baseline reproduction. ~30 minutes total. No GPU required.

Usage:
    python scripts/preflight.py --data-root data/test_fixtures/w2s

    # With reference baseline (requires Modal volume access):
    python scripts/preflight.py --data-root /data/w2s/data/normalized \
        --reference-model /data/w2s/net_data/trained_denoisers/M_1/epoch_49.pth

Rule: No `modal run` for GPU training until this script passes
and you've written down what you observed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def check_data_inspection(data_root: str, splits_path: str) -> bool:
    """Check 1: Print actual statistics of one sample."""
    print("\n" + "=" * 60)
    print("CHECK 1: Data inspection")
    print("=" * 60)

    from inverseops.data.w2s import W2SDataset

    ds = W2SDataset(root_dir=data_root, split="train", patch_size=0,
                    splits_path=splits_path)
    ds.prepare()

    if len(ds) == 0:
        print("FAIL: Dataset is empty")
        return False

    sample = ds[0]
    inp = sample["input"]
    tgt = sample["target"]

    print(f"  Dataset size: {len(ds)} samples")
    print(f"  input  shape={inp.shape}  dtype={inp.dtype}")
    print(f"  input  min={inp.min():.4f}  max={inp.max():.4f}")
    print(f"  input  mean={inp.mean():.4f}  std={inp.std():.4f}")
    print(f"  target shape={tgt.shape}  dtype={tgt.dtype}")
    print(f"  target min={tgt.min():.4f}  max={tgt.max():.4f}")
    print(f"  target mean={tgt.mean():.4f}  std={tgt.std():.4f}")
    print(f"  noise_level={sample['noise_level']}  fov_id={sample['fov_id']}  wl={sample['wavelength']}")

    # Sanity: values should be in roughly [-3, 20] for pre-normalized W2S
    if inp.max() > 100 or tgt.max() > 100:
        print("  WARNING: Values > 100 — data may not be pre-normalized")
        return False
    if abs(inp.mean()) > 5:
        print("  WARNING: Mean far from 0 — check normalization state")
        return False

    print("  PASS")
    return True


def check_roundtrip(data_root: str, splits_path: str) -> bool:
    """Check 2: Verify denormalize is the correct inverse."""
    print("\n" + "=" * 60)
    print("CHECK 2: Denormalize roundtrip")
    print("=" * 60)

    from inverseops.data.w2s import W2SDataset, W2S_MEAN, W2S_STD

    ds = W2SDataset(root_dir=data_root, split="train", patch_size=0,
                    splits_path=splits_path)
    ds.prepare()

    sample = ds[0]
    normalized = sample["target"]
    denormalized = ds.denormalize(normalized)
    renormalized = (denormalized - W2S_MEAN) / W2S_STD

    max_error = (normalized - renormalized).abs().max().item()
    print(f"  normalize(denormalize(x)) max error: {max_error:.6f}")
    print(f"  Denormalized range: [{denormalized.min():.2f}, {denormalized.max():.2f}]")
    print(f"  Denormalized mean: {denormalized.mean():.2f} (expect ~{W2S_MEAN})")

    if max_error > 1e-4:
        print(f"  FAIL: roundtrip error {max_error} > 1e-4")
        return False

    if denormalized.mean() < 50 or denormalized.mean() > 300:
        print(f"  FAIL: denormalized mean {denormalized.mean():.2f} outside [50, 300]")
        return False

    print("  PASS")
    return True


def check_metric_sanity() -> bool:
    """Check 3: PSNR on three known cases."""
    print("\n" + "=" * 60)
    print("CHECK 3: Metric sanity")
    print("=" * 60)

    sys.path.insert(0, ".")
    from scripts.run_evaluation import psnr_tensor

    img = torch.randn(1, 64, 64) * 50 + 128  # Simulates denormalized image

    # Case 1: identical → should be inf
    p_identical = psnr_tensor(img, img, data_range=255.0)
    print(f"  PSNR(img, img)               = {p_identical}    (expect inf)")
    if p_identical != float("inf"):
        print("  FAIL: identical images should give inf PSNR")
        return False

    # Case 2: vs zeros → should be low
    p_zeros = psnr_tensor(img, torch.zeros_like(img), data_range=255.0)
    print(f"  PSNR(img, zeros)             = {p_zeros:.2f} dB   (expect <15 dB)")
    if p_zeros > 20:
        print(f"  FAIL: PSNR vs zeros too high: {p_zeros:.2f}")
        return False

    # Case 3: small noise → should be 30-50 dB
    noisy = img + torch.randn_like(img) * 2
    p_noisy = psnr_tensor(img, noisy, data_range=255.0)
    print(f"  PSNR(img, img+small_noise)   = {p_noisy:.2f} dB   (expect 30-50 dB)")
    if not (25 < p_noisy < 55):
        print(f"  FAIL: PSNR with small noise outside [25, 55]: {p_noisy:.2f}")
        return False

    print("  PASS")
    return True


def check_trainer_psnr_sanity(data_root: str, splits_path: str) -> bool:
    """Check 5: Tiny smoke train — 1 epoch, verify PSNR is plausible."""
    print("\n" + "=" * 60)
    print("CHECK 5: Smoke train (1 epoch CPU)")
    print("=" * 60)

    from torch.utils.data import DataLoader

    from inverseops.data.w2s import W2SDataset
    from inverseops.training.losses import l1_loss
    from inverseops.training.trainer import Trainer

    ds = W2SDataset(root_dir=data_root, split="train", patch_size=32,
                    avg_levels=[1], splits_path=splits_path)
    ds.prepare()

    if len(ds) == 0:
        print("  SKIP: no training samples")
        return True

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = torch.nn.Conv2d(1, 1, 3, padding=1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=Path(tmp),
            max_epochs=1,
            use_amp=False,
            wandb_enabled=False,
            denormalize_fn=W2SDataset.denormalize,
            early_stopping_patience=100,
        )
        summary = trainer.train()

    psnr = summary["best_val_psnr"]
    print(f"  Val PSNR after 1 epoch: {psnr:.2f} dB")

    if psnr > 60:
        print(f"  FAIL: PSNR {psnr:.2f} suspiciously high — denormalization bug?")
        return False
    if psnr < 0:
        print(f"  FAIL: PSNR {psnr:.2f} negative — metric is broken")
        return False

    print("  PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Training readiness gate — run before every Modal GPU launch."
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to W2S normalized data root",
    )
    parser.add_argument(
        "--splits-path", type=str, default="inverseops/data/splits.json",
        help="Path to splits.json",
    )
    parser.add_argument(
        "--skip-smoke-train", action="store_true",
        help="Skip the smoke train check",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PREFLIGHT: Training Readiness Gate")
    print("=" * 60)

    results = {}

    results["data_inspection"] = check_data_inspection(args.data_root, args.splits_path)
    results["roundtrip"] = check_roundtrip(args.data_root, args.splits_path)
    results["metric_sanity"] = check_metric_sanity()

    if not args.skip_smoke_train:
        results["smoke_train"] = check_trainer_psnr_sanity(args.data_root, args.splits_path)

    # Summary
    print("\n" + "=" * 60)
    print("PREFLIGHT SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll checks passed. Safe to launch training.")
        return 0
    else:
        print("\nFAILED. Fix issues before launching training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""SR evaluation: run our trained SwinIR SR (avg400 → SIM) through the V3 harness.

Fork of scripts/modal_sr_calibration.py. Differences:
  - Loads our fine-tuned SwinIR checkpoint, not the W2S RRDBNet.
  - LR input is avg400 (clean LR) — matches the training config.
  - Reports the matched bicubic(avg400 → SIM) baseline at n=39 (the number
    quoted in Decision 19, 0.1754, was bicubic(avg1 → SIM) — it does NOT
    match this training setup and must not be used as the comparison floor).
  - Runs on BOTH val and test splits so the trainer's 25.75 dB val PSNR
    can be reconciled against full-image eval PSNR in the same space.
  - No cross-paper calibration gate — this is our own model on our own split.

Pipeline:
  1. Rebuild SwinIR from checkpoint config (pretrained=False) + load
     model_state_dict from best.pt. No re-download of DIV2K weights.
  2. Sliding-window SR inference via inverseops.evaluation.stitching
     (128x128 LR patches, stride 64). Z-score input, Z-score output —
     same convention as training. clamp=False so all RMSE conventions
     are reachable.
  3. Report, per split (val, test):
       - SSIM (clipped [0,1])
       - PSNR (unclipped [0,1], dr=1)      ← primary SR number
       - PSNR (clipped [0,1], dr=1)        ← reconciles with trainer 25.75 dB
       - RMSE clipped [0,1]
       - RMSE unclipped [0,1]
       - Bicubic(avg400 → SIM) RMSE + PSNR (matched baseline)
       - Per-FoV mean list (not just aggregate) to spot cherry-picks
     Aggregation: per-FoV mean over wavelengths, then mean +/- std across FoVs.

Usage:
    modal run scripts/modal_sr_eval_swinir.py
    modal run scripts/modal_sr_eval_swinir.py --checkpoint-path outputs/other/best.pt
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("inverseops-sr-eval-swinir")
vol = modal.Volume.from_name("inverseops-vol", create_if_missing=True)
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

# Image: same deps as training image + skimage for SSIM/bicubic
base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "timm>=0.9.0",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "scikit-image>=0.20",
)


def _source_ignore(path: Path) -> bool:
    skip = {
        "data", ".git", "__pycache__", "outputs",
        "artifacts", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    }
    top = path.parts[0] if path.parts else ""
    return top in skip


image = base_image.add_local_dir(".", remote_path="/app", ignore=_source_ignore)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/data": data_vol},
    timeout=3600,
)
def eval_sr(
    checkpoint_path: str = (
        "/vol/outputs/training_w2s_sr_swinir_2x/checkpoints/best.pt"
    ),
):
    """Evaluate SwinIR SR on val + test splits."""
    import json
    import sys
    from collections import defaultdict

    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim_skimage
    from skimage.transform import resize as skimage_resize

    sys.path.insert(0, "/app")

    from inverseops.evaluation.stitching import sliding_window_sr
    from inverseops.models import build_model

    W2S_MEAN = 154.54
    W2S_STD = 66.03

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load splits
    # ------------------------------------------------------------------
    with open("/app/inverseops/data/splits.json") as f:
        splits = json.load(f)
    val_fovs = splits["w2s"]["val"]
    test_fovs = splits["w2s"]["test"]
    wavelengths = [0, 1, 2]
    data_root = Path("/data/w2s/data/normalized")

    print(f"Val FoVs:  {val_fovs}  (n={len(val_fovs)})")
    print(f"Test FoVs: {test_fovs}  (n={len(test_fovs)})")

    # ------------------------------------------------------------------
    # Load checkpoint and rebuild model
    # ------------------------------------------------------------------
    print(f"\n=== Loading checkpoint: {checkpoint_path} ===")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Flip pretrained off — we load OUR fine-tuned weights below.
    config.setdefault("model", {})["pretrained"] = False
    print(f"  best_val_psnr (trainer):  {ckpt.get('best_val_psnr', '?')}")
    print(f"  best_epoch:               {ckpt.get('best_epoch', '?')}")
    print(f"  global_step:              {ckpt.get('global_step', '?')}")

    model = build_model(config, device=device)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if missing:
        print(f"  WARNING: missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print("  state_dict loaded cleanly (strict match)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e6:.1f}M)")
    model.eval()

    # ------------------------------------------------------------------
    # Shape smoke test — SwinIR window_size=8 requires dims divisible by 8.
    # The sliding-window stitcher feeds 128x128 (interior) and 64x128 /
    # 128x64 / 64x64 (border). All divisible by 8.
    # ------------------------------------------------------------------
    print("\n=== Model shape smoke test ===")
    for h, w in [(128, 128), (64, 128), (128, 64), (64, 64)]:
        test_in = torch.randn(1, 1, h, w).to(device)
        with torch.no_grad():
            test_o = model(test_in)
        if isinstance(test_o, list):
            test_o = test_o[-1]
        expected = (1, 1, h * 2, w * 2)
        if tuple(test_o.shape) != expected:
            raise ValueError(
                f"Shape smoke test failed for {(h,w)}: "
                f"got {tuple(test_o.shape)}, expected {expected}"
            )
        print(f"  {(h, w)} -> {tuple(test_o.shape)}  PASS")

    # ------------------------------------------------------------------
    # Helper metrics
    # ------------------------------------------------------------------
    def rmse_np(gt, pred):
        return float(np.sqrt(np.mean(
            (gt.astype(np.float64) - pred.astype(np.float64)) ** 2
        )))

    def psnr_np(gt, pred, data_range=1.0):
        mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(data_range ** 2 / mse))

    # ------------------------------------------------------------------
    # Evaluation loop — run on val and test separately
    # ------------------------------------------------------------------
    def run_split(fovs, split_name):
        print(f"\n{'='*70}")
        print(f"EVAL: SwinIR SR (avg400 -> SIM) on {split_name} split "
              f"({len(fovs)} FoVs x {len(wavelengths)} wavelengths)")
        print(f"{'='*70}")

        rmse_clip_by_fov = defaultdict(list)
        rmse_noclip_by_fov = defaultdict(list)
        psnr_clip_by_fov = defaultdict(list)
        psnr_noclip_by_fov = defaultdict(list)
        ssim_by_fov = defaultdict(list)

        bic_rmse_clip_by_fov = defaultdict(list)
        bic_rmse_noclip_by_fov = defaultdict(list)
        bic_psnr_clip_by_fov = defaultdict(list)
        bic_psnr_noclip_by_fov = defaultdict(list)
        bic_ssim_by_fov = defaultdict(list)

        pct_hr_outside_by_fov = defaultdict(list)

        n_processed = 0
        for fov_id in fovs:
            for wl in wavelengths:
                lr_path = data_root / "avg400" / f"{fov_id:03d}_{wl}.npy"
                hr_path = data_root / "sim" / f"{fov_id:03d}_{wl}.npy"

                if not lr_path.exists() or not hr_path.exists():
                    print(f"  SKIP: FoV {fov_id} wl {wl} — missing file")
                    continue

                lr_z = np.load(lr_path).astype(np.float32)
                hr_z = np.load(hr_path).astype(np.float32)

                # HR target in two conventions
                hr_01_clip = np.clip(hr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0
                hr_01_noclip = (hr_z * W2S_STD + W2S_MEAN) / 255.0

                # LR in [0,1] for bicubic baseline
                lr_01_clip = np.clip(lr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0
                lr_01_noclip = (lr_z * W2S_STD + W2S_MEAN) / 255.0

                # SwinIR inference — Z-score in, Z-score out, no clamp
                sr_z = sliding_window_sr(model, lr_z, device, clamp=False)

                sr_01_clip = np.clip(sr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0
                sr_01_noclip = (sr_z * W2S_STD + W2S_MEAN) / 255.0

                # Model metrics
                rmse_clip = rmse_np(hr_01_clip, sr_01_clip)
                rmse_noclip = rmse_np(hr_01_noclip, sr_01_noclip)
                psnr_clip = psnr_np(hr_01_clip, sr_01_clip, data_range=1.0)
                psnr_noclip = psnr_np(hr_01_noclip, sr_01_noclip, data_range=1.0)
                ssim_val = float(ssim_skimage(
                    hr_01_clip, sr_01_clip, data_range=1.0
                ))

                # Bicubic(avg400 -> SIM) — the MATCHED baseline for this run
                lr_up_clip = skimage_resize(
                    lr_01_clip, hr_01_clip.shape, order=3, anti_aliasing=False
                )
                lr_up_noclip = skimage_resize(
                    lr_01_noclip, hr_01_noclip.shape, order=3, anti_aliasing=False
                )
                bic_rmse_clip = rmse_np(hr_01_clip, lr_up_clip)
                bic_rmse_noclip = rmse_np(hr_01_noclip, lr_up_noclip)
                bic_psnr_clip = psnr_np(hr_01_clip, lr_up_clip, data_range=1.0)
                bic_psnr_noclip = psnr_np(
                    hr_01_noclip, lr_up_noclip, data_range=1.0
                )
                bic_ssim = float(ssim_skimage(
                    hr_01_clip, lr_up_clip, data_range=1.0
                ))

                pct_out = 100.0 * float(np.mean(
                    (hr_01_noclip < 0) | (hr_01_noclip > 1)
                ))

                rmse_clip_by_fov[fov_id].append(rmse_clip)
                rmse_noclip_by_fov[fov_id].append(rmse_noclip)
                psnr_clip_by_fov[fov_id].append(psnr_clip)
                psnr_noclip_by_fov[fov_id].append(psnr_noclip)
                ssim_by_fov[fov_id].append(ssim_val)

                bic_rmse_clip_by_fov[fov_id].append(bic_rmse_clip)
                bic_rmse_noclip_by_fov[fov_id].append(bic_rmse_noclip)
                bic_psnr_clip_by_fov[fov_id].append(bic_psnr_clip)
                bic_psnr_noclip_by_fov[fov_id].append(bic_psnr_noclip)
                bic_ssim_by_fov[fov_id].append(bic_ssim)

                pct_hr_outside_by_fov[fov_id].append(pct_out)
                n_processed += 1

            if fov_id in rmse_clip_by_fov:
                fov_rmse_c = float(np.mean(rmse_clip_by_fov[fov_id]))
                fov_psnr_c = float(np.mean(psnr_clip_by_fov[fov_id]))
                fov_ssim = float(np.mean(ssim_by_fov[fov_id]))
                fov_bic_psnr_c = float(np.mean(bic_psnr_clip_by_fov[fov_id]))
                print(
                    f"  FoV {fov_id:3d}: "
                    f"model PSNR_clip={fov_psnr_c:.2f}  RMSE_clip={fov_rmse_c:.4f}  "
                    f"SSIM={fov_ssim:.4f}  |  "
                    f"bicubic PSNR_clip={fov_bic_psnr_c:.2f}"
                )

        # Aggregate: per-FoV mean, then mean +/- std over FoVs
        def agg(d):
            per_fov = [float(np.mean(v)) for v in d.values()]
            return per_fov, float(np.mean(per_fov)), float(np.std(per_fov))

        fov_rmse_clip, mean_rmse_clip, std_rmse_clip = agg(rmse_clip_by_fov)
        fov_rmse_noclip, mean_rmse_noclip, std_rmse_noclip = agg(rmse_noclip_by_fov)
        fov_psnr_clip, mean_psnr_clip, std_psnr_clip = agg(psnr_clip_by_fov)
        fov_psnr_noclip, mean_psnr_noclip, std_psnr_noclip = agg(psnr_noclip_by_fov)
        fov_ssim, mean_ssim, std_ssim = agg(ssim_by_fov)

        (fov_bic_rmse_clip,
         mean_bic_rmse_clip, std_bic_rmse_clip) = agg(bic_rmse_clip_by_fov)
        (fov_bic_rmse_noclip,
         mean_bic_rmse_noclip, std_bic_rmse_noclip) = agg(bic_rmse_noclip_by_fov)
        (fov_bic_psnr_clip,
         mean_bic_psnr_clip, std_bic_psnr_clip) = agg(bic_psnr_clip_by_fov)
        (fov_bic_psnr_noclip,
         mean_bic_psnr_noclip, std_bic_psnr_noclip) = agg(bic_psnr_noclip_by_fov)
        (fov_bic_ssim,
         mean_bic_ssim, std_bic_ssim) = agg(bic_ssim_by_fov)

        (_,
         mean_pct_out, std_pct_out) = agg(pct_hr_outside_by_fov)

        print(f"\n  Processed: {n_processed} images "
              f"({len(rmse_clip_by_fov)} FoVs x {len(wavelengths)} wavelengths)")
        print(f"\n  --- SwinIR SR ({split_name}) ---")
        print(f"    PSNR clipped [0,1], dr=1:   "
              f"{mean_psnr_clip:.2f} +/- {std_psnr_clip:.2f} dB")
        print(f"    PSNR unclipped [0,1], dr=1: "
              f"{mean_psnr_noclip:.2f} +/- {std_psnr_noclip:.2f} dB")
        print(f"    RMSE clipped [0,1]:         "
              f"{mean_rmse_clip:.4f} +/- {std_rmse_clip:.4f}")
        print(f"    RMSE unclipped [0,1]:       "
              f"{mean_rmse_noclip:.4f} +/- {std_rmse_noclip:.4f}")
        print(f"    SSIM:                        "
              f"{mean_ssim:.4f} +/- {std_ssim:.4f}")
        print(f"\n  --- Bicubic(avg400 -> SIM) MATCHED baseline ({split_name}) ---")
        print(f"    PSNR clipped [0,1], dr=1:   "
              f"{mean_bic_psnr_clip:.2f} +/- {std_bic_psnr_clip:.2f} dB")
        print(f"    PSNR unclipped [0,1], dr=1: "
              f"{mean_bic_psnr_noclip:.2f} +/- {std_bic_psnr_noclip:.2f} dB")
        print(f"    RMSE clipped [0,1]:         "
              f"{mean_bic_rmse_clip:.4f} +/- {std_bic_rmse_clip:.4f}")
        print(f"    RMSE unclipped [0,1]:       "
              f"{mean_bic_rmse_noclip:.4f} +/- {std_bic_rmse_noclip:.4f}")
        print(f"    SSIM:                        "
              f"{mean_bic_ssim:.4f} +/- {std_bic_ssim:.4f}")
        print("\n  --- Deltas (SwinIR - bicubic) ---")
        print(f"    delta PSNR clipped:  "
              f"{mean_psnr_clip - mean_bic_psnr_clip:+.2f} dB")
        print(f"    delta PSNR unclipped:"
              f" {mean_psnr_noclip - mean_bic_psnr_noclip:+.2f} dB")
        print(f"    delta RMSE clipped:  "
              f"{mean_rmse_clip - mean_bic_rmse_clip:+.4f}")
        print(f"    delta SSIM:          "
              f"{mean_ssim - mean_bic_ssim:+.4f}")
        print(f"\n  HR saturation: {mean_pct_out:.2f}% +/- "
              f"{std_pct_out:.2f}% of pixels exceed [0,255] after denorm")

        return {
            "split": split_name,
            "n_fovs": len(rmse_clip_by_fov),
            "n_samples": n_processed,
            "fov_ids": sorted(rmse_clip_by_fov.keys()),
            "model": {
                "psnr_clip": {
                    "mean": mean_psnr_clip, "std": std_psnr_clip,
                    "per_fov": fov_psnr_clip,
                },
                "psnr_noclip": {
                    "mean": mean_psnr_noclip, "std": std_psnr_noclip,
                    "per_fov": fov_psnr_noclip,
                },
                "rmse_clip": {
                    "mean": mean_rmse_clip, "std": std_rmse_clip,
                    "per_fov": fov_rmse_clip,
                },
                "rmse_noclip": {
                    "mean": mean_rmse_noclip, "std": std_rmse_noclip,
                    "per_fov": fov_rmse_noclip,
                },
                "ssim": {
                    "mean": mean_ssim, "std": std_ssim, "per_fov": fov_ssim,
                },
            },
            "bicubic_avg400": {
                "psnr_clip": {
                    "mean": mean_bic_psnr_clip, "std": std_bic_psnr_clip,
                    "per_fov": fov_bic_psnr_clip,
                },
                "psnr_noclip": {
                    "mean": mean_bic_psnr_noclip, "std": std_bic_psnr_noclip,
                    "per_fov": fov_bic_psnr_noclip,
                },
                "rmse_clip": {
                    "mean": mean_bic_rmse_clip, "std": std_bic_rmse_clip,
                    "per_fov": fov_bic_rmse_clip,
                },
                "rmse_noclip": {
                    "mean": mean_bic_rmse_noclip, "std": std_bic_rmse_noclip,
                    "per_fov": fov_bic_rmse_noclip,
                },
                "ssim": {
                    "mean": mean_bic_ssim, "std": std_bic_ssim,
                    "per_fov": fov_bic_ssim,
                },
            },
            "hr_pct_outside_01": {
                "mean": mean_pct_out, "std": std_pct_out,
            },
        }

    val_result = run_split(val_fovs, "val")
    test_result = run_split(test_fovs, "test")

    # ------------------------------------------------------------------
    # Reconciliation against trainer val PSNR = 25.75 dB
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("RECONCILIATION")
    print(f"{'='*70}")
    trainer_val_psnr = float(ckpt.get("best_val_psnr", 0.0))
    full_image_val_psnr_clip = val_result["model"]["psnr_clip"]["mean"]
    gap = full_image_val_psnr_clip - trainer_val_psnr
    print(f"  Trainer val PSNR (patch-level, clipped [0,255]): "
          f"{trainer_val_psnr:.2f} dB")
    print(f"  Eval full-image val PSNR (clipped [0,1]):        "
          f"{full_image_val_psnr_clip:.2f} dB")
    print(f"  Gap (full-image - patch):                        {gap:+.2f} dB")
    print("  Note: full-image PSNR averages more pixels and tends to be")
    print("  higher than patch-level center-crop PSNR when the model works")
    print("  and lower when the model overfits to the center crop.")

    # ------------------------------------------------------------------
    # Persist result JSON to /vol for reproducibility
    # ------------------------------------------------------------------
    output_dir = Path("/vol/outputs/training_w2s_sr_swinir_2x")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sr_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "trainer_val_psnr": trainer_val_psnr,
                "val": val_result,
                "test": test_result,
                "reconciliation": {
                    "trainer_val_psnr_dr255": trainer_val_psnr,
                    "full_image_val_psnr_clip_dr1": full_image_val_psnr_clip,
                    "gap_dB": gap,
                },
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to: {out_path}")
    vol.commit()


@app.local_entrypoint()
def main(
    checkpoint_path: str = (
        "/vol/outputs/training_w2s_sr_swinir_2x/checkpoints/best.pt"
    ),
):
    print("Running SwinIR SR eval...")
    eval_sr.remote(checkpoint_path=checkpoint_path)

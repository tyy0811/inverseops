#!/usr/bin/env python3
"""Dataset-agnostic evaluation harness for denoising models.

Loads frozen splits from splits.json, runs inference on the test set,
and reports PSNR/SSIM as mean +/- std per condition (noise level or sigma).

Supports:
- W2S fluorescence microscopy (group by noise level, aggregate per FoV)
- IXI brain MRI (group by sigma, aggregate per subject)

Key design decisions:
- Denormalizes predictions and targets BEFORE computing metrics,
  then clamps to [0, data_range] for consistent comparison.
- Each dataset class owns its own normalization via denormalize().
- Aggregates per-unit first (FoV or subject), then mean +/- std across units.
- --calibration mode runs W2S pretrained baselines for harness validation (W2S only).

Usage:
    # Evaluate a retrained checkpoint (W2S)
    python scripts/run_evaluation.py \
        --data-root /data/w2s/data/normalized \
        --checkpoint outputs/training_w2s_swinir/best.pt \
        --model swinir --dataset w2s \
        --output-csv artifacts/v3/swinir_denoise_results.csv

    # Evaluate a retrained checkpoint (IXI)
    python scripts/run_evaluation.py \
        --data-root /data/ixi/T1 \
        --checkpoint outputs/training_ixi_swinir/best.pt \
        --model swinir --dataset ixi \
        --output-csv artifacts/v3/ixi_swinir_results.csv

    # Calibration check against W2S pretrained baselines (W2S only)
    python scripts/run_evaluation.py \
        --data-root /data/w2s/data/normalized \
        --calibration --dataset w2s \
        --calibration-dir /data/w2s/net_data/trained_denoisers/ \
        --output-csv artifacts/v3/calibration_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Tensor-based metrics (float data, fixed data range)
# ---------------------------------------------------------------------------
# Import the canonical data range mapping from the data registry.
# Single source of truth — also used by the Trainer for validation PSNR.
from inverseops.data import DATASET_DATA_RANGE


def psnr_tensor(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    data_range: float = 255.0,
) -> float:
    """Compute PSNR between two tensors.

    Args:
        reference: Ground truth tensor.
        prediction: Predicted tensor (same shape).
        data_range: Fixed peak signal value (must be constant across samples).

    Returns:
        PSNR in dB.
    """
    mse = torch.mean((reference.float() - prediction.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def ssim_tensor(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    data_range: float = 255.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute SSIM between two tensors.

    Args:
        reference: Ground truth tensor [1, H, W] or [H, W].
        prediction: Predicted tensor (same shape).
        data_range: Fixed dynamic range (must be constant across samples).

    Returns:
        SSIM in [0, 1].
    """

    ref = reference.squeeze().float().numpy()
    pred = prediction.squeeze().float().numpy()

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Gaussian window
    sigma = window_size / 6.0
    coords = np.arange(window_size) - (window_size - 1) / 2.0
    gauss_1d = np.exp(-(coords**2) / (2 * sigma**2))
    kernel = np.outer(gauss_1d, gauss_1d)
    kernel = kernel / kernel.sum()

    from numpy.lib.stride_tricks import sliding_window_view

    def _filter(img: np.ndarray) -> np.ndarray:
        windows = sliding_window_view(img, (window_size, window_size))
        return np.einsum("ijkl,kl->ij", windows, kernel)

    mu1 = _filter(ref)
    mu2 = _filter(pred)
    sigma1_sq = _filter(ref**2) - mu1**2
    sigma2_sq = _filter(pred**2) - mu2**2
    sigma12 = _filter(ref * pred) - mu1 * mu2

    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


# Per-dataset sample key mapping.
# group_key: the condition axis (noise level, sigma) — rows in the results table
# unit_key: the independent unit for aggregation (FoV, subject) — mean +/- std
_DATASET_SAMPLE_KEYS: dict[str, dict[str, str]] = {
    "w2s": {"group_key": "noise_level", "unit_key": "fov_id"},
    "ixi": {"group_key": "sigma", "unit_key": "subject_id"},
}


def _get_sample_keys(dataset_name: str) -> tuple[str, str]:
    """Return (group_key, unit_key) for a dataset."""
    keys = _DATASET_SAMPLE_KEYS.get(dataset_name, {})
    return keys.get("group_key", "noise_level"), keys.get("unit_key", "fov_id")


def evaluate_checkpoint(
    data_root: str,
    checkpoint_path: str,
    model_name: str,
    dataset_name: str = "w2s",
    splits_path: str = "inverseops/data/splits.json",
    output_csv: str | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluate a single checkpoint on the test split.

    Returns dict of {group: {"psnr_mean", "psnr_std",
    "ssim_mean", "ssim_std", "n_units"}}.
    Group is noise_level for W2S, sigma for IXI.
    """
    from inverseops.data import build_dataset
    from inverseops.models import build_model

    data_range = DATASET_DATA_RANGE.get(dataset_name, 255.0)
    group_key, unit_key = _get_sample_keys(dataset_name)

    # Build test dataset — full images, no cropping
    config = {
        "data": {
            "dataset": dataset_name,
            "train_root": data_root,
            "splits_path": splits_path,
            "patch_size": 0,  # Full image for eval
        },
        "model": {"name": model_name},
        "task": "denoise",
    }
    test_dataset = build_dataset(config, split="test", training=False)

    # Load model
    model_config = {
        "model": {"name": model_name, "pretrained": False},
        "task": "denoise",
    }
    model = build_model(model_config, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference and collect per-sample metrics
    # Structure: {group: {unit: [metric, ...], ...}}
    psnr_by_group_unit: dict = defaultdict(lambda: defaultdict(list))
    ssim_by_group_unit: dict = defaultdict(lambda: defaultdict(list))

    print(f"Evaluating {len(test_dataset)} samples (data_range={data_range})...")

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            inp = sample["input"].unsqueeze(0).to(device)  # [1, 1, H, W]
            target = sample["target"]  # [1, H, W]
            group = sample[group_key]
            unit = sample[unit_key]

            output = model(inp).squeeze(0).cpu()  # [1, H, W]

            # Denormalize BEFORE metrics, then clamp to valid range.
            # W2S denormalized data can exceed 255 (Decision 10: 13% of
            # pixels > 255). Clamping matches the uint8 PNG pipeline the
            # pretrained baselines were evaluated under.
            output_denorm = test_dataset.denormalize(output).clamp(0, data_range)
            target_denorm = test_dataset.denormalize(target).clamp(0, data_range)

            p = psnr_tensor(target_denorm, output_denorm, data_range=data_range)
            s = ssim_tensor(target_denorm, output_denorm, data_range=data_range)

            psnr_by_group_unit[group][unit].append(p)
            ssim_by_group_unit[group][unit].append(s)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(test_dataset)}")

    # Aggregate: per-unit mean, then mean +/- std across units
    results = _aggregate_results(psnr_by_group_unit, ssim_by_group_unit)
    _print_results(results, model_name, group_key=group_key)

    if output_csv:
        _write_csv(results, model_name, output_csv)

    return results


def _aggregate_results(
    psnr_by_group_unit: dict,
    ssim_by_group_unit: dict,
) -> dict:
    """Aggregate per-unit mean, then mean +/- std across units per group.

    Works for any dataset: units are FoVs (W2S) or subjects (IXI),
    groups are noise levels (W2S) or sigma values (IXI).
    """
    results = {}
    for group in sorted(psnr_by_group_unit.keys()):
        # Per-unit mean (e.g., average across wavelengths within each FoV)
        unit_psnrs = [
            np.mean(psnr_list) for psnr_list in psnr_by_group_unit[group].values()
        ]
        unit_ssims = [
            np.mean(ssim_list) for ssim_list in ssim_by_group_unit[group].values()
        ]

        results[group] = {
            "psnr_mean": float(np.mean(unit_psnrs)),
            "psnr_std": float(np.std(unit_psnrs)),
            "ssim_mean": float(np.mean(unit_ssims)),
            "ssim_std": float(np.std(unit_ssims)),
            "n_units": len(unit_psnrs),
        }
    return results


def _print_results(
    results: dict, model_name: str, group_key: str = "noise_level"
) -> None:
    group_label = "Noise Level" if group_key == "noise_level" else "Condition"
    print(f"\n{'=' * 65}")
    print(f"Results: {model_name}")
    print(f"{'=' * 65}")
    print(f"{group_label:>12}  {'PSNR (dB)':>16}  {'SSIM':>16}  {'Units':>6}")
    print(f"{'-' * 65}")
    for group in sorted(results.keys()):
        r = results[group]
        psnr_str = f"{r['psnr_mean']:.2f} +/- {r['psnr_std']:.2f}"
        ssim_str = f"{r['ssim_mean']:.4f} +/- {r['ssim_std']:.4f}"
        if group_key == "noise_level":
            group_str = f"avg{group:>9}"
        elif isinstance(group, float):
            group_str = f"s={group:>9.2f}"
        else:
            group_str = f"{group:>12}"
        print(f"{group_str}  {psnr_str:>16}  {ssim_str:>16}  {r['n_units']:>6}")
    print(f"{'=' * 65}")


def _write_csv(results: dict, model_name: str, output_csv: str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "condition",
                "psnr_mean",
                "psnr_std",
                "ssim_mean",
                "ssim_std",
                "n_units",
            ]
        )
        for group in sorted(results.keys()):
            r = results[group]
            writer.writerow(
                [
                    model_name,
                    group,
                    f"{r['psnr_mean']:.4f}",
                    f"{r['psnr_std']:.4f}",
                    f"{r['ssim_mean']:.6f}",
                    f"{r['ssim_std']:.6f}",
                    r["n_units"],
                ]
            )
    print(f"Results written to {output_csv}")


def run_calibration(
    data_root: str,
    calibration_dir: str,
    dataset_name: str = "w2s",
    splits_path: str = "inverseops/data/splits.json",
    output_csv: str | None = None,
    device: str = "cpu",
) -> dict:
    """Run W2S pretrained baselines for calibration check.

    Loads DnCNN/MemNet/RIDNet pretrained checkpoints from the W2S repo's
    net_data/trained_denoisers/ directory and evaluates them on the test split.

    W2S-only: calibration baselines are specific to the W2S dataset.
    Use evaluate_checkpoint() for other datasets.
    """
    if dataset_name != "w2s":
        print(
            f"ERROR: Calibration is only supported for W2S "
            f"(got --dataset {dataset_name}). "
            f"W2S pretrained baselines (DnCNN, MemNet, "
            f"RIDNet) are dataset-specific. Use "
            f"--checkpoint to evaluate retrained models "
            f"on other datasets."
        )
        sys.exit(1)

    from inverseops.data import build_dataset

    cal_dir = Path(calibration_dir)
    if not cal_dir.exists():
        print(f"ERROR: Calibration directory not found: {cal_dir}")
        print("Run scripts/download_w2s.py first to get W2S pretrained models.")
        sys.exit(1)

    data_range = DATASET_DATA_RANGE.get(dataset_name, 255.0)

    # Build test dataset
    config = {
        "data": {
            "dataset": dataset_name,
            "train_root": data_root,
            "splits_path": splits_path,
            "patch_size": 0,
        },
    }
    test_dataset = build_dataset(config, split="test", training=False)

    # Pre-group sample indices by noise level to avoid redundant iteration
    samples_by_level: dict[int, list[int]] = defaultdict(list)
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        samples_by_level[sample["noise_level"]].append(i)

    # Map W2S noise levels to their pretrained denoiser subdirectories
    # D_1, D_2, D_4, D_8, D_16 for DnCNN at avg1, avg2, avg4, avg8, avg16
    model_prefixes = {"DnCNN": "D", "MemNet": "M", "RIDNet": "R"}
    w2s_noise_levels = [1, 2, 4, 8, 16]

    all_results: dict[str, dict] = {}
    all_csv_rows: list[list] = []

    for model_name, prefix in model_prefixes.items():
        print(f"\n--- Calibration: {model_name} ---")

        psnr_by_level_fov: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        ssim_by_level_fov: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for noise_level in w2s_noise_levels:
            model_dir = cal_dir / f"{prefix}_{noise_level}"
            if not model_dir.exists():
                print(f"  SKIP: {model_dir} not found")
                continue

            # Find checkpoint file
            pth_files = list(model_dir.glob("*.pth"))
            if not pth_files:
                print(f"  SKIP: No .pth files in {model_dir}")
                continue

            ckpt_path = pth_files[0]
            print(f"  Loading {model_name} for avg{noise_level}: {ckpt_path.name}")

            try:
                # W2S pretrained models need the W2S repo's model definitions.
                # Add W2S repo to sys.path if available on the Modal volume.
                w2s_repo = Path("/data/w2s")
                if w2s_repo.exists() and str(w2s_repo) not in sys.path:
                    sys.path.insert(0, str(w2s_repo))

                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

                # W2S checkpoints may be raw state_dicts or wrapped models.
                # Try to load as a full model first (torch.save(model)),
                # then fall back to state_dict.
                if isinstance(ckpt, torch.nn.Module):
                    model = ckpt.to(device)
                else:
                    print(
                        f"  WARNING: Cannot auto-load "
                        f"{model_name} architecture. "
                        f"Checkpoint is a state_dict "
                        f"-- need model definition "
                        f"from W2S repo."
                    )
                    continue

                model.eval()
            except Exception as e:
                print(f"  ERROR loading {model_name} avg{noise_level}: {e}")
                continue

            # Evaluate only on test samples at this noise level
            level_indices = samples_by_level.get(noise_level, [])
            with torch.no_grad():
                for i in level_indices:
                    sample = test_dataset[i]
                    inp = sample["input"].unsqueeze(0).to(device)
                    target = sample["target"]
                    fov_id = sample["fov_id"]

                    output = model(inp).squeeze(0).cpu()

                    output_denorm = test_dataset.denormalize(output).clamp(
                        0, data_range
                    )
                    target_denorm = test_dataset.denormalize(target).clamp(
                        0, data_range
                    )

                    p = psnr_tensor(target_denorm, output_denorm, data_range=data_range)
                    s = ssim_tensor(target_denorm, output_denorm, data_range=data_range)

                    psnr_by_level_fov[noise_level][fov_id].append(p)
                    ssim_by_level_fov[noise_level][fov_id].append(s)

        results = _aggregate_results(psnr_by_level_fov, ssim_by_level_fov)
        _print_results(results, f"W2S-{model_name}", group_key="noise_level")
        all_results[model_name] = results

        for noise_level, r in sorted(results.items()):
            all_csv_rows.append(
                [
                    f"W2S-{model_name}",
                    noise_level,
                    f"{r['psnr_mean']:.4f}",
                    f"{r['psnr_std']:.4f}",
                    f"{r['ssim_mean']:.6f}",
                    f"{r['ssim_std']:.6f}",
                    r["n_units"],
                ]
            )

    # Fail loudly if no calibration results were produced — a silent
    # empty run is worse than a crash because it looks like a pass.
    models_with_results = [m for m, r in all_results.items() if r]
    if not models_with_results:
        print(
            "\nERROR: Calibration produced no results. All model checkpoints "
            "were skipped (likely state_dict format without matching architecture "
            "definitions). Use scripts/modal_calibration.py for W2S pretrained "
            "models that require explicit architecture construction."
        )
        sys.exit(1)

    n_expected_levels = len(w2s_noise_levels)
    for model_name, results in all_results.items():
        if results and len(results) < n_expected_levels:
            print(
                f"WARNING: {model_name} calibration covers "
                f"{len(results)}/{n_expected_levels} "
                f"noise levels -- some checkpoints "
                f"may be missing."
            )

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "condition",
                    "psnr_mean",
                    "psnr_std",
                    "ssim_mean",
                    "ssim_std",
                    "n_units",
                ]
            )
            for row in all_csv_rows:
                writer.writerow(row)
        print(f"\nCalibration results written to {output_csv}")

    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate denoising models on test split "
            "(W2S, IXI, or other registered datasets)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to dataset root (e.g. /data/w2s/data/normalized or /data/ixi/T1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="swinir",
        help="Model name (swinir, nafnet)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="w2s",
        help="Dataset name for registry lookup",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default="inverseops/data/splits.json",
        help="Path to splits.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to write CSV results",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run W2S pretrained baselines for calibration check",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Path to W2S net_data/trained_denoisers/ directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda or cpu, auto-detected if omitted)",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.calibration:
        if not args.calibration_dir:
            print("ERROR: --calibration requires --calibration-dir")
            return 1
        run_calibration(
            data_root=args.data_root,
            calibration_dir=args.calibration_dir,
            dataset_name=args.dataset,
            splits_path=args.splits_path,
            output_csv=args.output_csv,
            device=device,
        )
    else:
        if not args.checkpoint:
            print("ERROR: --checkpoint required (or use --calibration)")
            return 1
        evaluate_checkpoint(
            data_root=args.data_root,
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            dataset_name=args.dataset,
            splits_path=args.splits_path,
            output_csv=args.output_csv,
            device=device,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

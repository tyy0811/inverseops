#!/usr/bin/env python3
"""Day 3: Baseline evaluation of pretrained SwinIR on microscopy and natural images.

This script evaluates the pretrained SwinIR model on:
1. Microscopy images with synthetic Gaussian noise
2. Natural images with the same synthetic noise

Results are logged to Weights & Biases and saved as a CSV for analysis.

Usage:
    python scripts/run_evaluation.py \
        --microscopy-root data/raw/fmd \
        --natural-root data/raw/natural \
        --output-csv artifacts/baseline/baseline_metrics.csv

The output CSV supports the Day 3 decision gate:
    - If microscopy PSNR is clearly worse than natural images, proceed with fine-tuning
    - If not, consider pivoting to a more challenging microscopy subset (e.g., EM)
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
from PIL import Image

from inverseops.data.degradations import add_gaussian_noise
from inverseops.data.microscopy import MicroscopyDataset
from inverseops.evaluation.metrics import compute_psnr, compute_ssim

# Lazy import to avoid torch dependency issues
SwinIRBaseline = None


def _get_swinir_class():
    """Lazily import SwinIRBaseline to avoid torch import at module load."""
    global SwinIRBaseline
    if SwinIRBaseline is None:
        from inverseops.models.swinir import SwinIRBaseline as _SwinIRBaseline
        SwinIRBaseline = _SwinIRBaseline
    return SwinIRBaseline

# Supported noise levels for SwinIR denoising
SUPPORTED_SIGMAS: tuple[int, ...] = (15, 25, 50)


def compute_evidence_tier(microscopy_image_count: int, natural_image_count: int) -> str:
    """Compute evidence tier based on minimum sample count across domains.

    Args:
        microscopy_image_count: Number of unique microscopy images evaluated.
        natural_image_count: Number of unique natural images evaluated.

    Returns:
        "pilot" if min count <= 2, "moderate" if 3-9, "strong" if >= 10.
    """
    min_count = min(microscopy_image_count, natural_image_count)
    if min_count <= 2:
        return "pilot"
    elif min_count <= 9:
        return "moderate"
    else:
        return "strong"


def generate_smoke_fixtures(
    output_dir: Path, count: int = 5, seed: int = 42
) -> list[Path]:
    """Generate deterministic synthetic grayscale images for smoke testing.

    Creates simple patterns: gradient, checkerboard, circles, noise.

    Args:
        output_dir: Directory to save images.
        count: Number of images to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of paths to generated images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    size = 128
    paths = []

    patterns = ["gradient", "checkerboard", "circles", "noise", "stripes"]

    for i in range(count):
        pattern = patterns[i % len(patterns)]

        if pattern == "gradient":
            arr = np.tile(np.linspace(0, 255, size), (size, 1)).astype(np.uint8)
        elif pattern == "checkerboard":
            block = 16
            arr = np.zeros((size, size), dtype=np.uint8)
            for y in range(0, size, block):
                for x in range(0, size, block):
                    if ((y // block) + (x // block)) % 2 == 0:
                        arr[y : y + block, x : x + block] = 255
        elif pattern == "circles":
            y, x = np.ogrid[:size, :size]
            center = size // 2
            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            arr = ((np.sin(dist / 5) + 1) * 127.5).astype(np.uint8)
        elif pattern == "noise":
            arr = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
        else:  # stripes
            arr = np.zeros((size, size), dtype=np.uint8)
            for y in range(0, size, 8):
                arr[y : y + 4, :] = 255

        img = Image.fromarray(arr, mode="L")
        path = output_dir / f"smoke_{pattern}_{i:03d}.png"
        img.save(path)
        paths.append(path)

    return paths


@dataclass
class EvalResult:
    """Single evaluation result row."""

    sigma: int
    domain: str
    dataset_name: str
    is_real_data: bool
    image_name: str
    image_path: str
    psnr: float
    ssim: float
    noise_seed: int
    model_name: str
    model_checkpoint: str


def discover_images(
    root: Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
) -> list[Path]:
    """Recursively discover image files under root."""
    images = []
    for ext in extensions:
        images.extend(root.rglob(f"*{ext}"))
        images.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def determine_mode(
    microscopy_count: int,
    natural_count: int,
    force_smoke: bool,
) -> str:
    """Determine execution mode based on available data.

    Args:
        microscopy_count: Number of microscopy images found.
        natural_count: Number of natural images found.
        force_smoke: Whether --smoke-mode was specified.

    Returns:
        Mode string: 'full', 'partial', or 'smoke'.
    """
    if force_smoke:
        return "smoke"
    if microscopy_count > 0 and natural_count > 0:
        return "full"
    if microscopy_count > 0 or natural_count > 0:
        return "partial"
    return "smoke"


class _FinetunedModel:
    """Adapter wrapping a trainable SwinIR checkpoint for evaluation."""

    def __init__(self, model, device: str, checkpoint_path: Path) -> None:
        self._model = model
        self.device = device
        self.checkpoint_source = str(checkpoint_path)

    def predict_image(self, image: Image.Image) -> Image.Image:
        """Denoise a single image using the finetuned model."""
        import torch

        if image.mode != "L":
            image = image.convert("L")

        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(tensor)

        output_arr = output.squeeze().cpu().numpy()
        output_arr = np.clip(output_arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output_arr, mode="L")


def load_finetuned_models(
    checkpoint_path: Path, sigmas: list[int], device: str | None = None
) -> dict:
    """Load a finetuned SwinIR model from a training checkpoint.

    Returns the same model wrapped for each sigma level (single model
    evaluated across noise levels).
    """
    import torch
    from inverseops.models.swinir import get_trainable_swinir

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading finetuned checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_config = checkpoint.get("config", {})
    noise_level = ckpt_config.get("model", {}).get("noise_level", 25)

    model = get_trainable_swinir(
        noise_level=noise_level, pretrained=False, device=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapper = _FinetunedModel(model, device, checkpoint_path)
    return {sigma: wrapper for sigma in sigmas}


def load_models(sigmas: list[int]) -> dict:
    """Load SwinIR models for each sigma level.

    Models are cached to avoid redundant loading.

    Args:
        sigmas: List of sigma values to load models for.

    Returns:
        Dictionary mapping sigma to loaded model.
    """
    SwinIR = _get_swinir_class()
    models: dict = {}
    for sigma in sigmas:
        print(f"Loading SwinIR model (sigma={sigma})...")
        model = SwinIR(noise_level=sigma)
        model.load()
        models[sigma] = model
        print(f"  Loaded on device: {model.device}")
    return models


def evaluate_domain(
    models: dict,
    images: list[Path],
    domain: str,
    dataset_name: str,
    is_real_data: bool,
    sigmas: tuple[int, ...],
    seed: int,
    limit: int,
    domain_root: "Path | None" = None,
) -> list[EvalResult]:
    """Evaluate models on a set of images across all sigma levels.

    Loops sigma-first to minimize model switching and enable future batching.

    Args:
        models: Dictionary mapping sigma to loaded SwinIR model.
        images: List of image paths to evaluate.
        domain: Domain name (e.g., 'microscopy', 'natural').
        dataset_name: Dataset identifier (e.g., 'fmd', 'smoke_fixture').
        is_real_data: Whether this is real data (True) or synthetic (False).
        sigmas: Noise levels to test.
        seed: Random seed for noise generation.
        limit: Maximum number of images to evaluate.
        domain_root: Root directory for computing relative paths. If None, absolute
            paths are used.

    Returns:
        List of EvalResult for each (sigma, image) combination.
    """
    results = []
    images = images[:limit]

    for sigma_idx, sigma in enumerate(sigmas):
        model = models[sigma]
        checkpoint_name = Path(model.checkpoint_source).name
        print(f"  Running sigma={sigma} on {len(images)} {domain} images...")

        for i, img_path in enumerate(images):
            clean = Image.open(img_path).convert("L")

            noise_seed = seed + i * 100 + sigma_idx
            noisy = add_gaussian_noise(clean, sigma, seed=noise_seed)

            denoised = model.predict_image(noisy)

            psnr = compute_psnr(clean, denoised)
            ssim = compute_ssim(clean, denoised)

            # Compute relative path for traceability
            try:
                rel_path = (
                    str(img_path.relative_to(domain_root))
                    if domain_root is not None
                    else str(img_path)
                )
            except ValueError:
                rel_path = str(img_path)

            results.append(
                EvalResult(
                    sigma=sigma,
                    domain=domain,
                    dataset_name=dataset_name,
                    is_real_data=is_real_data,
                    image_name=img_path.name,
                    image_path=rel_path,
                    psnr=psnr,
                    ssim=ssim,
                    noise_seed=noise_seed,
                    model_name="swinir",
                    model_checkpoint=checkpoint_name,
                )
            )

            print(
                f"    {img_path.name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}"
            )

    return results


def aggregate_results(results: list[EvalResult]) -> dict:
    """Compute aggregate statistics by domain and sigma.

    Uses population std (ddof=0). For count=1, std=0.0.
    """
    from collections import defaultdict

    groups: dict[tuple[str, int], list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[(r.domain, r.sigma)].append(r)

    agg = {}
    for (domain, sigma), group in groups.items():
        psnrs = [r.psnr for r in group]
        ssims = [r.ssim for r in group]
        n = len(group)

        psnr_mean = sum(psnrs) / n
        ssim_mean = sum(ssims) / n

        # Population std (ddof=0)
        if n == 1:
            psnr_std = 0.0
            ssim_std = 0.0
        else:
            psnr_std = (sum((p - psnr_mean) ** 2 for p in psnrs) / n) ** 0.5
            ssim_std = (sum((s - ssim_mean) ** 2 for s in ssims) / n) ** 0.5

        agg[(domain, sigma)] = {
            "count": n,
            "psnr_mean": psnr_mean,
            "psnr_std": psnr_std,
            "ssim_mean": ssim_mean,
            "ssim_std": ssim_std,
            "dataset_name": group[0].dataset_name,
            "is_real_data": group[0].is_real_data,
            "model_name": group[0].model_name,
            "model_checkpoint": group[0].model_checkpoint,
        }

    return agg


def _print_psnr_table(agg: dict, sigmas: tuple[int, ...]) -> None:
    """Print the per-sigma PSNR table (shared by both model modes)."""
    domains = sorted(set(d for d, _ in agg.keys()))

    header = f"{'Domain':<15}"
    for sigma in sigmas:
        header += f"| sigma={sigma:<4} "
    print(header)
    print("-" * 70)

    for domain in domains:
        row = f"{domain:<15}"
        for sigma in sigmas:
            key = (domain, sigma)
            if key in agg:
                stats = agg[key]
                row += f"| {stats['psnr_mean']:.2f} dB   "
            else:
                row += f"| {'N/A':<10} "
        print(row)

    print("-" * 70)


def print_summary(
    agg: dict,
    sigmas: tuple[int, ...],
    model_mode: str = "pretrained",
    specialization: dict | None = None,
) -> None:
    """Print a compact summary table with mode-appropriate analysis."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    _print_psnr_table(agg, sigmas)

    if model_mode == "finetuned" and specialization is not None:
        _print_specialization_analysis(specialization)
    else:
        _print_decision_gate_analysis(agg)

    print("=" * 70)


def _print_decision_gate_analysis(agg: dict) -> None:
    """Print Day 3 decision gate analysis for pretrained runs."""
    print("\nDECISION GATE ANALYSIS:")

    microscopy_psnrs = []
    natural_psnrs = []

    for (domain, sigma), stats in agg.items():
        if domain == "microscopy":
            microscopy_psnrs.append(stats["psnr_mean"])
        elif domain == "natural":
            natural_psnrs.append(stats["psnr_mean"])

    if microscopy_psnrs and natural_psnrs:
        micro_mean = sum(microscopy_psnrs) / len(microscopy_psnrs)
        nat_mean = sum(natural_psnrs) / len(natural_psnrs)
        gap = nat_mean - micro_mean

        print(f"  Microscopy mean PSNR: {micro_mean:.2f} dB")
        print(f"  Natural mean PSNR:    {nat_mean:.2f} dB")
        print(f"  Gap (natural - micro): {gap:.2f} dB")
        print()

        if gap > 1.0:
            print(
                "  >> Microscopy appears harder than natural images (gap > 1 dB)."
            )
            print("  >> This suggests fine-tuning on microscopy data may help.")
        elif gap > 0.5:
            print("  >> Small gap detected. Fine-tuning may provide modest gains.")
        else:
            print(
                "  >> Microscopy performance is similar to natural images."
            )
            print(
                "  >> Consider using a more challenging microscopy subset (e.g., EM)."
            )
    else:
        print("  Could not compute gap - need both microscopy and natural results.")


def _print_specialization_analysis(specialization: dict) -> None:
    """Print specialization/tradeoff analysis for finetuned runs."""
    print("\nSPECIALIZATION ANALYSIS:")

    micro_psnr = specialization.get("microscopy_mean_psnr")
    nat_psnr = specialization.get("natural_mean_psnr")
    nat_minus_micro = specialization.get("overall_natural_minus_micro_psnr")

    if micro_psnr is not None:
        print(f"  Microscopy mean PSNR:              {micro_psnr:.2f} dB")
    if nat_psnr is not None:
        print(f"  Natural mean PSNR:                 {nat_psnr:.2f} dB")
    if nat_minus_micro is not None:
        print(f"  overall_natural_minus_micro_psnr:   {nat_minus_micro:.2f} dB")

    spec_detected = specialization.get("specialization_detected")
    evidence_tier = specialization.get("evidence_tier")

    print(f"  Specialization detected:           {spec_detected}")
    print(f"  Evidence tier:                     {evidence_tier}")
    print()

    recommendation = specialization.get("recommendation", "")
    if recommendation:
        print(f"  >> {recommendation}")


def save_specialization_summary(
    agg: dict,
    output_path: Path,
    mode: str,
    sigmas: list[int],
    evidence_tier: str,
    microscopy_image_count: int,
    natural_image_count: int,
    compare_csv_path: Path | None = None,
) -> dict:
    """Save specialization_summary.json for finetuned evaluation runs.

    Uses compare_summary.csv for pretrained-vs-finetuned deltas when available.
    When compare data is unavailable, delta-dependent fields are set to None.

    Returns:
        The specialization summary dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute current-run metrics from agg
    micro_psnrs = [
        stats["psnr_mean"] for (domain, _), stats in agg.items()
        if "microscopy" in domain.lower()
    ]
    nat_psnrs = [
        stats["psnr_mean"] for (domain, _), stats in agg.items()
        if "natural" in domain.lower()
    ]
    micro_ssims = [
        stats["ssim_mean"] for (domain, _), stats in agg.items()
        if "microscopy" in domain.lower()
    ]
    nat_ssims = [
        stats["ssim_mean"] for (domain, _), stats in agg.items()
        if "natural" in domain.lower()
    ]

    microscopy_mean_psnr = (
        round(sum(micro_psnrs) / len(micro_psnrs), 4) if micro_psnrs else None
    )
    natural_mean_psnr = (
        round(sum(nat_psnrs) / len(nat_psnrs), 4) if nat_psnrs else None
    )
    microscopy_mean_ssim = (
        round(sum(micro_ssims) / len(micro_ssims), 4) if micro_ssims else None
    )
    natural_mean_ssim = (
        round(sum(nat_ssims) / len(nat_ssims), 4) if nat_ssims else None
    )

    if microscopy_mean_psnr is not None and natural_mean_psnr is not None:
        overall_nat_minus_micro_psnr = round(
            natural_mean_psnr - microscopy_mean_psnr, 4
        )
    else:
        overall_nat_minus_micro_psnr = None

    if microscopy_mean_ssim is not None and natural_mean_ssim is not None:
        overall_nat_minus_micro_ssim = round(
            natural_mean_ssim - microscopy_mean_ssim, 4
        )
    else:
        overall_nat_minus_micro_ssim = None

    # Load compare_summary.csv for pretrained-vs-finetuned deltas
    compare_data: dict[tuple[int, str], dict] = {}
    compare_summary_path_str: str | None = None
    if compare_csv_path is not None and compare_csv_path.exists():
        compare_summary_path_str = str(compare_csv_path)
        with open(compare_csv_path) as f:
            for row in csv.DictReader(f):
                key = (int(row["sigma"]), row["domain"])
                compare_data[key] = row

    has_compare = len(compare_data) > 0

    # Build per-sigma breakdown
    per_sigma: dict[str, dict] = {}
    micro_delta_psnrs: list[float] = []
    nat_delta_psnrs: list[float] = []

    for sigma in sigmas:
        entry: dict[str, float | None] = {}

        # Current metrics from agg
        micro_key_agg = None
        nat_key_agg = None
        for (domain, s) in agg.keys():
            if s == sigma:
                if "microscopy" in domain.lower():
                    micro_key_agg = (domain, s)
                elif "natural" in domain.lower():
                    nat_key_agg = (domain, s)

        entry["microscopy_psnr"] = (
            round(agg[micro_key_agg]["psnr_mean"], 4) if micro_key_agg else None
        )
        entry["natural_psnr"] = (
            round(agg[nat_key_agg]["psnr_mean"], 4) if nat_key_agg else None
        )

        if entry["microscopy_psnr"] is not None and entry["natural_psnr"] is not None:
            entry["natural_minus_micro_psnr"] = round(
                entry["natural_psnr"] - entry["microscopy_psnr"], 4
            )
        else:
            entry["natural_minus_micro_psnr"] = None

        # Deltas from compare CSV
        micro_compare = compare_data.get((sigma, "microscopy"))
        nat_compare = compare_data.get((sigma, "natural"))

        if micro_compare:
            entry["microscopy_delta_psnr_vs_pretrained"] = round(
                float(micro_compare["delta_psnr"]), 4
            )
            entry["microscopy_delta_ssim_vs_pretrained"] = round(
                float(micro_compare["delta_ssim"]), 4
            )
            micro_delta_psnrs.append(float(micro_compare["delta_psnr"]))
        else:
            entry["microscopy_delta_psnr_vs_pretrained"] = None
            entry["microscopy_delta_ssim_vs_pretrained"] = None

        if nat_compare:
            entry["natural_delta_psnr_vs_pretrained"] = round(
                float(nat_compare["delta_psnr"]), 4
            )
            entry["natural_delta_ssim_vs_pretrained"] = round(
                float(nat_compare["delta_ssim"]), 4
            )
            nat_delta_psnrs.append(float(nat_compare["delta_psnr"]))
        else:
            entry["natural_delta_psnr_vs_pretrained"] = None
            entry["natural_delta_ssim_vs_pretrained"] = None

        per_sigma[str(sigma)] = entry

    # Compute specialization flags from pretrained deltas
    if has_compare and micro_delta_psnrs and nat_delta_psnrs:
        mean_micro_delta = sum(micro_delta_psnrs) / len(micro_delta_psnrs)
        mean_nat_delta = sum(nat_delta_psnrs) / len(nat_delta_psnrs)
        microscopy_improved = mean_micro_delta > 0
        natural_regressed = mean_nat_delta < 0
        specialization_detected = microscopy_improved and natural_regressed
    else:
        microscopy_improved = None
        natural_regressed = None
        specialization_detected = None

    # Recommendation
    recommendation = _specialization_recommendation(
        microscopy_improved, natural_regressed, has_compare
    )

    notes = (
        "Post-fine-tuning tradeoff summary. "
        "This is NOT a Day 3 dataset-screening decision. "
        "Use compare_summary.csv for full pretrained-vs-finetuned deltas."
    )

    summary = {
        "model_mode": "finetuned",
        "mode": mode,
        "evidence_tier": evidence_tier,
        "microscopy_image_count": microscopy_image_count,
        "natural_image_count": natural_image_count,
        "sigma_list": sigmas,
        "microscopy_mean_psnr": microscopy_mean_psnr,
        "natural_mean_psnr": natural_mean_psnr,
        "microscopy_mean_ssim": microscopy_mean_ssim,
        "natural_mean_ssim": natural_mean_ssim,
        "overall_natural_minus_micro_psnr": overall_nat_minus_micro_psnr,
        "overall_natural_minus_micro_ssim": overall_nat_minus_micro_ssim,
        "specialization_detected": specialization_detected,
        "microscopy_improved_vs_pretrained": microscopy_improved,
        "natural_regressed_vs_pretrained": natural_regressed,
        "per_sigma": per_sigma,
        "recommendation": recommendation,
        "notes": notes,
        "compare_summary_path": compare_summary_path_str,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Specialization summary saved to: {output_path}")

    return summary


def _specialization_recommendation(
    microscopy_improved: bool | None,
    natural_regressed: bool | None,
    has_compare: bool,
) -> str:
    """Generate recommendation string for finetuned runs."""
    if not has_compare:
        return (
            "Pretrained baseline comparison unavailable. "
            "Current metrics show finetuned model performance; "
            "rerun with --baseline-csv to assess improvement vs pretrained."
        )

    if microscopy_improved and natural_regressed:
        return (
            "Fine-tuning produced domain specialization: "
            "improved microscopy denoising with reduced "
            "out-of-domain natural-image performance."
        )
    elif microscopy_improved and not natural_regressed:
        return (
            "Fine-tuning improved both microscopy and natural-image "
            "denoising on this benchmark slice."
        )
    else:
        return (
            "Fine-tuning did not improve microscopy enough on this "
            "benchmark slice; review training setup and checkpoint quality."
        )


def save_csv(results: list[EvalResult], output_path: Path, mode: str) -> None:
    """Save per-image results to baseline_metrics.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sigma", "noise_type", "domain", "dataset_name", "is_real_data",
        "image_name", "image_path", "psnr", "ssim", "seed", "model_name",
        "model_checkpoint", "mode"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "sigma": r.sigma,
                "noise_type": "synthetic_gaussian",
                "domain": r.domain,
                "dataset_name": r.dataset_name,
                "is_real_data": r.is_real_data,
                "image_name": r.image_name,
                "image_path": r.image_path,
                "psnr": f"{r.psnr:.4f}",
                "ssim": f"{r.ssim:.4f}",
                "seed": r.noise_seed,
                "model_name": r.model_name,
                "model_checkpoint": r.model_checkpoint,
                "mode": mode,
            })

    print(f"\nResults saved to: {output_path}")


def save_summary_json(agg: dict, output_path: Path) -> None:
    """Save aggregate statistics to JSON."""
    # Convert tuple keys to strings
    json_agg = {}
    for (domain, sigma), stats in agg.items():
        key = f"{domain}_sigma{sigma}"
        json_agg[key] = stats

    with open(output_path, "w") as f:
        json.dump(json_agg, f, indent=2)


def save_summary_csv(
    agg: dict,
    output_path: Path,
    seed: int,
    mode: str,
    decision_gate_valid: bool,
    evidence_tier: str,
) -> None:
    """Save aggregate statistics to baseline_summary.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sigma", "noise_type", "domain", "dataset_name", "is_real_data",
        "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
        "seed", "model_name", "model_checkpoint", "mode", "decision_gate_valid",
        "evidence_tier"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (domain, sigma), stats in sorted(agg.items()):
            writer.writerow({
                "sigma": sigma,
                "noise_type": "synthetic_gaussian",
                "domain": domain,
                "dataset_name": stats["dataset_name"],
                "is_real_data": stats["is_real_data"],
                "count": stats["count"],
                "psnr_mean": f"{stats['psnr_mean']:.4f}",
                "psnr_std": f"{stats['psnr_std']:.4f}",
                "ssim_mean": f"{stats['ssim_mean']:.4f}",
                "ssim_std": f"{stats['ssim_std']:.4f}",
                "seed": seed,
                "model_name": stats["model_name"],
                "model_checkpoint": stats["model_checkpoint"],
                "mode": mode,
                "decision_gate_valid": decision_gate_valid,
                "evidence_tier": evidence_tier,
            })

    print(f"Summary saved to: {output_path}")


def save_decision_json(
    agg: dict,
    output_path: Path,
    mode: str,
    sigmas: list[int],
    gap_threshold_db: float,
    is_pilot: bool = False,
) -> dict:
    """Save day3_decision.json with decision gate analysis.

    Returns:
        The decision dictionary for use in stdout summary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count images per domain type (total across all sigmas)
    microscopy_count = sum(
        stats["count"] for (domain, _), stats in agg.items()
        if "microscopy" in domain.lower()
    )
    natural_count = sum(
        stats["count"] for (domain, _), stats in agg.items()
        if "natural" in domain.lower()
    )

    # Compute unique image counts (divide by number of sigmas evaluated)
    num_sigmas = len(sigmas)
    microscopy_image_count = microscopy_count // num_sigmas if num_sigmas > 0 else 0
    natural_image_count = natural_count // num_sigmas if num_sigmas > 0 else 0

    # Compute evidence tier
    evidence_tier = compute_evidence_tier(microscopy_image_count, natural_image_count)

    # Compute per-sigma gaps
    per_sigma_gap_db = {}
    for sigma in sigmas:
        micro_key = None
        nat_key = None
        for (domain, s) in agg.keys():
            if s == sigma:
                if "microscopy" in domain.lower():
                    micro_key = (domain, s)
                elif "natural" in domain.lower():
                    nat_key = (domain, s)

        if micro_key and nat_key:
            gap = agg[nat_key]["psnr_mean"] - agg[micro_key]["psnr_mean"]
            per_sigma_gap_db[str(sigma)] = round(gap, 2)

    # Compute overall means
    micro_psnrs = [
        stats["psnr_mean"] for (domain, _), stats in agg.items()
        if "microscopy" in domain.lower()
    ]
    nat_psnrs = [
        stats["psnr_mean"] for (domain, _), stats in agg.items()
        if "natural" in domain.lower()
    ]

    overall_micro_psnr = sum(micro_psnrs) / len(micro_psnrs) if micro_psnrs else None
    overall_natural_psnr = sum(nat_psnrs) / len(nat_psnrs) if nat_psnrs else None

    if overall_micro_psnr is not None and overall_natural_psnr is not None:
        overall_gap_db = round(overall_natural_psnr - overall_micro_psnr, 2)
    else:
        overall_gap_db = None

    # Decision logic: valid only when full mode, all required sigmas, both domains
    required_sigmas = set(SUPPORTED_SIGMAS)
    observed_sigmas = set(sigmas)
    has_microscopy = microscopy_image_count > 0
    has_natural = natural_image_count > 0
    decision_gate_valid = (
        mode == "full"
        and observed_sigmas == required_sigmas
        and has_microscopy
        and has_natural
    )
    dataset_locked = decision_gate_valid

    if decision_gate_valid and overall_gap_db is not None:
        if overall_gap_db > gap_threshold_db:
            recommendation = (
                f"Retain FMD as primary dataset; microscopy underperforms "
                f"natural images by {overall_gap_db:.1f} dB under matched "
                "synthetic Gaussian corruption."
            )
        elif overall_gap_db > 0.5:
            recommendation = (
                f"Modest domain gap detected ({overall_gap_db:.1f} dB). "
                "Fine-tuning may provide incremental gains."
            )
        else:
            recommendation = (
                "Microscopy performance is similar to natural images. "
                "Consider using a more challenging microscopy subset (e.g., EM)."
            )
    else:
        recommendation = (
            "Insufficient real microscopy-vs-natural evidence for Day 3 decision gate."
        )

    # Notes with pilot-specific caveat
    if is_pilot:
        notes = (
            "This is a protocol-complete pilot benchmark on a minimal subset; "
            "evidence strength is limited by sample count."
        )
    else:
        notes = "Day 3 uses synthetic Gaussian noise on clean reference images."

    decision = {
        "mode": mode,
        "decision_gate_valid": decision_gate_valid,
        "dataset_locked": dataset_locked,
        "microscopy_image_count": microscopy_image_count,
        "natural_image_count": natural_image_count,
        "evidence_tier": evidence_tier,
        "sigma_list": sigmas,
        "gap_threshold_db": gap_threshold_db,
        "per_sigma_gap_db": per_sigma_gap_db,
        "overall_micro_psnr": (
            round(overall_micro_psnr, 2) if overall_micro_psnr else None
        ),
        "overall_natural_psnr": (
            round(overall_natural_psnr, 2) if overall_natural_psnr else None
        ),
        "overall_gap_db": overall_gap_db,
        "recommendation": recommendation,
        "notes": notes,
    }

    with open(output_path, "w") as f:
        json.dump(decision, f, indent=2)

    print(f"Decision saved to: {output_path}")

    return decision


def build_comparison_csv(
    baseline_csv: Path,
    finetuned_csv: Path,
    output_path: Path,
    checkpoint_path: Path,
) -> None:
    """Build comparison CSV from pretrained and finetuned summary CSVs.

    Matches rows by (sigma, domain) and computes deltas.
    """
    baseline = {}
    with open(baseline_csv) as f:
        for row in csv.DictReader(f):
            key = (int(row["sigma"]), row["domain"])
            baseline[key] = row

    finetuned = {}
    with open(finetuned_csv) as f:
        for row in csv.DictReader(f):
            key = (int(row["sigma"]), row["domain"])
            finetuned[key] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sigma", "domain",
        "pretrained_psnr_mean", "finetuned_psnr_mean", "delta_psnr",
        "pretrained_ssim_mean", "finetuned_ssim_mean", "delta_ssim",
        "checkpoint_path",
    ]

    all_keys = sorted(set(baseline.keys()) & set(finetuned.keys()))
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sigma, domain in all_keys:
            b = baseline[(sigma, domain)]
            ft = finetuned[(sigma, domain)]
            pre_psnr = float(b["psnr_mean"])
            ft_psnr = float(ft["psnr_mean"])
            pre_ssim = float(b["ssim_mean"])
            ft_ssim = float(ft["ssim_mean"])

            writer.writerow({
                "sigma": sigma,
                "domain": domain,
                "pretrained_psnr_mean": f"{pre_psnr:.4f}",
                "finetuned_psnr_mean": f"{ft_psnr:.4f}",
                "delta_psnr": f"{ft_psnr - pre_psnr:.4f}",
                "pretrained_ssim_mean": f"{pre_ssim:.4f}",
                "finetuned_ssim_mean": f"{ft_ssim:.4f}",
                "delta_ssim": f"{ft_ssim - pre_ssim:.4f}",
                "checkpoint_path": str(checkpoint_path),
            })

    print(f"Comparison saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained SwinIR on microscopy and natural images."
    )
    parser.add_argument(
        "--microscopy-root",
        type=Path,
        default=Path("data/raw/fmd"),
        help="Root directory for microscopy images",
    )
    parser.add_argument(
        "--natural-root",
        type=Path,
        default=None,
        help="Root directory for natural reference images (optional)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to use for microscopy images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of images per domain to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/baseline/baseline_metrics.csv"),
        help="Output CSV path for per-image metrics",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV path for summary (default: next to output-csv)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="inverseops-baseline",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--single-sigma",
        type=int,
        choices=[15, 25, 50],
        default=None,
        help="Evaluate only a single sigma (default: all)",
    )
    parser.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help="Allow partial/smoke mode instead of error when datasets missing",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Force smoke mode with synthetic images",
    )
    parser.add_argument(
        "--smoke-count",
        type=int,
        default=5,
        help="Number of synthetic images per domain in smoke mode",
    )
    parser.add_argument(
        "--gap-threshold-db",
        type=float,
        default=1.0,
        help="PSNR gap threshold (dB) for domain difference (default: 1.0)",
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="Fast development mode: --single-sigma 50 --limit 2 --no-wandb",
    )
    parser.add_argument(
        "--pilot-dev",
        action="store_true",
        help="Laptop-friendly full-sigma pilot run: all sigmas, tiny matched subset, no W&B",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a training checkpoint (used with --model-mode finetuned)",
    )
    parser.add_argument(
        "--model-mode",
        type=str,
        choices=["pretrained", "finetuned"],
        default="pretrained",
        help="Model mode: pretrained (default) or finetuned",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Pretrained summary CSV for comparison (produces compare_summary.csv)",
    )

    args = parser.parse_args()

    # Validate finetuned args
    if args.model_mode == "finetuned" and args.checkpoint is None:
        print("Error: --checkpoint is required when --model-mode is 'finetuned'")
        return 1
    if args.model_mode == "finetuned" and args.checkpoint and not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    # Apply pilot-dev defaults (full sigma coverage, minimal subset)
    if args.pilot_dev:
        args.single_sigma = None  # Force full sigma set
        args.limit = min(args.limit, 1)  # Cap to 1 image per domain
        args.no_wandb = True
        print(
            "PILOT DEV MODE: full sigma set [15,25,50], "
            f"limit={args.limit} per domain, no_wandb=True"
        )

    # Apply fast-dev defaults
    if args.fast_dev:
        if args.single_sigma is None:
            args.single_sigma = 50
        args.limit = min(args.limit, 2)
        args.no_wandb = True
        print(
            f"FAST DEV MODE: single_sigma={args.single_sigma}, "
            f"limit<={args.limit}, no_wandb=True"
        )

    # Determine sigmas to evaluate
    if args.single_sigma:
        sigmas = (args.single_sigma,)
    else:
        sigmas = SUPPORTED_SIGMAS  # (15, 25, 50)

    # Set default summary CSV path
    if args.summary_csv is None:
        args.summary_csv = args.output_csv.parent / "baseline_summary.csv"

    # Collect images from each domain
    microscopy_images: list[Path] = []
    natural_images: list[Path] = []
    microscopy_dataset_name = "fmd"
    natural_dataset_name = "natural"

    if not args.smoke_mode:
        # Try to load microscopy images
        if args.microscopy_root and args.microscopy_root.exists():
            try:
                microscopy_ds = MicroscopyDataset(
                    root_dir=args.microscopy_root,
                    split=args.split,
                    seed=args.seed,
                )
                microscopy_ds.prepare()
                microscopy_images = [
                    microscopy_ds.image_path(i) for i in range(len(microscopy_ds))
                ]
                print(
                    f"Found {len(microscopy_images)} microscopy images "
                    f"({args.split} split)"
                )
            except ValueError as e:
                print(f"Warning: Could not load microscopy images: {e}")

        # Try to load natural images
        if args.natural_root and args.natural_root.exists():
            natural_images = discover_images(args.natural_root)
            if natural_images:
                print(f"Found {len(natural_images)} natural images")
            else:
                print(f"Warning: No images found in {args.natural_root}")

    # Determine execution mode
    mode = determine_mode(
        microscopy_count=len(microscopy_images),
        natural_count=len(natural_images),
        force_smoke=args.smoke_mode,
    )

    # Check if we should fail on missing data
    if mode != "full" and not args.allow_missing_datasets and not args.smoke_mode:
        print("Error: Missing datasets and --allow-missing-datasets not set.")
        print("  Microscopy images: ", len(microscopy_images))
        print("  Natural images: ", len(natural_images))
        print(
            "Use --allow-missing-datasets for partial mode "
            "or --smoke-mode for smoke testing."
        )
        return 1

    decision_gate_valid = mode == "full"

    # Print mode warning
    if mode == "smoke":
        print("\n" + "=" * 80)
        print("WARNING: SMOKE MODE - RESULTS ARE NOT SCIENTIFICALLY MEANINGFUL")
        print("These outputs verify pipeline execution only.")
        print("DO NOT use for Day 3 dataset decision.")
        print("=" * 80 + "\n")
    elif mode == "partial":
        print("\n" + "=" * 80)
        print("WARNING: PARTIAL MODE - Only one domain available")
        print("Day 3 decision gate cannot be concluded with one domain.")
        print("=" * 80 + "\n")

    # Initialize smoke directories (set only if mode == smoke)
    smoke_micro_dir = None
    smoke_nat_dir = None

    # Generate smoke fixtures if needed
    if mode == "smoke":
        import tempfile
        smoke_dir = Path(tempfile.mkdtemp(prefix="inverseops_smoke_"))
        smoke_micro_dir = smoke_dir / "microscopy"
        smoke_nat_dir = smoke_dir / "natural"

        microscopy_images = generate_smoke_fixtures(
            smoke_micro_dir, count=args.smoke_count, seed=args.seed
        )
        natural_images = generate_smoke_fixtures(
            smoke_nat_dir, count=args.smoke_count, seed=args.seed + 1000
        )
        microscopy_dataset_name = "smoke_fixture"
        natural_dataset_name = "smoke_fixture"
        print(f"Generated {args.smoke_count} smoke fixtures per domain")

    # Load models
    if args.model_mode == "finetuned":
        print(f"\nLoading finetuned model from: {args.checkpoint}")
        models = load_finetuned_models(args.checkpoint, list(sigmas))
    else:
        print(f"\nLoading SwinIR models for sigmas: {sigmas}")
        models = load_models(list(sigmas))

    # Initialize W&B
    if not args.no_wandb:
        if wandb is None:
            print("Error: wandb is required for W&B logging. Install with: pip install wandb")
            return 1
        checkpoint_sources = {
            sigma: models[sigma].checkpoint_source for sigma in sigmas
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "mode": mode,
                "decision_gate_valid": decision_gate_valid,
                "microscopy_root": (
                    str(args.microscopy_root) if args.microscopy_root else None
                ),
                "natural_root": str(args.natural_root) if args.natural_root else None,
                "split": args.split,
                "limit": args.limit,
                "seed": args.seed,
                "sigmas": list(sigmas),
                "gap_threshold_db": args.gap_threshold_db,
                "checkpoint_sources": checkpoint_sources,
                "device": models[sigmas[0]].device,
                "output_csv": str(args.output_csv),
                "summary_csv": str(args.summary_csv),
            },
        )

    all_results: list[EvalResult] = []

    # Evaluate microscopy
    if microscopy_images:
        domain_name = "smoke_microscopy" if mode == "smoke" else "microscopy"
        print(f"\nEvaluating {domain_name} images...")
        micro_domain_root = (
            smoke_micro_dir if mode == "smoke" else args.microscopy_root
        )
        micro_results = evaluate_domain(
            models=models,
            images=microscopy_images,
            domain=domain_name,
            dataset_name=microscopy_dataset_name,
            is_real_data=(mode != "smoke"),
            sigmas=sigmas,
            seed=args.seed,
            limit=args.limit,
            domain_root=micro_domain_root,
        )
        all_results.extend(micro_results)

    # Evaluate natural
    if natural_images:
        domain_name = "smoke_natural" if mode == "smoke" else "natural"
        print(f"\nEvaluating {domain_name} images...")
        nat_domain_root = smoke_nat_dir if mode == "smoke" else args.natural_root
        nat_results = evaluate_domain(
            models=models,
            images=natural_images,
            domain=domain_name,
            dataset_name=natural_dataset_name,
            is_real_data=(mode != "smoke"),
            sigmas=sigmas,
            seed=args.seed,
            limit=args.limit,
            domain_root=nat_domain_root,
        )
        all_results.extend(nat_results)

    # Aggregate results
    agg = aggregate_results(all_results)

    # Compute image counts and evidence tier for outputs
    num_sigmas = len(sigmas)
    microscopy_result_count = sum(
        1 for r in all_results if "microscopy" in r.domain.lower()
    )
    natural_result_count = sum(
        1 for r in all_results if "natural" in r.domain.lower()
    )
    microscopy_image_count = microscopy_result_count // num_sigmas if num_sigmas > 0 else 0
    natural_image_count = natural_result_count // num_sigmas if num_sigmas > 0 else 0
    evidence_tier = compute_evidence_tier(microscopy_image_count, natural_image_count)

    # Save per-image and summary CSVs (shared by both modes)
    save_csv(all_results, args.output_csv, mode)
    save_summary_csv(
        agg, args.summary_csv, args.seed, mode, decision_gate_valid, evidence_tier
    )

    # Branch on model mode for artifact generation
    if args.model_mode == "finetuned":
        # 1. Build compare_summary.csv first (needed by specialization summary)
        compare_path = None
        if args.baseline_csv and args.baseline_csv.exists():
            compare_path = args.output_csv.parent / "compare_summary.csv"
            build_comparison_csv(
                args.baseline_csv, args.summary_csv, compare_path, args.checkpoint
            )
        elif args.baseline_csv:
            print(f"Warning: Baseline CSV not found: {args.baseline_csv}")

        # 2. Save specialization summary (NOT day3_decision.json)
        specialization_path = args.output_csv.parent / "specialization_summary.json"
        specialization_result = save_specialization_summary(
            agg=agg,
            output_path=specialization_path,
            mode=mode,
            sigmas=list(sigmas),
            evidence_tier=evidence_tier,
            microscopy_image_count=microscopy_image_count,
            natural_image_count=natural_image_count,
            compare_csv_path=compare_path,
        )

        # 3. Print summary with specialization analysis
        print_summary(
            agg, sigmas,
            model_mode="finetuned",
            specialization=specialization_result,
        )

        # 4. W&B artifact uses specialization_summary.json
        wandb_artifact_path = specialization_path

    else:
        # Pretrained: keep Day 3 decision gate flow
        decision_json_path = args.output_csv.parent / "day3_decision.json"
        decision_result = save_decision_json(
            agg, decision_json_path, mode, list(sigmas), args.gap_threshold_db,
            is_pilot=args.pilot_dev,
        )

        print_summary(agg, sigmas, model_mode="pretrained")

        specialization_result = None
        wandb_artifact_path = decision_json_path

    # Log to W&B
    if not args.no_wandb:
        for (domain, sigma), stats in agg.items():
            wandb.log({
                f"{domain}/sigma_{sigma}/psnr_mean": stats["psnr_mean"],
                f"{domain}/sigma_{sigma}/ssim_mean": stats["ssim_mean"],
                f"{domain}/sigma_{sigma}/count": stats["count"],
            })

        table_data = [
            [r.sigma, r.domain, r.image_name, r.image_path, r.psnr, r.ssim,
             r.model_checkpoint]
            for r in all_results
        ]
        wandb_table = wandb.Table(
            columns=["sigma", "domain", "image_name", "image_path", "psnr", "ssim",
                     "checkpoint"],
            data=table_data,
        )
        wandb.log({"results_table": wandb_table})

        artifact = wandb.Artifact("evaluation_metrics", type="evaluation")
        artifact.add_file(str(args.output_csv))
        artifact.add_file(str(args.summary_csv))
        artifact.add_file(str(wandb_artifact_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("\nW&B run complete.")

    # Print final summary
    print("\n" + "=" * 50)
    print("RUN SUMMARY")
    print("=" * 50)
    print(f"  Mode: {mode}")
    print(f"  Model mode: {args.model_mode}")
    print(f"  Sigmas: {list(sigmas)}")
    print(f"  Microscopy images: {microscopy_image_count}")
    print(f"  Natural images: {natural_image_count}")
    print(f"  Evidence tier: {evidence_tier}")

    if args.model_mode == "finetuned":
        spec_detected = specialization_result.get("specialization_detected")
        print(f"  Specialization detected: {spec_detected}")
        print(f"  Artifact: specialization_summary.json")
    else:
        print(f"  Decision gate valid: {decision_result['decision_gate_valid']}")
        print(f"  Artifact: day3_decision.json")

    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())

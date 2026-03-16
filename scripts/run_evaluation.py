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
import wandb
from PIL import Image

from inverseops.data.degradations import SUPPORTED_SIGMAS, add_gaussian_noise
from inverseops.data.microscopy import MicroscopyDataset
from inverseops.evaluation.metrics import compute_psnr, compute_ssim
from inverseops.models.swinir import SwinIRBaseline


def generate_smoke_fixtures(output_dir: Path, count: int = 5, seed: int = 42) -> list[Path]:
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
    image_name: str
    psnr: float
    ssim: float


def discover_images(
    root: Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
) -> list[Path]:
    """Recursively discover image files under root."""
    images = []
    for ext in extensions:
        images.extend(root.rglob(f"*{ext}"))
        images.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def evaluate_domain(
    model: SwinIRBaseline,
    images: list[Path],
    domain: str,
    sigmas: tuple[int, ...],
    seed: int,
    limit: int,
) -> list[EvalResult]:
    """Evaluate model on a set of images across all sigma levels.

    Args:
        model: Loaded SwinIR model.
        images: List of image paths to evaluate.
        domain: Domain name ('microscopy' or 'natural').
        sigmas: Noise levels to test.
        seed: Random seed for noise generation.
        limit: Maximum number of images to evaluate.

    Returns:
        List of EvalResult for each (image, sigma) combination.
    """
    results = []
    images = images[:limit]

    for i, img_path in enumerate(images):
        # Load clean image as grayscale
        clean = Image.open(img_path).convert("L")

        for sigma_idx, sigma in enumerate(sigmas):
            # Generate noisy image with deterministic seed
            # Use image index + sigma index for reproducibility
            noise_seed = seed + i * 100 + sigma_idx
            noisy = add_gaussian_noise(clean, sigma, seed=noise_seed)

            # Denoise
            denoised = model.predict_image(noisy)

            # Compute metrics
            psnr = compute_psnr(clean, denoised)
            ssim = compute_ssim(clean, denoised)

            results.append(
                EvalResult(
                    sigma=sigma,
                    domain=domain,
                    image_name=img_path.name,
                    psnr=psnr,
                    ssim=ssim,
                )
            )

            print(
                f"  [{domain}] {img_path.name} sigma={sigma}: "
                f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}"
            )

    return results


def aggregate_results(results: list[EvalResult]) -> dict:
    """Compute aggregate statistics by domain and sigma."""
    from collections import defaultdict

    # Group by (domain, sigma)
    groups: dict[tuple[str, int], list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[(r.domain, r.sigma)].append(r)

    agg = {}
    for (domain, sigma), group in groups.items():
        psnrs = [r.psnr for r in group]
        ssims = [r.ssim for r in group]
        agg[(domain, sigma)] = {
            "count": len(group),
            "psnr_mean": sum(psnrs) / len(psnrs),
            "psnr_std": (
                sum((p - sum(psnrs) / len(psnrs)) ** 2 for p in psnrs) / len(psnrs)
            )
            ** 0.5,
            "ssim_mean": sum(ssims) / len(ssims),
        }

    return agg


def print_summary(agg: dict, sigmas: tuple[int, ...]) -> None:
    """Print a compact summary table."""
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 70)

    # Group by domain
    domains = sorted(set(d for d, _ in agg.keys()))

    # Header
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

    # Decision gate analysis
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

    print("=" * 70)


def save_csv(results: list[EvalResult], output_path: Path) -> None:
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "domain", "image_name", "psnr", "ssim"])
        for r in results:
            writer.writerow([r.sigma, r.domain, r.image_name, r.psnr, r.ssim])

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
        help="Output CSV path",
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
        "--noise-level",
        type=int,
        default=25,
        choices=[15, 25, 50],
        help="Noise level for the SwinIR model (default: 25)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.microscopy_root.exists():
        print(f"Error: Microscopy root not found: {args.microscopy_root}")
        print("Run 'bash scripts/download_data.sh' and add images to data/raw/fmd/")
        return 1

    if args.natural_root and not args.natural_root.exists():
        print(f"Error: Natural image root not found: {args.natural_root}")
        return 1

    sigmas = SUPPORTED_SIGMAS  # (15, 25, 50)

    # Initialize W&B
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "microscopy_root": str(args.microscopy_root),
                "natural_root": str(args.natural_root) if args.natural_root else None,
                "split": args.split,
                "limit": args.limit,
                "seed": args.seed,
                "sigmas": list(sigmas),
                "model_noise_level": args.noise_level,
                "task": "baseline_evaluation",
            },
        )

    # Load model
    print(f"\nLoading SwinIR model (noise_level={args.noise_level})...")
    model = SwinIRBaseline(noise_level=args.noise_level)
    model.load()
    print(f"Model loaded on device: {model.device}")

    all_results: list[EvalResult] = []

    # Evaluate microscopy
    print(f"\nEvaluating microscopy images ({args.split} split)...")
    microscopy_ds = MicroscopyDataset(
        root_dir=args.microscopy_root,
        split=args.split,
        seed=args.seed,
    )
    microscopy_ds.prepare()
    microscopy_images = [microscopy_ds.image_path(i) for i in range(len(microscopy_ds))]
    print(f"Found {len(microscopy_images)} images in {args.split} split")

    micro_results = evaluate_domain(
        model=model,
        images=microscopy_images,
        domain="microscopy",
        sigmas=sigmas,
        seed=args.seed,
        limit=args.limit,
    )
    all_results.extend(micro_results)

    # Evaluate natural images if provided
    if args.natural_root:
        print("\nEvaluating natural images...")
        natural_images = discover_images(args.natural_root)
        if not natural_images:
            print(f"Warning: No images found in {args.natural_root}")
        else:
            print(f"Found {len(natural_images)} natural images")
            nat_results = evaluate_domain(
                model=model,
                images=natural_images,
                domain="natural",
                sigmas=sigmas,
                seed=args.seed,
                limit=args.limit,
            )
            all_results.extend(nat_results)

    # Aggregate and display results
    agg = aggregate_results(all_results)
    print_summary(agg, sigmas)

    # Save outputs
    save_csv(all_results, args.output_csv)

    summary_json_path = args.output_csv.parent / "summary.json"
    save_summary_json(agg, summary_json_path)
    print(f"Summary saved to: {summary_json_path}")

    # Log to W&B
    if not args.no_wandb:
        # Log aggregate metrics
        for (domain, sigma), stats in agg.items():
            wandb.log(
                {
                    f"{domain}/sigma_{sigma}/psnr_mean": stats["psnr_mean"],
                    f"{domain}/sigma_{sigma}/ssim_mean": stats["ssim_mean"],
                    f"{domain}/sigma_{sigma}/count": stats["count"],
                }
            )

        # Log summary table
        table_data = [
            [r.sigma, r.domain, r.image_name, r.psnr, r.ssim] for r in all_results
        ]
        wandb_table = wandb.Table(
            columns=["sigma", "domain", "image_name", "psnr", "ssim"],
            data=table_data,
        )
        wandb.log({"results_table": wandb_table})

        # Log artifacts
        artifact = wandb.Artifact("baseline_metrics", type="evaluation")
        artifact.add_file(str(args.output_csv))
        artifact.add_file(str(summary_json_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("\nW&B run complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

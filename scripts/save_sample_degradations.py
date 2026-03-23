#!/usr/bin/env python3
"""Generate and save sample degraded images for visual inspection.

Usage:
    python scripts/save_sample_degradations.py [root_dir]

Arguments:
    root_dir: Path to microscopy images (default: data/raw/fmd)

Outputs saved to artifacts/samples/:
    - clean.png
    - noisy_sigma_15.png
    - noisy_sigma_25.png
    - noisy_sigma_50.png
    - metadata.json
"""

import json
import sys
from pathlib import Path


def main(root_dir: str = "data/raw/fmd") -> None:
    """Load a clean image and generate noisy variants."""
    from inverseops.data.degradations import SUPPORTED_SIGMAS, generate_noisy_variants
    from inverseops.data.microscopy import MicroscopyDataset

    # Setup paths
    root = Path(root_dir)
    output_dir = Path("artifacts/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    try:
        dataset = MicroscopyDataset(root_dir=root, split="train", seed=42)
        dataset.prepare()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if len(dataset) == 0:
        print(
            f"Error: No images found in train split.\n"
            f"Check that image files exist in {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load first image
    clean_image = dataset.load_image(0)
    source_path = dataset.image_path(0)
    print(f"Loaded clean image: {source_path}")

    # Save clean image
    clean_path = output_dir / "clean.png"
    clean_image.save(clean_path)
    print(f"Saved: {clean_path}")

    # Generate and save noisy variants
    variants = generate_noisy_variants(clean_image, sigmas=SUPPORTED_SIGMAS, seed=42)

    output_files = {"clean": str(clean_path)}
    for sigma, noisy_image in variants.items():
        noisy_path = output_dir / f"noisy_sigma_{sigma}.png"
        noisy_image.save(noisy_path)
        print(f"Saved: {noisy_path}")
        output_files[f"noisy_sigma_{sigma}"] = str(noisy_path)

    # Save metadata
    metadata = {
        "source_image": str(source_path.name),
        "source_path": str(source_path),
        "seed": 42,
        "sigma_values": list(SUPPORTED_SIGMAS),
        "output_files": output_files,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    print(f"\nAll samples saved to {output_dir}/")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/fmd"
    main(root)

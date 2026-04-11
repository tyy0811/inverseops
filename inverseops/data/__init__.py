"""Data loading, dataset interfaces, and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inverseops.data.degradations import (
    SUPPORTED_SIGMAS,
    add_gaussian_noise,
    generate_noisy_variants,
)
from inverseops.data.transforms import center_crop, normalize_to_uint8, to_grayscale
from inverseops.data.ixi import IXIDataset
from inverseops.data.w2s import W2SDataset

__all__ = [
    "SUPPORTED_SIGMAS",
    "add_gaussian_noise",
    "generate_noisy_variants",
    "to_grayscale",
    "center_crop",
    "normalize_to_uint8",
    "DATASET_REGISTRY",
    "DATASET_DATA_RANGE",
    "build_dataset",
    "W2SDataset",
    "IXIDataset",
]

DATASET_REGISTRY: dict[str, type] = {
    "w2s": W2SDataset,
    "ixi": IXIDataset,
}

# Fixed data range per dataset for PSNR/SSIM computation.
# Must be constant across all samples to produce comparable numbers.
# Used by both the Trainer (validation PSNR) and run_evaluation.py.
DATASET_DATA_RANGE: dict[str, float] = {
    "w2s": 255.0,  # original intensities roughly 0-255
    "ixi": 1.0,    # normalized to [0, 1]; denormalize is identity
}

# Keys forwarded from data config to dataset constructor, per dataset
_DATASET_FORWARD_KEYS: dict[str, set[str]] = {
    "w2s": {"avg_levels", "patch_size", "seed", "val_fraction", "test_fraction"},
    "ixi": {"sigma", "patch_size", "seed", "val_fraction", "test_fraction"},
}
# Top-level config keys forwarded to dataset constructor, per dataset.
# W2S uses "task" to switch between denoise/sr; IXI is denoise-only.
_TOP_LEVEL_FORWARD_KEYS: dict[str, set[str]] = {
    "w2s": {"task"},
    "ixi": set(),
}


def build_dataset(config: dict, split: str = "train", **kwargs: Any):
    """Build a dataset from config using the registry.

    Instantiates the dataset class, forwards relevant config keys as
    constructor kwargs, calls prepare(), and returns a ready-to-use dataset.

    Config key: config["data"]["dataset"] selects the registered dataset class.
    """
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset", "")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: {sorted(DATASET_REGISTRY.keys())}"
        )

    cls = DATASET_REGISTRY[dataset_name]

    # Build constructor kwargs from config
    ctor_kwargs: dict[str, Any] = {
        "root_dir": data_cfg.get("train_root", ""),
        "split": split,
        **kwargs,
    }

    # Forward splits_path if present
    splits_path = data_cfg.get("splits_path")
    if splits_path:
        ctor_kwargs["splits_path"] = Path(splits_path)

    # Forward dataset-specific keys from data config
    forward_keys = _DATASET_FORWARD_KEYS.get(dataset_name, set())
    for key in forward_keys:
        if key in data_cfg:
            ctor_kwargs[key] = data_cfg[key]

    # Forward top-level config keys (e.g. task for W2S denoise/sr switch)
    top_keys = _TOP_LEVEL_FORWARD_KEYS.get(dataset_name, set())
    for key in top_keys:
        if key in config and key not in ctor_kwargs:
            ctor_kwargs[key] = config[key]

    dataset = cls(**ctor_kwargs)
    dataset.prepare()
    return dataset

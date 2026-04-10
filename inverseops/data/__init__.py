"""Data loading, dataset interfaces, and registry."""

from __future__ import annotations

from typing import Any

from inverseops.data.degradations import (
    SUPPORTED_SIGMAS,
    add_gaussian_noise,
    generate_noisy_variants,
)
from inverseops.data.transforms import center_crop, normalize_to_uint8, to_grayscale

__all__ = [
    "SUPPORTED_SIGMAS",
    "add_gaussian_noise",
    "generate_noisy_variants",
    "to_grayscale",
    "center_crop",
    "normalize_to_uint8",
    "DATASET_REGISTRY",
    "build_dataset",
]

# W2S registered in Phase 1 Day 1, IXI in Phase 2 Day 5
DATASET_REGISTRY: dict[str, type] = {}


def build_dataset(config: dict, split: str = "train", **kwargs: Any):
    """Build a dataset from config using the registry.

    Config key: config["data"]["dataset"] selects the registered dataset class.
    """
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset", "")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]

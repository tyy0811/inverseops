"""Data loading, dataset interfaces, and registry."""

from __future__ import annotations

from typing import Any

from inverseops.data.degradations import (
    SUPPORTED_SIGMAS,
    add_gaussian_noise,
    generate_noisy_variants,
)
from inverseops.data.microscopy import MicroscopyDataset
from inverseops.data.torch_datasets import MicroscopyTrainDataset, RealNoiseTrainDataset
from inverseops.data.transforms import center_crop, normalize_to_uint8, to_grayscale

__all__ = [
    "MicroscopyDataset",
    "MicroscopyTrainDataset",
    "SUPPORTED_SIGMAS",
    "add_gaussian_noise",
    "generate_noisy_variants",
    "to_grayscale",
    "center_crop",
    "normalize_to_uint8",
    "DATASET_REGISTRY",
    "build_dataset",
]

DATASET_REGISTRY: dict[str, type] = {
    "synthetic": MicroscopyTrainDataset,
    "real": RealNoiseTrainDataset,
}


def build_dataset(config: dict, split: str = "train", **kwargs: Any):
    """Build a dataset from config using the registry.

    Config key: config["data"]["noise_source"] (default: "synthetic").
    """
    data_cfg = config.get("data", {})
    noise_source = data_cfg.get("noise_source", "synthetic")
    if noise_source not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type: {noise_source!r}. "
            f"Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[noise_source]

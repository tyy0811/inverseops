"""Tests for model and dataset registries."""

import pytest

from inverseops.config import validate_config
from inverseops.data import DATASET_REGISTRY, build_dataset
from inverseops.models import MODEL_REGISTRY, build_model


def test_model_registry_has_swinir():
    """SwinIR is registered."""
    assert "swinir" in MODEL_REGISTRY


def test_build_model_unknown_raises():
    """Unknown model name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"model": {"name": "nonexistent"}})


def test_dataset_registry_has_synthetic():
    """Synthetic dataset is registered."""
    assert "synthetic" in DATASET_REGISTRY


def test_build_dataset_unknown_raises():
    """Unknown dataset type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset type"):
        build_dataset({"data": {"noise_source": "nonexistent"}}, split="train")


def test_validate_config_valid():
    """Valid config passes validation."""
    config = {
        "seed": 42,
        "task": "denoise",
        "model": {"name": "swinir", "pretrained": True, "noise_level": 25},
        "data": {
            "train_root": "/tmp",
            "val_root": "/tmp",
            "batch_size": 4,
            "patch_size": 128,
            "sigmas": [25],
        },
        "training": {"epochs": 10, "learning_rate": 1e-4, "loss": "l1"},
        "output_dir": "/tmp/out",
    }
    validate_config(config)  # Should not raise


def test_validate_config_bad_model():
    """Invalid model name raises ValueError."""
    config = {
        "task": "denoise",
        "model": {"name": "invalid_model"},
        "data": {"train_root": "/tmp", "val_root": "/tmp"},
        "training": {},
        "output_dir": "/tmp",
    }
    with pytest.raises(ValueError, match="Unknown model"):
        validate_config(config)


def test_validate_config_bad_task():
    """Invalid task raises ValueError."""
    config = {
        "task": "inpainting",
        "model": {"name": "swinir"},
        "data": {"train_root": "/tmp", "val_root": "/tmp"},
        "training": {},
        "output_dir": "/tmp",
    }
    with pytest.raises(ValueError, match="Unknown task"):
        validate_config(config)

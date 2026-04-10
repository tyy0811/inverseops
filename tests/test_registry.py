"""Tests for model and dataset registries."""

import pytest

from inverseops.config import validate_config
from inverseops.models import MODEL_REGISTRY, build_model


def test_model_registry_has_swinir():
    """SwinIR is registered."""
    assert "swinir" in MODEL_REGISTRY


def test_build_model_unknown_raises():
    """Unknown model name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"model": {"name": "nonexistent"}})


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


def test_validate_config_incompatible_model_task():
    """NAFNet + SR task raises ValueError."""
    config = {
        "task": "sr",
        "model": {"name": "nafnet"},
        "data": {"train_root": "/tmp", "val_root": "/tmp"},
        "training": {},
        "output_dir": "/tmp",
    }
    with pytest.raises(ValueError, match="does not support task"):
        validate_config(config)

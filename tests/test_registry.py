"""Tests for model and dataset registries."""

from inverseops.models import MODEL_REGISTRY, build_model


def test_model_registry_has_swinir():
    """SwinIR is registered."""
    assert "swinir" in MODEL_REGISTRY


def test_build_model_unknown_raises():
    """Unknown model name raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"model": {"name": "nonexistent"}})


from inverseops.data import DATASET_REGISTRY, build_dataset


def test_dataset_registry_has_synthetic():
    """Synthetic dataset is registered."""
    assert "synthetic" in DATASET_REGISTRY


def test_build_dataset_unknown_raises():
    """Unknown dataset type raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown dataset type"):
        build_dataset({"data": {"noise_source": "nonexistent"}}, split="train")

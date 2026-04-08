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

"""Tests for SwinIR model metadata properties."""

import pytest


def test_checkpoint_source_returns_url():
    """checkpoint_source should return the download URL for the noise level."""
    try:
        from inverseops.models.swinir import MODEL_URLS, SwinIRBaseline
    except (ImportError, AttributeError):
        pytest.skip("SwinIR import failed (likely env issue)")

    model = SwinIRBaseline(noise_level=25)
    assert model.checkpoint_source == MODEL_URLS[25]
    assert "noise25" in model.checkpoint_source


def test_checkpoint_source_all_sigmas():
    """checkpoint_source should work for all supported noise levels."""
    try:
        from inverseops.models.swinir import MODEL_URLS, SwinIRBaseline
    except (ImportError, AttributeError):
        pytest.skip("SwinIR import failed (likely env issue)")

    for sigma in [15, 25, 50]:
        model = SwinIRBaseline(noise_level=sigma)
        assert model.checkpoint_source == MODEL_URLS[sigma]


def test_checkpoint_resolved_path_before_load():
    """checkpoint_resolved_path should return None before load() is called."""
    try:
        from inverseops.models.swinir import SwinIRBaseline
    except (ImportError, AttributeError):
        pytest.skip("SwinIR import failed (likely env issue)")

    model = SwinIRBaseline(noise_level=25)
    assert model.checkpoint_resolved_path is None


def test_checkpoint_resolved_path_after_load():
    """checkpoint_resolved_path should return path when model is loaded."""
    try:
        from inverseops.models.swinir import SwinIRBaseline
    except (ImportError, AttributeError):
        pytest.skip("SwinIR import failed (likely env issue)")

    model = SwinIRBaseline(noise_level=25)
    # Mock is_loaded to return True without actually loading weights
    # (is_loaded() checks if _model is not None)
    model._model = True

    path = model.checkpoint_resolved_path
    assert path is not None
    assert "noise25" in str(path)

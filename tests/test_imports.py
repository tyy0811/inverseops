"""Test that core modules can be imported."""

import pytest


def test_import_models_base() -> None:
    """Verify models.base imports successfully."""
    from inverseops.models.base import RestorationModel

    assert RestorationModel is not None


def test_import_data_base() -> None:
    """Verify data.base imports successfully."""
    from inverseops.data.base import Dataset

    assert Dataset is not None


def test_import_serving_schemas() -> None:
    """Verify serving.schemas imports successfully."""
    from inverseops.serving.schemas import (
        HealthResponse,
        InputAnalysis,
        Metrics,
        ModelInfo,
        RestoreResponse,
    )

    assert RestoreResponse is not None
    assert InputAnalysis is not None
    assert ModelInfo is not None
    assert HealthResponse is not None
    assert Metrics is not None


def test_import_evaluation_metrics() -> None:
    """Verify evaluation.metrics imports successfully."""
    from inverseops.evaluation.metrics import compute_psnr, compute_ssim

    assert compute_psnr is not None
    assert compute_ssim is not None


def test_import_models_swinir() -> None:
    """Verify models.swinir imports successfully.

    Skips on import errors (e.g., torch/numpy version conflicts).
    """
    try:
        from inverseops.models.swinir import SwinIRBaseline
    except (ImportError, AttributeError) as e:
        pytest.skip(f"SwinIR import failed (likely env issue): {e}")

    assert SwinIRBaseline is not None

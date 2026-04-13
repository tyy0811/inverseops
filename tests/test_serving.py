"""Tests for the FastAPI serving layer.

Uses create_app() with mocked models to avoid downloading weights in CI.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from inverseops.serving import qc
from inverseops.serving.app import MAX_UPLOAD_BYTES

# --- QC unit tests ---


def test_validate_input_valid():
    """Valid grayscale image passes validation."""
    img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L")
    assert qc.validate_input(img) == []


def test_validate_input_too_small():
    """Image smaller than 8x8 is rejected."""
    img = Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L")
    issues = qc.validate_input(img)
    assert any("too small" in i for i in issues)


def test_validate_input_too_large():
    """Image larger than MAX_RESOLUTION is rejected."""
    img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L")
    img._size = (3000, 3000)
    issues = qc.validate_input(img)
    assert any("exceeds max" in i for i in issues)


def test_validate_output_valid():
    """Clean output passes validation."""
    arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    valid, issues = qc.validate_output(arr)
    assert valid
    assert issues == []


def test_validate_output_nan():
    """Output with NaN is flagged."""
    arr = np.array([0.0, float("nan"), 1.0], dtype=np.float32)
    valid, issues = qc.validate_output(arr)
    assert not valid
    assert any("NaN" in i for i in issues)


def test_validate_output_inf():
    """Output with Inf is flagged."""
    arr = np.array([0.0, float("inf"), 1.0], dtype=np.float32)
    valid, issues = qc.validate_output(arr)
    assert not valid
    assert any("Inf" in i for i in issues)


def test_decide_denoise_good():
    """In-range noise with valid output -> good."""
    assert qc.decide_denoise(25.0, None, True) == "good"


def test_decide_denoise_out_of_range():
    """Noise outside calibrated range -> out_of_range."""
    assert qc.decide_denoise(100.0, None, True) == "out_of_range"


def test_decide_denoise_review_on_invalid_output():
    """Invalid output -> review regardless of noise level."""
    assert qc.decide_denoise(25.0, None, False) == "review"


def test_decide_denoise_uses_estimate_when_no_hint():
    """Falls back to estimated noise level when user doesn't provide one."""
    assert qc.decide_denoise(None, 30.0, True) == "good"
    assert qc.decide_denoise(None, 100.0, True) == "out_of_range"


def test_decide_sr_good_on_valid_output():
    """decide_sr returns 'good' when output is valid."""
    assert qc.decide_sr(output_valid=True) == "good"


def test_decide_sr_review_on_invalid_output():
    """decide_sr returns 'review' when output is invalid."""
    assert qc.decide_sr(output_valid=False) == "review"


def test_decide_sr_has_no_calibrated_range_concept():
    """decide_sr must not take noise_level parameters — SR has no sigma."""
    import inspect

    sig = inspect.signature(qc.decide_sr)
    assert "noise_level" not in sig.parameters
    assert "noise_level_estimated" not in sig.parameters


# --- API integration tests (mocked models) ---


def _make_png_bytes(size: int = 64) -> bytes:
    """Create a valid PNG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 256, (size, size), dtype=np.uint8),
        mode="L",
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _mock_model(raw_output=None):
    """Create a mock that behaves like SwinIRBaseline.

    Args:
        raw_output: Optional raw float32 array for predict_raw.
                    Defaults to zeros (valid output).
    """
    model = MagicMock()
    model.is_loaded.return_value = True

    if raw_output is None:
        raw_output = np.zeros((64, 64), dtype=np.float32)

    model.predict_raw.return_value = raw_output
    model.predict_image.return_value = Image.fromarray(
        np.zeros((64, 64), dtype=np.uint8), mode="L"
    )
    return model


def _mock_models(raw_output=None):
    """Create a dict of mocked models keyed by logical registry name."""
    return {"w2s_denoise_swinir": _mock_model(raw_output)}


@pytest.fixture
def client():
    """FastAPI TestClient with mocked models."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import create_app

    test_app = create_app(models=_mock_models())
    with TestClient(test_app) as c:
        yield c


def test_health_endpoint(client):
    """GET /health returns healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metrics_endpoint(client):
    """GET /metrics returns Prometheus text."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "restore_requests_total" in resp.text


def test_restore_success(client):
    """POST /restore with valid image returns PNG with metadata."""
    png = _make_png_bytes()
    resp = client.post(
        "/restore",
        files={"file": ("test.png", png, "image/png")},
        data={"noise_level": "25"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert "X-Restore-Decision" in resp.headers
    assert resp.headers["X-Restore-Status"] == "completed"


def test_restore_response_task_field_defaults_to_denoise(client):
    """POST /restore sets X-Restore-Task='denoise' on the response."""
    png = _make_png_bytes()
    resp = client.post(
        "/restore",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Restore-Task") == "denoise"


def test_restore_without_noise_level(client):
    """POST /restore works without noise_level (uses default denoise model)."""
    png = _make_png_bytes()
    resp = client.post(
        "/restore",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200


def test_restore_invalid_file(client):
    """POST /restore with non-image data returns 400."""
    resp = client.post(
        "/restore",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_restore_out_of_range(client):
    """POST /restore with noise_level outside calibrated range."""
    png = _make_png_bytes()
    resp = client.post(
        "/restore",
        files={"file": ("test.png", png, "image/png")},
        data={"noise_level": "100"},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Restore-Decision"] == "out_of_range"


def test_restore_inference_failure():
    """POST /restore returns 500 when model.predict_raw raises."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import create_app

    models = _mock_models()
    models["w2s_denoise_swinir"].predict_raw.side_effect = RuntimeError("CUDA OOM")
    test_app = create_app(models=models)
    with TestClient(test_app) as c:
        png = _make_png_bytes()
        resp = c.post(
            "/restore",
            files={"file": ("test.png", png, "image/png")},
        )
    assert resp.status_code == 500


def test_restore_rejects_oversized_upload():
    """POST /restore with >MAX_UPLOAD_BYTES returns 413."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import create_app

    test_app = create_app(models=_mock_models())
    with TestClient(test_app) as c:
        oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1)
        resp = c.post(
            "/restore",
            files={
                "file": ("big.bin", oversized, "image/png")
            },
        )
    assert resp.status_code == 413


def test_restore_detects_nan_in_raw_output():
    """QC catches NaN from model before uint8 clipping masks it."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import create_app

    nan_output = np.full((64, 64), float("nan"), dtype=np.float32)
    models = _mock_models(raw_output=nan_output)
    test_app = create_app(models=models)
    with TestClient(test_app) as c:
        png = _make_png_bytes()
        resp = c.post(
            "/restore",
            files={"file": ("test.png", png, "image/png")},
            data={"noise_level": "25"},
        )
    assert resp.status_code == 200
    assert resp.headers["X-Restore-Decision"] == "review"


def test_registry_has_default_denoise_model():
    """Registry contains the default denoise model with required keys."""
    from inverseops.serving.app import (
        CHECKPOINT_REGISTRY,
        DEFAULT_DENOISE_MODEL,
    )

    assert DEFAULT_DENOISE_MODEL in CHECKPOINT_REGISTRY
    entry = CHECKPOINT_REGISTRY[DEFAULT_DENOISE_MODEL]
    for key in ("path", "task", "dataset", "build_config"):
        assert key in entry, f"registry entry missing required key: {key}"
    assert entry["task"] == "denoise"


def test_registry_has_default_sr_model():
    """Registry contains the default SR model with required keys."""
    from inverseops.serving.app import (
        CHECKPOINT_REGISTRY,
        DEFAULT_SR_MODEL,
    )

    assert DEFAULT_SR_MODEL in CHECKPOINT_REGISTRY
    entry = CHECKPOINT_REGISTRY[DEFAULT_SR_MODEL]
    assert entry["task"] == "sr"
    assert entry["build_config"]["model"]["scale"] == 2


def test_lifespan_raises_on_missing_checkpoint(monkeypatch, tmp_path):
    """Lifespan must raise RuntimeError with the missing file path in the message."""
    from fastapi.testclient import TestClient

    import inverseops.serving.app as app_module

    monkeypatch.setattr(app_module, "CHECKPOINT_ROOT", tmp_path)
    bogus_registry = {
        "bogus_model": {
            "path": "this_file_does_not_exist.pt",
            "task": "denoise",
            "dataset": "w2s",
            "build_config": {},
        }
    }
    monkeypatch.setattr(app_module, "CHECKPOINT_REGISTRY", bogus_registry)
    monkeypatch.setattr(app_module, "DEFAULT_DENOISE_MODEL", "bogus_model")

    test_app = app_module.create_app()  # no models= kwarg; hits the lifespan path
    with pytest.raises(RuntimeError, match="this_file_does_not_exist.pt"):
        with TestClient(test_app):
            pass


def test_estimate_noise_level_returns_positive():
    """estimate_noise_level returns a positive sigma on noisy input."""
    arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    sigma = qc.estimate_noise_level(img)
    assert sigma > 0


def test_estimate_noise_level_low_on_flat():
    """estimate_noise_level returns low sigma on a flat image."""
    arr = np.full((64, 64), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    sigma = qc.estimate_noise_level(img)
    assert sigma < 5.0

"""Tests for the /super_resolve FastAPI endpoint.

Uses create_app() with mocked models keyed by logical registry name.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


def _make_png_bytes(size: int = 64) -> bytes:
    """Create a valid grayscale PNG as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 256, (size, size), dtype=np.uint8),
        mode="L",
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _mock_sr_model():
    """Mock that behaves like the SR model in the registry.

    The sliding_window_sr stitcher calls model(tensor) directly with an
    LR patch shaped (1, 1, h, w) and expects back (1, 1, 2h, 2w).
    """
    model = MagicMock()
    model.is_loaded.return_value = True

    def _forward(x):
        import torch

        b, c, h, w = x.shape
        return torch.zeros(b, c, h * 2, w * 2)

    model.side_effect = _forward
    model.eval.return_value = model
    return model


def _mock_models_multitask():
    """Mock models dict with both denoise and SR entries."""
    from tests.test_serving import _mock_model

    return {
        "w2s_denoise_swinir": _mock_model(),
        "w2s_sr_swinir_2x": _mock_sr_model(),
    }


@pytest.fixture
def sr_client():
    """FastAPI TestClient with mocked multi-task models."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import create_app

    test_app = create_app(models=_mock_models_multitask())
    with TestClient(test_app) as c:
        yield c


def test_super_resolve_success(sr_client):
    """POST /super_resolve with valid image returns 2x-sized PNG."""
    png = _make_png_bytes(size=64)
    resp = sr_client.post(
        "/super_resolve",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers.get("X-Restore-Task") == "sr"
    out = Image.open(io.BytesIO(resp.content))
    assert out.size == (128, 128)


def test_super_resolve_invalid_file(sr_client):
    """POST /super_resolve with non-image data returns 400."""
    resp = sr_client.post(
        "/super_resolve",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_super_resolve_rejects_oversized_upload():
    """POST /super_resolve with >MAX_UPLOAD_BYTES returns 413."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import MAX_UPLOAD_BYTES, create_app

    test_app = create_app(models=_mock_models_multitask())
    with TestClient(test_app) as c:
        oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1)
        resp = c.post(
            "/super_resolve",
            files={"file": ("big.bin", oversized, "image/png")},
        )
    assert resp.status_code == 413


def test_super_resolve_detects_nan_in_raw_output():
    """QC catches NaN from SR model; decision is 'review'."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import create_app

    def _nan_forward(x):
        import torch

        b, c, h, w = x.shape
        return torch.full((b, c, h * 2, w * 2), float("nan"))

    models = _mock_models_multitask()
    models["w2s_sr_swinir_2x"].side_effect = _nan_forward

    test_app = create_app(models=models)
    with TestClient(test_app) as c:
        png = _make_png_bytes(size=64)
        resp = c.post(
            "/super_resolve",
            files={"file": ("test.png", png, "image/png")},
        )
    assert resp.status_code == 200
    assert resp.headers["X-Restore-Decision"] == "review"


def test_super_resolve_output_dimensions_are_2x_input(sr_client):
    """Output image dimensions are exactly 2x input."""
    for size in (64, 128, 256):
        png = _make_png_bytes(size=size)
        resp = sr_client.post(
            "/super_resolve",
            files={"file": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200
        out = Image.open(io.BytesIO(resp.content))
        assert out.size == (size * 2, size * 2), (
            f"Expected {(size * 2, size * 2)}, got {out.size}"
        )


def test_super_resolve_task_header_is_sr(sr_client):
    """X-Restore-Task response header is 'sr' for /super_resolve."""
    png = _make_png_bytes()
    resp = sr_client.post(
        "/super_resolve",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.headers["X-Restore-Task"] == "sr"


def test_super_resolve_503_when_model_not_loaded():
    """POST /super_resolve returns 503 if SR model is not in the mock dict."""
    from fastapi.testclient import TestClient

    from inverseops.serving.app import create_app
    from tests.test_serving import _mock_model

    models = {"w2s_denoise_swinir": _mock_model()}
    test_app = create_app(models=models)
    with TestClient(test_app) as c:
        png = _make_png_bytes()
        resp = c.post(
            "/super_resolve",
            files={"file": ("test.png", png, "image/png")},
        )
    assert resp.status_code == 503

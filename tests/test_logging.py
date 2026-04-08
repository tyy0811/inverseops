"""Tests for structured logging middleware."""

import asyncio
import io
from unittest.mock import MagicMock

import numpy as np
from PIL import Image


def test_request_id_in_response_header():
    """POST /restore returns X-Request-ID header."""
    from httpx import ASGITransport, AsyncClient

    mock_model = MagicMock()
    mock_model.is_loaded.return_value = True
    mock_model.predict_raw.return_value = np.zeros((64, 64), dtype=np.float32)

    from inverseops.serving.app import create_app

    app = create_app(models={25: mock_model})

    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            img = Image.fromarray(
                np.zeros((64, 64), dtype=np.uint8), mode="L"
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            response = await client.post(
                "/restore",
                files={"file": ("test.png", buf, "image/png")},
            )
            assert "x-request-id" in response.headers

    asyncio.run(_test())


def test_request_id_propagated_from_header():
    """Custom X-Request-ID header is echoed back."""
    from httpx import ASGITransport, AsyncClient

    mock_model = MagicMock()
    mock_model.is_loaded.return_value = True

    from inverseops.serving.app import create_app

    app = create_app(models={25: mock_model})
    custom_id = "test-request-12345"

    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get(
                "/health",
                headers={"X-Request-ID": custom_id},
            )
            assert response.headers.get("x-request-id") == custom_id

    asyncio.run(_test())

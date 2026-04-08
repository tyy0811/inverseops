"""Tests for NAFNet model wrapper."""

import numpy as np
import pytest
import torch
from PIL import Image


class TestNAFNetWrapper:
    """Tests for NAFNetBaseline wrapper — mirrors test_swinir_metadata.py patterns."""

    def test_nafnet_init_default(self):
        """NAFNetBaseline initializes with defaults."""
        from inverseops.models.nafnet import NAFNetBaseline

        model = NAFNetBaseline(device="cpu")
        assert not model.is_loaded()
        assert model.device == "cpu"

    def test_nafnet_predict_raw_without_load_raises(self):
        """Calling predict_raw before load() raises RuntimeError."""
        from inverseops.models.nafnet import NAFNetBaseline

        model = NAFNetBaseline(device="cpu")
        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L")
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict_raw(img)

    def test_nafnet_forward_pass_shape(self):
        """NAFNet forward pass preserves spatial dimensions."""
        from inverseops.models._nafnet_arch import NAFNet

        model = NAFNet(img_channel=1, width=32)
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape


class TestNAFNetRegistry:
    """NAFNet is accessible via model registry."""

    def test_nafnet_in_registry(self):
        from inverseops.models import MODEL_REGISTRY

        assert "nafnet" in MODEL_REGISTRY

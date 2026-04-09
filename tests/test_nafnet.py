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

    def test_nafnet_raw_arch_forward_pass_shape(self):
        """Raw RGB NAFNet forward pass preserves spatial dimensions."""
        from inverseops.models._nafnet_arch import NAFNet

        model = NAFNet(img_channel=3, width=32)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape


class TestGrayscaleRGBWrapper:
    """Tests for the grayscale↔RGB wrapper used in training."""

    def test_wrapper_accepts_grayscale_returns_grayscale(self):
        """Wrapper takes [B,1,H,W] input and returns [B,1,H,W] output."""
        from inverseops.models._nafnet_arch import NAFNet
        from inverseops.models.nafnet import _GrayscaleRGBWrapper

        rgb_model = NAFNet(img_channel=3, width=32)
        wrapper = _GrayscaleRGBWrapper(rgb_model)
        wrapper.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = wrapper(x)
        assert y.shape == (1, 1, 64, 64)

    def test_get_trainable_nafnet_grayscale_io(self):
        """get_trainable_nafnet returns model with grayscale in/out."""
        from inverseops.models.nafnet import get_trainable_nafnet

        model = get_trainable_nafnet(pretrained=False, device="cpu")
        model.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 64, 64)


class TestNAFNetRegistry:
    """NAFNet is accessible via model registry."""

    def test_nafnet_in_registry(self):
        from inverseops.models import MODEL_REGISTRY

        assert "nafnet" in MODEL_REGISTRY

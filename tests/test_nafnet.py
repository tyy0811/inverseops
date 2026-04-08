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


class TestNAFNetWeightAdaptation:
    """Tests for RGB→grayscale weight adaptation."""

    def test_adapt_rgb_to_grayscale_intro_conv(self):
        """Input conv weights are averaged across 3 channels to 1."""
        from inverseops.models._nafnet_arch import NAFNet
        from inverseops.models.nafnet import _adapt_rgb_to_grayscale

        rgb_model = NAFNet(img_channel=3, width=32)
        gray_model = NAFNet(img_channel=1, width=32)

        rgb_sd = rgb_model.state_dict()
        adapted = _adapt_rgb_to_grayscale(rgb_sd, gray_model)

        assert adapted["intro.weight"].shape[1] == 1
        assert adapted["ending.weight"].shape[0] == 1

    def test_adapt_noop_when_already_grayscale(self):
        """No changes when weights already match grayscale architecture."""
        from inverseops.models._nafnet_arch import NAFNet
        from inverseops.models.nafnet import _adapt_rgb_to_grayscale

        gray_model = NAFNet(img_channel=1, width=32)
        gray_sd = gray_model.state_dict()
        adapted = _adapt_rgb_to_grayscale(gray_sd, gray_model)

        for k in gray_sd:
            assert adapted[k].shape == gray_sd[k].shape


class TestNAFNetRegistry:
    """NAFNet is accessible via model registry."""

    def test_nafnet_in_registry(self):
        from inverseops.models import MODEL_REGISTRY

        assert "nafnet" in MODEL_REGISTRY

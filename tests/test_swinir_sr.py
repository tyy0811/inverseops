"""Tests for SwinIR super-resolution mode."""

import torch


def test_build_sr_config_returns_valid_kwargs():
    """SR config builds SwinIR model that accepts low-res input."""
    from inverseops.models.swinir import get_trainable_swinir

    model = get_trainable_swinir(
        task="sr", scale=2, pretrained=False, device="cpu"
    )
    # SR model should upscale: input 32x32 -> output 64x64
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)


def test_build_sr_config_4x():
    """4x SR model produces 4x upscaled output."""
    from inverseops.models.swinir import get_trainable_swinir

    model = get_trainable_swinir(
        task="sr", scale=4, pretrained=False, device="cpu"
    )
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 128, 128)


def test_denoise_mode_unchanged():
    """Default denoise mode still works after adding SR support."""
    from inverseops.models.swinir import get_trainable_swinir

    model = get_trainable_swinir(
        task="denoise", pretrained=False, device="cpu"
    )
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)


def test_build_model_threads_task_sr():
    """build_model with task=sr creates an SR model, not denoising."""
    from inverseops.models import build_model

    config = {
        "task": "sr",
        "model": {"name": "swinir", "pretrained": False},
        "data": {"scale": 2},
    }
    model = build_model(config, device="cpu")
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    # SR 2x: 32x32 -> 64x64
    assert y.shape == (1, 1, 64, 64)

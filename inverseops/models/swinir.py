"""SwinIR baseline model wrapper for grayscale image denoising.

This module provides a minimal wrapper around the pretrained SwinIR model
for grayscale Gaussian denoising. The model expects grayscale input and
produces grayscale output.

Pretrained weights are downloaded from the official SwinIR repository:
https://github.com/JingyunLiang/SwinIR

Model architecture: SwinIR-M (medium) trained on DFWB dataset.
Supported noise levels: 15, 25, 50 (sigma).
"""

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from inverseops.models._swinir_arch import SwinIR

# Pretrained model URLs from official SwinIR releases
MODEL_URLS: dict[int, str] = {
    15: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    25: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    50: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth",
}

# Default cache directory for downloaded weights (override via INVERSEOPS_CACHE env var)
_cache_base = Path(
    os.environ.get("INVERSEOPS_CACHE", Path.home() / ".cache" / "inverseops")
)
DEFAULT_CACHE_DIR = _cache_base / "models"


class SwinIRBaseline:
    """Wrapper for pretrained SwinIR grayscale denoising model.

    This class loads a pretrained SwinIR model and provides a simple interface
    for denoising grayscale images. The model is loaded lazily on first use
    or explicitly via the load() method.

    Grayscale handling:
        - Input PIL images are converted to grayscale mode 'L' if needed
        - The model expects single-channel input (in_chans=1)
        - Output is returned as grayscale PIL Image

    Example:
        >>> model = SwinIRBaseline(noise_level=25)
        >>> model.load()
        >>> denoised = model.predict_image(noisy_image)
    """

    def __init__(
        self,
        noise_level: Literal[15, 25, 50] = 25,
        device: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        """Initialize the SwinIR baseline wrapper.

        Args:
            noise_level: Target noise level (sigma). Must be 15, 25, or 50.
                The model trained on sigma=25 works reasonably for nearby levels.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
            cache_dir: Directory to cache downloaded weights.
        """
        if noise_level not in MODEL_URLS:
            raise ValueError(f"noise_level must be 15, 25, or 50, got {noise_level}")

        self.noise_level = noise_level
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: SwinIR | None = None

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    @property
    def checkpoint_source(self) -> str:
        """Return the download URL for this checkpoint."""
        return MODEL_URLS[self.noise_level]

    @property
    def checkpoint_resolved_path(self) -> Path | None:
        """Return the local cached path if weights are loaded."""
        if not self.is_loaded():
            return None
        return self.cache_dir / Path(MODEL_URLS[self.noise_level]).name

    def load(self) -> None:
        """Load the pretrained model weights.

        Downloads weights if not cached locally, then loads the model.
        """
        if self._model is not None:
            return

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Download or load from cache
        url = MODEL_URLS[self.noise_level]
        filename = Path(url).name
        weight_path = self.cache_dir / filename

        if not weight_path.exists():
            self._download_weights(url, weight_path)

        # Build model with grayscale denoising architecture
        # Architecture parameters from official SwinIR test script for gray_dn
        model = SwinIR(
            upscale=1,
            in_chans=1,  # Grayscale input
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )

        # Load weights
        pretrained = torch.load(
            weight_path, map_location=self.device, weights_only=True
        )
        # Official SwinIR weights use 'params' key
        state_dict = pretrained.get("params", pretrained)
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        model.to(self.device)
        self._model = model

    def _download_weights(self, url: str, dest: Path) -> None:
        """Download weights from URL to destination path."""
        import urllib.request

        print(f"Downloading SwinIR weights from {url}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")

    @torch.no_grad()
    def predict_raw(self, image: Image.Image) -> np.ndarray:
        """Denoise a single image, returning the raw float32 output.

        Args:
            image: Input PIL Image (will be converted to grayscale if needed).

        Returns:
            Raw model output as float32 numpy array, shape [H, W],
            range approximately [0, 1]. Not clipped — may contain values
            outside [0, 1] or NaN/Inf if inference is numerically unstable.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if image.mode != "L":
            image = image.convert("L")

        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        output = self._model(tensor)
        return output.squeeze().cpu().numpy()

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> Image.Image:
        """Denoise a single image.

        Args:
            image: Input PIL Image (will be converted to grayscale if needed).

        Returns:
            Denoised PIL Image in grayscale mode 'L'.

        Raises:
            RuntimeError: If model is not loaded.
        """
        output_arr = self.predict_raw(image)
        output_arr = np.clip(output_arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output_arr, mode="L")


# SR pretrained weight URLs (classical variant, grayscale)
SR_MODEL_URLS: dict[int, str] = {
    2: (
        "https://github.com/JingyunLiang/SwinIR/releases/download/"
        "v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    ),
    4: (
        "https://github.com/JingyunLiang/SwinIR/releases/download/"
        "v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    ),
}


def _build_denoise_config() -> dict:
    """Return SwinIR constructor kwargs for grayscale denoising."""
    return dict(
        upscale=1,
        in_chans=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )


def _build_sr_config(scale: int) -> dict:
    """Return SwinIR constructor kwargs for classical SR."""
    return dict(
        upscale=scale,
        in_chans=1,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def get_trainable_swinir(
    task: str = "denoise",
    scale: int = 1,
    noise_level: int = 25,
    pretrained: bool = True,
    device: str | None = None,
    cache_dir: Path | str | None = None,
    **kwargs,
) -> SwinIR:
    """Return trainable SwinIR model.

    Args:
        task: 'denoise' or 'sr'.
        scale: Upscaling factor for SR (2 or 4). Ignored for denoise.
        noise_level: Target noise level for denoising (15, 25, or 50).
        pretrained: If True, load pretrained weights.
        device: Device to place model on. None for auto.
        cache_dir: Directory for cached weights.

    Returns:
        SwinIR nn.Module in training mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    if task == "denoise":
        model_config = _build_denoise_config()
        weight_url = MODEL_URLS.get(noise_level)
        if noise_level not in MODEL_URLS:
            raise ValueError(
                f"noise_level must be 15, 25, or 50, got {noise_level}"
            )
    elif task == "sr":
        model_config = _build_sr_config(scale)
        weight_url = SR_MODEL_URLS.get(scale)
        if scale not in SR_MODEL_URLS:
            raise ValueError(f"SR scale must be 2 or 4, got {scale}")
    else:
        raise ValueError(
            f"Unknown task: {task!r}. Must be 'denoise' or 'sr'."
        )

    model = SwinIR(**model_config)

    if pretrained and weight_url:
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(weight_url).name
        weight_path = cache_dir / filename

        if not weight_path.exists():
            import urllib.request

            print(f"Downloading SwinIR weights from {weight_url}...")
            urllib.request.urlretrieve(weight_url, weight_path)
            print(f"Saved to {weight_path}")

        state_dict = torch.load(
            weight_path, map_location=device, weights_only=True
        )
        state_dict = state_dict.get("params", state_dict)
        model.load_state_dict(state_dict, strict=True)

    model.train()
    model.to(device)
    return model

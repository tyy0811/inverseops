"""NAFNet model wrapper for grayscale image denoising.

Mirrors the SwinIRBaseline interface for drop-in model swapping.
Pretrained weights: NAFNet-width32 trained on SIDD real-world denoising.

Source: https://github.com/megvii-research/NAFNet
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inverseops.models._nafnet_arch import NAFNet

# Pretrained weights URL — mirror to GitHub release for stability
# Original: Google Drive link from NAFNet repo
PRETRAINED_URL = (
    "https://github.com/megvii-research/NAFNet/releases/download/"
    "v0.0.0/NAFNet-SIDD-width32.pth"
)

_cache_base = Path(
    os.environ.get("INVERSEOPS_CACHE", Path.home() / ".cache" / "inverseops")
)
DEFAULT_CACHE_DIR = _cache_base / "models"


class NAFNetBaseline:
    """Wrapper for NAFNet grayscale denoising model.

    Mirrors SwinIRBaseline interface: lazy loading, device auto-detection,
    predict_raw() and predict_image() methods.
    """

    def __init__(
        self,
        device: str | None = None,
        cache_dir: Path | str | None = None,
        width: int = 32,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.width = width

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: NAFNet | None = None

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    @property
    def checkpoint_source(self) -> str:
        """Return the download URL for pretrained weights."""
        return PRETRAINED_URL

    def load(self) -> None:
        """Load pretrained NAFNet weights."""
        if self._model is not None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(PRETRAINED_URL).name
        weight_path = self.cache_dir / filename

        if not weight_path.exists():
            self._download_weights(PRETRAINED_URL, weight_path)

        # NAFNet-width32 for grayscale denoising
        model = NAFNet(img_channel=1, width=self.width)

        pretrained = torch.load(
            weight_path, map_location=self.device, weights_only=True
        )
        # NAFNet checkpoints may use 'params' or 'state_dict' key
        if "params" in pretrained:
            state_dict = pretrained["params"]
        elif "state_dict" in pretrained:
            state_dict = pretrained["state_dict"]
        else:
            state_dict = pretrained

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.device)
        self._model = model

    def _download_weights(self, url: str, dest: Path) -> None:
        """Download weights from URL to destination path."""
        import urllib.request

        print(f"Downloading NAFNet weights from {url}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")

    @torch.no_grad()
    def predict_raw(self, image: Image.Image) -> np.ndarray:
        """Denoise a single image, returning raw float32 output.

        Args:
            image: Input PIL Image (converted to grayscale if needed).

        Returns:
            Raw model output as float32 numpy array, shape [H, W].
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
            image: Input PIL Image (converted to grayscale if needed).

        Returns:
            Denoised PIL Image in grayscale mode 'L'.
        """
        output_arr = self.predict_raw(image)
        output_arr = np.clip(output_arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output_arr, mode="L")


def get_trainable_nafnet(
    pretrained: bool = True,
    device: str | None = None,
    cache_dir: Path | str | None = None,
    width: int = 32,
    **kwargs,
) -> NAFNet:
    """Return trainable NAFNet model for grayscale denoising.

    Args:
        pretrained: If True, load SIDD pretrained weights.
        device: Device to place model on. None for auto.
        cache_dir: Directory for cached weights.
        width: NAFNet width (32 or 64).

    Returns:
        NAFNet nn.Module in training mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    model = NAFNet(img_channel=1, width=width)

    if pretrained:
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(PRETRAINED_URL).name
        weight_path = cache_dir / filename

        if not weight_path.exists():
            import urllib.request
            print(f"Downloading NAFNet weights from {PRETRAINED_URL}...")
            urllib.request.urlretrieve(PRETRAINED_URL, weight_path)
            print(f"Saved to {weight_path}")

        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)

    model.train()
    model.to(device)
    return model

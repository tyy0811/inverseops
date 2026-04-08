# inverseops/data/torch_datasets.py
"""PyTorch Dataset wrappers for training.

Wraps existing dataset classes with training-time transforms.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from inverseops.data.microscopy import MicroscopyDataset


class MicroscopyTrainDataset(Dataset):
    """Torch Dataset wrapper for microscopy training.

    Wraps MicroscopyDataset with:
    - Tensor conversion
    - Random/center patch cropping
    - On-the-fly Gaussian noise generation

    Args:
        base_dataset: Prepared MicroscopyDataset instance.
        patch_size: Size of crops (patches are square).
        sigmas: Noise levels to sample from.
        seed: Random seed for reproducibility.
        training: If True, use random crops and random sigma.
                  If False, use center crops and deterministic behavior.
    """

    def __init__(
        self,
        base_dataset: MicroscopyDataset,
        patch_size: int = 128,
        sigmas: tuple[int, ...] = (15, 25, 50),
        seed: int = 42,
        training: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.sigmas = sigmas
        self.seed = seed
        self.training = training

        # Create RNG for training mode
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        # Load clean image as PIL
        clean_pil = self.base_dataset.load_image(index)
        image_name = self.base_dataset.image_path(index).name

        # Convert to numpy array
        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0

        # Crop
        if self.training:
            clean_crop = self._random_crop(clean_arr)
            sigma = self._rng.choice(self.sigmas)
            noise_seed = None  # Random noise each time
        else:
            clean_crop = self._center_crop(clean_arr)
            sigma = self.sigmas[index % len(self.sigmas)]
            noise_seed = self.seed + index

        # Add Gaussian noise
        noisy_crop = self._add_noise(clean_crop, sigma, seed=noise_seed)

        # Convert to tensors [1, H, W]
        target = torch.from_numpy(clean_crop).unsqueeze(0)
        input_tensor = torch.from_numpy(noisy_crop).unsqueeze(0)

        return {
            "input": input_tensor,
            "target": target,
            "sigma": int(sigma),
            "image_name": image_name,
        }

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        """Random crop to patch_size x patch_size."""
        h, w = arr.shape
        if h < self.patch_size or w < self.patch_size:
            # Pad if needed
            arr = self._pad_to_size(arr, self.patch_size)
            h, w = arr.shape

        top = self._rng.integers(0, h - self.patch_size + 1)
        left = self._rng.integers(0, w - self.patch_size + 1)
        return arr[top : top + self.patch_size, left : left + self.patch_size]

    def _center_crop(self, arr: np.ndarray) -> np.ndarray:
        """Center crop to patch_size x patch_size."""
        h, w = arr.shape
        if h < self.patch_size or w < self.patch_size:
            arr = self._pad_to_size(arr, self.patch_size)
            h, w = arr.shape

        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        return arr[top : top + self.patch_size, left : left + self.patch_size]

    def _pad_to_size(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Pad array to at least size x size using reflection."""
        h, w = arr.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(
                arr,
                ((0, pad_h), (0, pad_w)),
                mode="reflect",
            )
        return arr

    def _add_noise(
        self, arr: np.ndarray, sigma: float, seed: int | None = None
    ) -> np.ndarray:
        """Add Gaussian noise to normalized [0, 1] array."""
        rng = np.random.default_rng(seed)
        # sigma is in [0, 255] scale, normalize to [0, 1]
        sigma_normalized = sigma / 255.0
        noise = rng.normal(0, sigma_normalized, arr.shape).astype(np.float32)
        noisy = arr + noise
        return np.clip(noisy, 0, 1).astype(np.float32)


class RealNoiseTrainDataset(Dataset):
    """Torch Dataset wrapper for real-noise microscopy training.

    Wraps RealNoiseMicroscopyDataset with tensor conversion and patch cropping.
    Returns the same dict format as MicroscopyTrainDataset: {input, target, ...}.
    """

    def __init__(
        self,
        base_dataset,
        patch_size: int = 128,
        seed: int = 42,
        training: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.seed = seed
        self.training = training
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        noisy_pil, clean_pil = self.base_dataset.load_pair(index)
        pair = self.base_dataset.pairs[index]

        noisy_arr = np.array(noisy_pil, dtype=np.float32) / 255.0
        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0

        if self.training:
            noisy_crop, clean_crop = self._random_crop_pair(noisy_arr, clean_arr)
        else:
            noisy_crop, clean_crop = self._center_crop_pair(noisy_arr, clean_arr)

        input_tensor = torch.from_numpy(noisy_crop).unsqueeze(0)
        target = torch.from_numpy(clean_crop).unsqueeze(0)

        return {
            "input": input_tensor,
            "target": target,
            "sigma": -1,  # Unknown sigma for real noise
            "image_name": pair.capture_name,
        }

    def _random_crop_pair(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random crop both arrays at the same position."""
        h, w = a.shape
        ps = self.patch_size
        if h < ps or w < ps:
            a = self._pad_to_size(a, ps)
            b = self._pad_to_size(b, ps)
            h, w = a.shape

        top = self._rng.integers(0, h - ps + 1)
        left = self._rng.integers(0, w - ps + 1)
        return (
            a[top : top + ps, left : left + ps],
            b[top : top + ps, left : left + ps],
        )

    def _center_crop_pair(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center crop both arrays at the same position."""
        h, w = a.shape
        ps = self.patch_size
        if h < ps or w < ps:
            a = self._pad_to_size(a, ps)
            b = self._pad_to_size(b, ps)
            h, w = a.shape

        top = (h - ps) // 2
        left = (w - ps) // 2
        return (
            a[top : top + ps, left : left + ps],
            b[top : top + ps, left : left + ps],
        )

    def _pad_to_size(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Pad array to at least size x size using reflection."""
        h, w = arr.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        return arr


class SRTrainDataset(Dataset):
    """Torch Dataset wrapper for super-resolution training.

    Creates LR/HR pairs by bicubic downsampling clean images.
    LR input has shape [1, patch_size//scale, patch_size//scale],
    HR target has shape [1, patch_size, patch_size].
    """

    def __init__(
        self,
        base_dataset: MicroscopyDataset,
        patch_size: int = 128,
        scale: int = 2,
        seed: int = 42,
        training: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.scale = scale
        self.seed = seed
        self.training = training
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image as PILImage

        clean_pil = self.base_dataset.load_image(index)
        image_name = self.base_dataset.image_path(index).name

        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0

        # Crop HR patch
        if self.training:
            hr_crop = self._random_crop(clean_arr)
        else:
            hr_crop = self._center_crop(clean_arr)

        # Create LR via bicubic downsampling
        hr_pil = PILImage.fromarray(
            (hr_crop * 255).astype(np.uint8), mode="L"
        )
        lr_size = self.patch_size // self.scale
        lr_pil = hr_pil.resize((lr_size, lr_size), PILImage.BICUBIC)
        lr_arr = np.array(lr_pil, dtype=np.float32) / 255.0

        input_tensor = torch.from_numpy(lr_arr).unsqueeze(0)
        target = torch.from_numpy(hr_crop).unsqueeze(0)

        return {
            "input": input_tensor,
            "target": target,
            "sigma": 0,
            "image_name": image_name,
        }

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        ps = self.patch_size
        if h < ps or w < ps:
            arr = self._pad_to_size(arr, ps)
            h, w = arr.shape
        top = self._rng.integers(0, h - ps + 1)
        left = self._rng.integers(0, w - ps + 1)
        return arr[top : top + ps, left : left + ps]

    def _center_crop(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        ps = self.patch_size
        if h < ps or w < ps:
            arr = self._pad_to_size(arr, ps)
            h, w = arr.shape
        top = (h - ps) // 2
        left = (w - ps) // 2
        return arr[top : top + ps, left : left + ps]

    def _pad_to_size(self, arr: np.ndarray, size: int) -> np.ndarray:
        h, w = arr.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        return arr

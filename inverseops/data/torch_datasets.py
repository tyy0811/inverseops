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

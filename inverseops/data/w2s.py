"""W2S (Widefield2SIM) fluorescence microscopy dataset loader.

Loads pre-computed normalized .npy files from the W2S dataset.
No on-the-fly frame averaging needed — the W2S repo ships averages
at each noise level (avg1, avg2, avg4, avg8, avg16, avg400).

Data layout:
    root/avg{N}/{FoV:03d}_{wavelength}.npy    (512x512 float32)
    root/sim/{FoV:03d}_{wavelength}.npy        (1024x1024 float32, HR GT)

120 FoVs (001-120) x 3 wavelengths (0, 1, 2) = 360 files per noise level.

Key design:
- Splits by FoV ID (not by file) to prevent evaluation contamination
- Uses frozen splits from splits.json when available
- Three wavelengths per FoV treated as separate samples (3x data)
- Z-score normalization: mean=154.54, std=66.03 (from W2S paper)

Reference: Qiao et al., "Evaluation and development of deep neural
networks for image super-resolution in optical microscopy" (2021).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

# W2S paper normalization constants
W2S_MEAN = 154.54
W2S_STD = 66.03

# Valid noise levels (pre-computed in W2S repo)
VALID_AVG_LEVELS = (1, 2, 4, 8, 16)
DEFAULT_AVG_LEVELS = [1, 2, 4, 8, 16]


@dataclass
class W2SSample:
    """Metadata for a single W2S training sample."""

    fov_id: int
    wavelength: int
    avg_level: int


class W2SDataset(TorchDataset):
    """PyTorch Dataset for W2S fluorescence microscopy.

    Loads pre-computed .npy files directly — no frame averaging needed.

    Two modes:
    - task="denoise": input=avg{N} (noisy), target=avg400 (clean). 512x512.
    - task="sr": input=avg400 (clean LR), target=sim (HR SIM). 512->1024.

    Args:
        root_dir: Path to W2S data root (parent of avg1/, avg2/, ..., avg400/).
        split: One of "train", "val", "test".
        task: "denoise" or "sr".
        splits_path: Path to splits.json. If None, computes deterministic split.
        avg_levels: Which noise levels to include (default: [1, 2, 4, 8, 16]).
            Ignored when task="sr".
        patch_size: Size of random crops for training (0 for full image).
            For SR, this is the LR patch size (HR crop is 2x).
        seed: Random seed for splits and cropping.
        training: If True, random crops. If False, center crops.
        val_fraction: Fraction for validation if no splits.json.
        test_fraction: Fraction for test if no splits.json.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        task: str = "denoise",
        splits_path: str | Path | None = None,
        avg_levels: list[int] | None = None,
        patch_size: int = 128,
        seed: int = 42,
        training: bool | None = None,
        val_fraction: float = 0.11,
        test_fraction: float = 0.11,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task
        self.splits_path = Path(splits_path) if splits_path else None
        self.avg_levels = avg_levels or DEFAULT_AVG_LEVELS
        self.patch_size = patch_size
        self.seed = seed
        self.training = training if training is not None else (split == "train")
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        self.samples: list[W2SSample] = []
        self._prepared = False

    def prepare(self) -> None:
        """Discover FoVs from .npy files, apply splits, build sample list."""
        # Discover FoVs from any avg level directory
        fov_ids, wavelengths = self._discover_fovs()

        if not fov_ids:
            raise ValueError(
                f"No .npy files found in {self.root_dir}/avg*/ directories"
            )

        # Get FoVs for this split
        split_fov_ids = self._get_split_fov_ids(sorted(fov_ids))

        if self.task == "sr":
            # SR: one sample per (fov, wavelength) — avg400 input, sim target
            for fov_id in sorted(split_fov_ids):
                for wl in sorted(wavelengths):
                    lr_path = self.root_dir / "avg400" / f"{fov_id:03d}_{wl}.npy"
                    hr_path = self.root_dir / "sim" / f"{fov_id:03d}_{wl}.npy"
                    if lr_path.exists() and hr_path.exists():
                        self.samples.append(
                            W2SSample(
                                fov_id=fov_id,
                                wavelength=wl,
                                avg_level=400,  # LR source for SR
                            )
                        )
        else:
            # Denoise: one entry per (fov, wavelength, avg_level)
            for fov_id in sorted(split_fov_ids):
                for wl in sorted(wavelengths):
                    for avg_level in self.avg_levels:
                        npy_path = (
                            self.root_dir / f"avg{avg_level}" / f"{fov_id:03d}_{wl}.npy"
                        )
                        if npy_path.exists():
                            self.samples.append(
                                W2SSample(
                                    fov_id=fov_id,
                                    wavelength=wl,
                                    avg_level=avg_level,
                                )
                            )

        self._prepared = True

        # Sanity print: visible every time a dataset is constructed
        if self.samples:
            sample = self[0]
            inp, tgt = sample["input"], sample["target"]
            print(
                f"=== W2SDataset sanity ({self.split}, {len(self.samples)} samples) ==="
            )
            print(
                f"  input  range=[{inp.min():.3f}, "
                f"{inp.max():.3f}]  "
                f"mean={inp.mean():.3f}"
            )
            print(
                f"  target range=[{tgt.min():.3f}, "
                f"{tgt.max():.3f}]  "
                f"mean={tgt.mean():.3f}"
            )

    def _discover_fovs(self) -> tuple[set[int], set[int]]:
        """Discover FoV IDs and wavelengths from .npy filenames.

        Scans the first available avg directory for files named
        {FoV:03d}_{wavelength}.npy.

        Returns:
            (set of FoV IDs, set of wavelength indices)
        """
        fov_ids: set[int] = set()
        wavelengths: set[int] = set()

        # Scan from any available avg level directory
        for level in list(VALID_AVG_LEVELS) + [400]:
            level_dir = self.root_dir / f"avg{level}"
            if not level_dir.exists():
                continue

            for npy_file in level_dir.glob("*.npy"):
                parts = npy_file.stem.split("_")
                if len(parts) == 2:
                    try:
                        fov_ids.add(int(parts[0]))
                        wavelengths.add(int(parts[1]))
                    except ValueError:
                        continue

            if fov_ids:
                break  # Found files, no need to check other levels

        return fov_ids, wavelengths

    def _get_split_fov_ids(self, all_fov_ids: list[int]) -> list[int]:
        """Get FoV IDs for this split, from splits.json or computed."""
        if self.splits_path and self.splits_path.exists():
            with open(self.splits_path) as f:
                splits = json.load(f)
            w2s_splits = splits.get("w2s", splits)
            if self.split not in w2s_splits:
                raise ValueError(
                    f"Split {self.split!r} not in splits.json. "
                    f"Available: {sorted(w2s_splits.keys())}"
                )
            return w2s_splits[self.split]

        # Compute splits deterministically
        rng = random.Random(self.seed)
        ids = list(all_fov_ids)
        rng.shuffle(ids)

        n = len(ids)
        n_test = max(1, int(n * self.test_fraction))
        n_val = max(1, int(n * self.val_fraction))

        test_ids = ids[:n_test]
        val_ids = ids[n_test : n_test + n_val]
        train_ids = ids[n_test + n_val :]

        split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        if self.split not in split_map:
            raise ValueError(f"Unknown split: {self.split!r}")
        return split_map[self.split]

    def fov_ids(self) -> list[int]:
        """Return sorted list of unique FoV IDs in this split."""
        if not self._prepared:
            raise RuntimeError("Call prepare() first")
        return sorted({s.fov_id for s in self.samples})

    def split_names(self) -> list[str]:
        return ["train", "val", "test"]

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """Reverse Z-score normalization for metric computation.

        The eval harness calls dataset.denormalize() before computing
        PSNR/SSIM, so each dataset owns its own normalization.
        """
        return tensor * W2S_STD + W2S_MEAN

    def __len__(self) -> int:
        if not self._prepared:
            raise RuntimeError("Call prepare() first")
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self._prepared:
            raise RuntimeError("Call prepare() first")

        sample = self.samples[index]

        if self.task == "sr":
            return self._getitem_sr(sample, index)
        return self._getitem_denoise(sample, index)

    def _getitem_denoise(self, sample: W2SSample, index: int) -> dict[str, Any]:
        # Load noisy input (pre-computed average at this noise level)
        noisy_path = (
            self.root_dir
            / f"avg{sample.avg_level}"
            / f"{sample.fov_id:03d}_{sample.wavelength}.npy"
        )
        noisy_arr = np.load(noisy_path).astype(np.float32)

        # Load clean reference (avg400)
        clean_path = (
            self.root_dir / "avg400" / f"{sample.fov_id:03d}_{sample.wavelength}.npy"
        )
        clean_arr = np.load(clean_path).astype(np.float32)

        # W2S .npy files are ALREADY Z-score normalized by the W2S repo
        # (mean≈0, std≈1). Do NOT apply normalization again.
        # denormalize() reverses the repo's normalization back to [0,255].

        # Crop — per-call RNG seeded from (seed, index) so each sample gets a
        # unique but reproducible crop, independent of DataLoader worker fork.
        if self.patch_size > 0:
            if self.training:
                rng = np.random.default_rng((self.seed, index))
                noisy_crop, clean_crop = self._random_crop_pair(
                    noisy_arr, clean_arr, rng
                )
            else:
                noisy_crop, clean_crop = self._center_crop_pair(noisy_arr, clean_arr)
        else:
            noisy_crop, clean_crop = noisy_arr, clean_arr

        # To tensors [1, H, W]
        input_t = torch.from_numpy(noisy_crop).unsqueeze(0).float()
        target_t = torch.from_numpy(clean_crop).unsqueeze(0).float()

        return {
            "input": input_t,
            "target": target_t,
            "noise_level": sample.avg_level,
            "fov_id": sample.fov_id,
            "wavelength": sample.wavelength,
        }

    def _getitem_sr(self, sample: W2SSample, index: int) -> dict[str, Any]:
        # Load LR input (avg400, 512x512)
        lr_path = (
            self.root_dir / "avg400" / f"{sample.fov_id:03d}_{sample.wavelength}.npy"
        )
        lr_arr = np.load(lr_path).astype(np.float32)

        # Load HR target (SIM ground truth, 1024x1024)
        hr_path = self.root_dir / "sim" / f"{sample.fov_id:03d}_{sample.wavelength}.npy"
        hr_arr = np.load(hr_path).astype(np.float32)

        # W2S .npy files are ALREADY Z-score normalized by the W2S repo.

        # Crop — coordinated at 2x scale
        if self.patch_size > 0:
            if self.training:
                rng = np.random.default_rng((self.seed, index))
                lr_crop, hr_crop = self._random_crop_pair_sr(lr_arr, hr_arr, rng)
            else:
                lr_crop, hr_crop = self._center_crop_pair_sr(lr_arr, hr_arr)
        else:
            lr_crop, hr_crop = lr_arr, hr_arr

        input_t = torch.from_numpy(lr_crop).unsqueeze(0).float()
        target_t = torch.from_numpy(hr_crop).unsqueeze(0).float()

        return {
            "input": input_t,
            "target": target_t,
            "noise_level": 400,
            "fov_id": sample.fov_id,
            "wavelength": sample.wavelength,
        }

    def _random_crop_pair(
        self,
        a: np.ndarray,
        b: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random crop both arrays at the same position."""
        h, w = a.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            a = self._pad(a, ps)
            b = self._pad(b, ps)
            h, w = a.shape[:2]
        top = int(rng.integers(0, h - ps + 1))
        left = int(rng.integers(0, w - ps + 1))
        return a[top : top + ps, left : left + ps], b[top : top + ps, left : left + ps]

    def _center_crop_pair(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center crop both arrays."""
        h, w = a.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            a = self._pad(a, ps)
            b = self._pad(b, ps)
            h, w = a.shape[:2]
        top = (h - ps) // 2
        left = (w - ps) // 2
        return a[top : top + ps, left : left + ps], b[top : top + ps, left : left + ps]

    def _random_crop_pair_sr(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random crop LR and HR at coordinated positions (2x scale)."""
        h, w = lr.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            lr = self._pad(lr, ps)
            hr = self._pad(hr, ps * 2)
            h, w = lr.shape[:2]
        top = int(rng.integers(0, h - ps + 1))
        left = int(rng.integers(0, w - ps + 1))
        lr_crop = lr[top : top + ps, left : left + ps]
        hr_crop = hr[top * 2 : (top + ps) * 2, left * 2 : (left + ps) * 2]
        return lr_crop, hr_crop

    def _center_crop_pair_sr(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center crop LR and HR at coordinated positions (2x scale)."""
        h, w = lr.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            lr = self._pad(lr, ps)
            hr = self._pad(hr, ps * 2)
            h, w = lr.shape[:2]
        top = (h - ps) // 2
        left = (w - ps) // 2
        lr_crop = lr[top : top + ps, left : left + ps]
        hr_crop = hr[top * 2 : (top + ps) * 2, left * 2 : (left + ps) * 2]
        return lr_crop, hr_crop

    def _pad(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Pad array to at least size x size using reflection."""
        h, w = arr.shape[:2]
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        return arr

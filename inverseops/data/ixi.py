"""IXI brain MRI dataset loader for medical imaging denoising demo.

Extracts 2D axial slices from 3D T1 NIfTI volumes, applies Rician noise
on-the-fly, and provides subject-level train/val/test splits.

Rician noise model: add Gaussian noise to real and imaginary components
in k-space, then take magnitude. This is the clinically appropriate noise
model for MRI magnitude images.

Key design:
- Subject-level split (not slice-level — slices from same subject are correlated)
- Sigma as fraction of mean signal intensity
- Central axial slices only (skip noisy edge slices)
- Per-subject normalization to [0, 1] using robust percentile (1st/99th)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

# Number of central axial slices to extract per subject
N_CENTRAL_SLICES = 30


class IXIDataset(TorchDataset):
    """PyTorch Dataset for IXI brain MRI denoising.

    Args:
        root_dir: Path containing IXI NIfTI files.
        split: One of "train", "val", "test".
        splits_path: Path to splits.json.
        sigma: Rician noise level as fraction of mean signal intensity.
        patch_size: Crop size (0 for full slice).
        seed: Random seed.
        training: Random crops if True, center crops if False.
        val_fraction: Val fraction if no splits.json.
        test_fraction: Test fraction if no splits.json.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        splits_path: str | Path | None = None,
        sigma: float = 0.10,
        patch_size: int = 128,
        seed: int = 42,
        training: bool | None = None,
        val_fraction: float = 0.10,
        test_fraction: float = 0.10,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.splits_path = Path(splits_path) if splits_path else None
        self.sigma = sigma
        self.patch_size = patch_size
        self.seed = seed
        self.training = training if training is not None else (split == "train")
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        self._slices: list[tuple[int, np.ndarray]] = []  # (subject_id, 2d_array)
        self._prepared = False

    def prepare(self) -> None:
        """Discover subjects, apply splits, extract central axial slices."""
        import nibabel as nib

        # Discover NIfTI files
        nifti_files = sorted(self.root_dir.glob("*.nii.gz")) + sorted(
            self.root_dir.glob("*.nii")
        )
        if not nifti_files:
            raise ValueError(f"No NIfTI files found in {self.root_dir}")

        # Parse subject IDs from filenames (IXI{NNN}-{Site}-...)
        subject_map: dict[int, Path] = {}
        for f in nifti_files:
            name = f.name
            try:
                subject_id = int(name.split("-")[0].replace("IXI", ""))
            except (ValueError, IndexError):
                continue
            subject_map[subject_id] = f

        all_subject_ids = sorted(subject_map.keys())

        # Get split subject IDs
        split_ids = self._get_split_ids(all_subject_ids)

        # Warn if frozen splits reference subjects not on disk
        missing = set(split_ids) - set(all_subject_ids)
        if missing:
            n_missing = len(missing)
            n_expected = len(split_ids)
            print(
                f"WARNING: {n_missing}/{n_expected} split "
                f"subject IDs not found on disk. "
                f"splits.json may assume a different "
                f"subject count than what was downloaded."
                f" Missing IDs (first 10): "
                f"{sorted(missing)[:10]}"
            )

        # Extract central axial slices.
        # NOTE: This loads all volumes eagerly. For the full IXI training set
        # (~460 subjects), this consumes ~3-5 GB RAM. If memory is tight,
        # switch to lazy loading (store paths + slice indices, load in __getitem__).
        for subject_id in sorted(split_ids):
            if subject_id not in subject_map:
                continue
            vol_path = subject_map[subject_id]
            nii_img: nib.Nifti1Image = nib.load(vol_path)  # type: ignore[assignment]
            vol = np.asarray(
                nii_img.dataobj, dtype=np.float32
            )

            # Normalize per-subject using robust percentile
            foreground = vol[vol > 0]
            if len(foreground) == 0:
                continue
            p1, p99 = np.percentile(foreground, [1, 99])
            if p99 > p1:
                vol = (vol - p1) / (p99 - p1)
            vol = np.clip(vol, 0, 1)

            # Extract central axial slices
            n_slices = vol.shape[2]
            center = n_slices // 2
            half = N_CENTRAL_SLICES // 2
            start = max(0, center - half)
            end = min(n_slices, center + half)

            for z in range(start, end):
                axial_slice = vol[:, :, z]
                if axial_slice.mean() > 0.01:  # Skip near-empty slices
                    self._slices.append((subject_id, axial_slice))

        self._prepared = True

        # Sanity print: visible every time a dataset is constructed
        if self._slices:
            mem_bytes = sum(arr.nbytes for _, arr in self._slices)
            mem_mb = mem_bytes / (1024 * 1024)
            sample = self[0]
            inp, tgt = sample["input"], sample["target"]
            n_subjects = len(self.subject_ids())
            print(
                f"=== IXIDataset sanity ({self.split}, "
                f"{len(self._slices)} slices from "
                f"{n_subjects} subjects, "
                f"{mem_mb:.0f} MB) ==="
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

    def _get_split_ids(self, all_ids: list[int]) -> list[int]:
        """Get subject IDs for this split."""
        if self.splits_path and self.splits_path.exists():
            with open(self.splits_path) as f:
                splits = json.load(f)
            ixi_splits = splits.get("ixi", splits)
            return ixi_splits[self.split]

        rng = random.Random(self.seed)
        ids = list(all_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(n * self.test_fraction))
        n_val = max(1, int(n * self.val_fraction))
        split_map = {
            "test": ids[:n_test],
            "val": ids[n_test : n_test + n_val],
            "train": ids[n_test + n_val :],
        }
        return split_map[self.split]

    def subject_ids(self) -> list[int]:
        """Return sorted unique subject IDs in this split."""
        if not self._prepared:
            raise RuntimeError("Call prepare() first")
        return sorted({sid for sid, _ in self._slices})

    def split_names(self) -> list[str]:
        return ["train", "val", "test"]

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """IXI data is normalized to [0, 1] -- denormalize is identity.

        Provided so the eval harness can call dataset.denormalize()
        uniformly across all dataset types.
        """
        return tensor

    def __len__(self) -> int:
        if not self._prepared:
            raise RuntimeError("Call prepare() first")
        return len(self._slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_id, stored_arr = self._slices[index]
        clean_arr = stored_arr.copy()  # don't mutate the stored slice

        # Add Rician noise with per-sample deterministic seeding.
        # Using (seed, index) avoids the classic PyTorch bug where
        # multi-worker DataLoaders share RNG state across workers.
        sample_rng = np.random.default_rng((self.seed, index))
        noisy_arr = self._add_rician_noise(clean_arr, self.sigma, sample_rng)

        # Crop — uses sample_rng (not self._rng) so crops are deterministic
        # per sample even with multi-worker DataLoaders.
        if self.patch_size > 0:
            if self.training:
                noisy_arr, clean_arr = self._random_crop(
                    noisy_arr, clean_arr, sample_rng
                )
            else:
                noisy_arr, clean_arr = self._center_crop(noisy_arr, clean_arr)

        input_t = torch.from_numpy(noisy_arr).unsqueeze(0).float()
        target_t = torch.from_numpy(clean_arr).unsqueeze(0).float()

        return {
            "input": input_t,
            "target": target_t,
            "sigma": self.sigma,
            "subject_id": subject_id,
        }

    @staticmethod
    def _add_rician_noise(
        arr: np.ndarray, sigma: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Add Rician noise to a magnitude image.

        Rician noise model: noisy = |clean + n_real + j*n_imag|
        where n_real, n_imag ~ N(0, sigma).

        Uses per-sample RNG to avoid shared state across DataLoader workers.
        """
        n_real = rng.normal(0, sigma, arr.shape).astype(np.float32)
        n_imag = rng.normal(0, sigma, arr.shape).astype(np.float32)
        noisy = np.sqrt((arr + n_real) ** 2 + n_imag**2)
        return np.clip(noisy, 0, 1).astype(np.float32)

    def _random_crop(
        self,
        a: np.ndarray,
        b: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = a.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            a = np.pad(a, ((0, max(0, ps - h)), (0, max(0, ps - w))), mode="reflect")
            b = np.pad(b, ((0, max(0, ps - h)), (0, max(0, ps - w))), mode="reflect")
            h, w = a.shape[:2]
        top = rng.integers(0, h - ps + 1)
        left = rng.integers(0, w - ps + 1)
        return a[top : top + ps, left : left + ps], b[top : top + ps, left : left + ps]

    def _center_crop(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = a.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            a = np.pad(a, ((0, max(0, ps - h)), (0, max(0, ps - w))), mode="reflect")
            b = np.pad(b, ((0, max(0, ps - h)), (0, max(0, ps - w))), mode="reflect")
            h, w = a.shape[:2]
        top = (h - ps) // 2
        left = (w - ps) // 2
        return a[top : top + ps, left : left + ps], b[top : top + ps, left : left + ps]

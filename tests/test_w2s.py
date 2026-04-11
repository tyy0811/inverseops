"""Tests for W2S dataset loader.

Uses synthetic .npy fixtures matching the real W2S layout:
  root/avg{N}/{FoV:03d}_{wavelength}.npy
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch


def _create_w2s_fixture(
    root: Path,
    n_fovs: int = 6,
    n_wavelengths: int = 3,
    img_size: int = 64,
    avg_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 400),
) -> dict:
    """Create synthetic .npy files matching W2S normalized layout.

    Structure:
        root/avg1/001_0.npy, 001_1.npy, 001_2.npy, ...
        root/avg2/001_0.npy, ...
        ...
        root/avg400/001_0.npy, ...  (clean reference)
        root/sim/001_0.npy, ...     (HR ground truth, 2x resolution)
    """
    # W2S .npy files are ALREADY Z-score normalized (mean≈0, std≈1).
    # Fixture must match this — values near 0, not near 154.54.
    for level in avg_levels:
        level_dir = root / f"avg{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        for fov in range(1, n_fovs + 1):
            for wl in range(n_wavelengths):
                # Noisier levels have higher variance (in normalized space)
                noise_scale = 0.3 / max(level, 1)
                rng = np.random.default_rng(fov * 100 + wl * 10 + level)
                arr = rng.normal(0, 1.0, (img_size, img_size)).astype(np.float32)
                arr += rng.normal(0, noise_scale, (img_size, img_size)).astype(
                    np.float32
                )
                np.save(level_dir / f"{fov:03d}_{wl}.npy", arr)

    # SIM HR ground truth (2x resolution, also pre-normalized)
    sim_dir = root / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    for fov in range(1, n_fovs + 1):
        for wl in range(n_wavelengths):
            rng = np.random.default_rng(fov * 100 + wl * 10 + 999)
            arr = rng.normal(0, 1.0, (img_size * 2, img_size * 2)).astype(np.float32)
            np.save(sim_dir / f"{fov:03d}_{wl}.npy", arr)

    return {
        "n_fovs": n_fovs,
        "n_wavelengths": n_wavelengths,
        "total_per_level": n_fovs * n_wavelengths,
    }


class TestW2SDataset:
    def test_discovers_fovs(self):
        """Dataset discovers all FoVs from .npy files."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=4, n_wavelengths=3)

            ds = W2SDataset(root_dir=root, split="train")
            ds.prepare()
            assert len(ds) > 0

    def test_split_by_fov_not_file(self):
        """Train and test splits contain different FoVs — no leakage."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=10, n_wavelengths=1)

            train_ds = W2SDataset(root_dir=root, split="train")
            train_ds.prepare()
            test_ds = W2SDataset(root_dir=root, split="test")
            test_ds.prepare()

            train_fovs = set(train_ds.fov_ids())
            test_fovs = set(test_ds.fov_ids())
            assert train_fovs.isdisjoint(test_fovs), (
                f"FoV leakage: {train_fovs & test_fovs}"
            )

    def test_frozen_splits_from_json(self):
        """Dataset uses splits.json when it exists."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=10, n_wavelengths=1)

            splits = {
                "w2s": {
                    "train": [1, 2, 3, 4, 5, 6, 7, 8],
                    "val": [9],
                    "test": [10],
                }
            }
            splits_path = root / "splits.json"
            with open(splits_path, "w") as f:
                json.dump(splits, f)

            ds = W2SDataset(root_dir=root, split="test", splits_path=splits_path)
            ds.prepare()
            assert ds.fov_ids() == [10]

    def test_returns_correct_dict_keys(self):
        """Output dict matches training pipeline contract."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=4)

            ds = W2SDataset(root_dir=root, split="train", patch_size=32)
            ds.prepare()
            item = ds[0]

            assert set(item.keys()) == {
                "input",
                "target",
                "noise_level",
                "fov_id",
                "wavelength",
            }
            assert item["input"].shape == (1, 32, 32)
            assert item["target"].shape == (1, 32, 32)
            assert item["input"].dtype == torch.float32

    def test_different_noise_levels(self):
        """Dataset includes entries for each requested avg level."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=4, n_wavelengths=1)

            ds = W2SDataset(root_dir=root, split="train", avg_levels=[1, 8])
            ds.prepare()
            noise_levels = {ds[i]["noise_level"] for i in range(len(ds))}
            assert 1 in noise_levels
            assert 8 in noise_levels

    def test_no_fov_overlap_across_all_splits(self):
        """Train + val + test FoVs are fully disjoint and cover all data."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=10, n_wavelengths=1)

            all_fovs: set[int] = set()
            for split in ("train", "val", "test"):
                ds = W2SDataset(root_dir=root, split=split)
                ds.prepare()
                fovs = set(ds.fov_ids())
                assert fovs.isdisjoint(all_fovs), f"Overlap in {split}"
                all_fovs.update(fovs)
            assert len(all_fovs) == 10

    def test_wavelengths_treated_as_separate_samples(self):
        """3 wavelengths per FoV means 3x samples per noise level."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Use frozen splits so we know exactly which FoVs land in train
            _create_w2s_fixture(root, n_fovs=4, n_wavelengths=3)

            splits = {"w2s": {"train": [1, 2], "val": [3], "test": [4]}}
            splits_path = root / "splits.json"
            with open(splits_path, "w") as f:
                json.dump(splits, f)

            ds = W2SDataset(
                root_dir=root,
                split="train",
                avg_levels=[1],
                splits_path=splits_path,
            )
            ds.prepare()
            # 2 FoVs x 3 wavelengths x 1 noise level = 6 samples
            assert len(ds) == 6
            wavelengths = {ds[i]["wavelength"] for i in range(len(ds))}
            assert wavelengths == {0, 1, 2}

    def test_data_is_prenormalized(self):
        """Data from .npy is already Z-score normalized (mean≈0, std≈1)."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=4)

            ds = W2SDataset(root_dir=root, split="train", patch_size=0)
            ds.prepare()
            item = ds[0]
            # Pre-normalized data: values centered near 0
            assert item["target"].mean().abs() < 3.0
            # Denormalized data: values in original intensity space
            denorm = ds.denormalize(item["target"])
            assert denorm.mean() > 100  # Should be near W2S_MEAN=154.54

    def test_denormalize_roundtrip(self):
        """denormalize(normalize(x)) recovers original values."""
        from inverseops.data.w2s import W2S_MEAN, W2S_STD, W2SDataset

        original = torch.tensor([154.54, 200.0, 100.0])
        normalized = (original - W2S_MEAN) / W2S_STD
        recovered = W2SDataset.denormalize(normalized)
        assert torch.allclose(original, recovered, atol=0.01)


class TestW2SDatasetSR:
    def test_sr_returns_2x_target(self):
        """SR mode: input is LR (patch_size), target is HR (2x patch_size)."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=4)

            ds = W2SDataset(root_dir=root, split="train", task="sr", patch_size=32)
            ds.prepare()
            item = ds[0]

            assert item["input"].shape == (1, 32, 32)
            assert item["target"].shape == (1, 64, 64)

    def test_sr_sample_count(self):
        """SR has one sample per (FoV, wavelength) — no noise-level axis."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=4, n_wavelengths=3)

            splits = {"w2s": {"train": [1, 2], "val": [3], "test": [4]}}
            splits_path = root / "splits.json"
            with open(splits_path, "w") as f:
                json.dump(splits, f)

            ds = W2SDataset(
                root_dir=root,
                split="train",
                task="sr",
                splits_path=splits_path,
            )
            ds.prepare()
            # 2 FoVs x 3 wavelengths x 1 (no noise levels) = 6
            assert len(ds) == 6

    def test_sr_noise_level_is_400(self):
        """SR samples report noise_level=400 (avg400 is the LR source)."""
        from inverseops.data.w2s import W2SDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_w2s_fixture(root, n_fovs=4)

            ds = W2SDataset(root_dir=root, split="train", task="sr", patch_size=32)
            ds.prepare()
            assert ds[0]["noise_level"] == 400


class TestW2SRegistry:
    def test_w2s_in_registry(self):
        from inverseops.data import DATASET_REGISTRY

        assert "w2s" in DATASET_REGISTRY

"""Tests for real-noise microscopy dataset loader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _create_fmd_structure(root: Path, num_specimens: int = 4, num_captures: int = 3):
    """Create a minimal FMD-like directory structure for testing."""
    for specimen_id in range(1, num_specimens + 1):
        # Raw noisy captures
        raw_dir = root / "raw" / str(specimen_id)
        raw_dir.mkdir(parents=True)
        for cap in range(num_captures):
            img = Image.fromarray(
                np.random.randint(50, 200, (64, 64), dtype=np.uint8), mode="L"
            )
            img.save(raw_dir / f"capture_{cap:03d}.png")

        # Ground truth (averaged)
        gt_dir = root / "gt" / str(specimen_id)
        gt_dir.mkdir(parents=True)
        img = Image.fromarray(
            np.full((64, 64), 128, dtype=np.uint8), mode="L"
        )
        img.save(gt_dir / "avg50.png")


class TestRealNoiseMicroscopyDataset:

    def test_discovers_pairs(self):
        """Loader finds noisy/clean pairs from FMD structure."""
        from inverseops.data.microscopy_real import RealNoiseMicroscopyDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fmd_structure(root, num_specimens=4, num_captures=3)

            ds = RealNoiseMicroscopyDataset(root_dir=root, split="train")
            ds.prepare()
            assert len(ds) > 0

    def test_split_by_specimen(self):
        """Train and test splits contain different specimens."""
        from inverseops.data.microscopy_real import RealNoiseMicroscopyDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fmd_structure(root, num_specimens=10, num_captures=3)

            train_ds = RealNoiseMicroscopyDataset(root_dir=root, split="train")
            train_ds.prepare()
            test_ds = RealNoiseMicroscopyDataset(root_dir=root, split="test")
            test_ds.prepare()

            train_specimens = {p.specimen_id for p in train_ds.pairs}
            test_specimens = {p.specimen_id for p in test_ds.pairs}
            assert train_specimens.isdisjoint(test_specimens)

    def test_deterministic_split(self):
        """Same seed produces same split."""
        from inverseops.data.microscopy_real import RealNoiseMicroscopyDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fmd_structure(root, num_specimens=10, num_captures=3)

            ds1 = RealNoiseMicroscopyDataset(root_dir=root, split="train", seed=42)
            ds1.prepare()
            ds2 = RealNoiseMicroscopyDataset(root_dir=root, split="train", seed=42)
            ds2.prepare()
            assert len(ds1) == len(ds2)

    def test_ground_truth_is_averaged(self):
        """Each pair uses avg50.png as target, not another noisy capture."""
        from inverseops.data.microscopy_real import RealNoiseMicroscopyDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fmd_structure(root, num_specimens=4, num_captures=3)

            ds = RealNoiseMicroscopyDataset(root_dir=root, split="train")
            ds.prepare()
            for pair in ds.pairs:
                assert "avg50" in pair.target_path.name or "gt" in str(
                    pair.target_path
                )


class TestRealNoiseTrainDataset:

    def test_returns_correct_dict_keys(self):
        """Output dict has same keys as MicroscopyTrainDataset."""
        from inverseops.data.microscopy_real import RealNoiseMicroscopyDataset
        from inverseops.data.torch_datasets import RealNoiseTrainDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fmd_structure(root, num_specimens=4, num_captures=3)

            base = RealNoiseMicroscopyDataset(root_dir=root, split="train")
            base.prepare()
            ds = RealNoiseTrainDataset(base_dataset=base, patch_size=32)

            item = ds[0]
            assert set(item.keys()) == {"input", "target", "sigma", "image_name"}

    def test_tensor_format_matches_synthetic(self):
        """Output tensors have same dtype, shape pattern, value range."""
        from inverseops.data.microscopy_real import RealNoiseMicroscopyDataset
        from inverseops.data.torch_datasets import RealNoiseTrainDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fmd_structure(root, num_specimens=4, num_captures=3)

            base = RealNoiseMicroscopyDataset(root_dir=root, split="train")
            base.prepare()
            ds = RealNoiseTrainDataset(base_dataset=base, patch_size=32)

            item = ds[0]
            assert item["input"].shape == (1, 32, 32)
            assert item["target"].shape == (1, 32, 32)
            assert item["input"].dtype == torch.float32
            assert item["target"].dtype == torch.float32
            assert 0.0 <= item["input"].min() <= item["input"].max() <= 1.0

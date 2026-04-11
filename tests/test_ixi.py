"""Tests for IXI brain MRI dataset loader."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch


def _create_ixi_structure(root: Path, n_subjects: int = 6) -> None:
    """Create minimal IXI-like NIfTI files for testing."""
    import nibabel as nib

    sites = ["Guys", "HH", "IOP"]
    for i in range(1, n_subjects + 1):
        site = sites[i % len(sites)]
        vol = np.random.rand(64, 64, 30).astype(np.float32) * 1000
        img = nib.Nifti1Image(vol, affine=np.eye(4))
        fname = root / f"IXI{i:03d}-{site}-{1000 + i}-T1.nii.gz"
        nib.save(img, fname)


class TestIXIDataset:
    def test_discovers_subjects(self):
        from inverseops.data.ixi import IXIDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=6)
            ds = IXIDataset(root_dir=root, split="train")
            ds.prepare()
            assert len(ds) > 0

    def test_split_by_subject_not_slice(self):
        from inverseops.data.ixi import IXIDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=10)

            train_ds = IXIDataset(root_dir=root, split="train")
            train_ds.prepare()
            test_ds = IXIDataset(root_dir=root, split="test")
            test_ds.prepare()

            train_subjects = set(train_ds.subject_ids())
            test_subjects = set(test_ds.subject_ids())
            assert train_subjects.isdisjoint(test_subjects)

    def test_rician_noise_applied(self):
        from inverseops.data.ixi import IXIDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=4)

            ds = IXIDataset(root_dir=root, split="train", sigma=0.10)
            ds.prepare()
            item = ds[0]

            # Input (noisy) should differ from target (clean)
            assert not torch.allclose(item["input"], item["target"])

    def test_returns_correct_dict_keys(self):
        from inverseops.data.ixi import IXIDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=4)

            ds = IXIDataset(root_dir=root, split="train", patch_size=32)
            ds.prepare()
            item = ds[0]

            assert set(item.keys()) == {"input", "target", "sigma", "subject_id"}
            assert item["input"].shape == (1, 32, 32)
            assert item["target"].shape == (1, 32, 32)
            assert item["input"].dtype == torch.float32

    def test_frozen_splits(self):
        from inverseops.data.ixi import IXIDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=10)

            splits = {
                "ixi": {"train": [1, 2, 3, 4, 5, 6, 7, 8], "val": [9], "test": [10]}
            }
            sp = root / "splits.json"
            with open(sp, "w") as f:
                json.dump(splits, f)

            ds = IXIDataset(root_dir=root, split="test", splits_path=sp)
            ds.prepare()
            assert ds.subject_ids() == [10]

    def test_denormalize_is_identity(self):
        from inverseops.data.ixi import IXIDataset

        t = torch.rand(1, 64, 64)
        assert torch.equal(IXIDataset.denormalize(t), t)

    def test_data_range_zero_to_one(self):
        from inverseops.data.ixi import IXIDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=4)

            ds = IXIDataset(root_dir=root, split="train", patch_size=0)
            ds.prepare()
            item = ds[0]

            # Target (clean) should be in [0, 1] after percentile normalization
            assert item["target"].min() >= 0.0
            assert item["target"].max() <= 1.0


class TestIXIRegistry:
    def test_ixi_in_registry(self):
        from inverseops.data import DATASET_REGISTRY

        assert "ixi" in DATASET_REGISTRY

    def test_build_dataset_forwards_sigma(self):
        """Verify sigma flows from config through build_dataset to IXIDataset."""
        from inverseops.data import build_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=4)

            config = {
                "task": "denoise",
                "data": {
                    "dataset": "ixi",
                    "train_root": str(root),
                    "val_root": str(root),
                    "sigma": 0.20,
                    "patch_size": 32,
                },
            }
            ds = build_dataset(config, split="train")
            assert ds.sigma == 0.20


class TestIXIMissingSplitWarning:
    def test_warns_on_missing_subjects(self, capsys):
        """Frozen splits referencing absent subject IDs should print a warning."""
        from inverseops.data.ixi import IXIDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_ixi_structure(root, n_subjects=4)  # IDs 1-4

            # splits.json references IDs 1-10, but only 1-4 exist on disk
            splits = {
                "ixi": {"train": [1, 2, 3, 4, 5, 6, 7, 8], "val": [9], "test": [10]}
            }
            sp = root / "splits.json"
            with open(sp, "w") as f:
                json.dump(splits, f)

            ds = IXIDataset(root_dir=root, split="train", splits_path=sp)
            ds.prepare()
            captured = capsys.readouterr()
            assert "not found on disk" in captured.out

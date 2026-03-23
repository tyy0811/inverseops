"""Day 2 tests for data loading and degradation utilities."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from inverseops.data.degradations import (
    SUPPORTED_SIGMAS,
    add_gaussian_noise,
    generate_noisy_variants,
)
from inverseops.data.microscopy import MicroscopyDataset
from inverseops.data.transforms import center_crop, normalize_to_uint8, to_grayscale


def create_test_image(
    size: tuple[int, int] = (64, 64), value: int = 128
) -> Image.Image:
    """Create a simple grayscale test image."""
    arr = np.full(size, value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def create_test_dataset(tmp_dir: Path, num_images: int = 10) -> None:
    """Create test images in a temporary directory."""
    for i in range(num_images):
        img = create_test_image(value=100 + i)
        img.save(tmp_dir / f"image_{i:03d}.png")


class TestMicroscopyDataset:
    """Tests for MicroscopyDataset."""

    def test_finds_image_files(self) -> None:
        """Dataset discovers image files in directory."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_test_dataset(tmp_path, num_images=5)

            dataset = MicroscopyDataset(root_dir=tmp_path, split="train")
            dataset.prepare()

            assert len(dataset) > 0

    def test_split_deterministic(self) -> None:
        """Same seed produces same split."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_test_dataset(tmp_path, num_images=20)

            ds1 = MicroscopyDataset(root_dir=tmp_path, split="train", seed=42)
            ds1.prepare()
            paths1 = [ds1.image_path(i) for i in range(len(ds1))]

            ds2 = MicroscopyDataset(root_dir=tmp_path, split="train", seed=42)
            ds2.prepare()
            paths2 = [ds2.image_path(i) for i in range(len(ds2))]

            assert paths1 == paths2

    def test_splits_disjoint_and_complete(self) -> None:
        """Train/val/test splits are disjoint and cover all files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            num_images = 20
            create_test_dataset(tmp_path, num_images=num_images)

            train_ds = MicroscopyDataset(
                root_dir=tmp_path, split="train", val_fraction=0.2, test_fraction=0.2
            )
            val_ds = MicroscopyDataset(
                root_dir=tmp_path, split="val", val_fraction=0.2, test_fraction=0.2
            )
            test_ds = MicroscopyDataset(
                root_dir=tmp_path, split="test", val_fraction=0.2, test_fraction=0.2
            )

            train_ds.prepare()
            val_ds.prepare()
            test_ds.prepare()

            train_paths = {train_ds.image_path(i) for i in range(len(train_ds))}
            val_paths = {val_ds.image_path(i) for i in range(len(val_ds))}
            test_paths = {test_ds.image_path(i) for i in range(len(test_ds))}

            # Disjoint
            assert train_paths.isdisjoint(val_paths)
            assert train_paths.isdisjoint(test_paths)
            assert val_paths.isdisjoint(test_paths)

            # Complete
            all_paths = train_paths | val_paths | test_paths
            assert len(all_paths) == num_images

    def test_load_image_grayscale(self) -> None:
        """Loaded images are grayscale."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_test_dataset(tmp_path, num_images=3)

            dataset = MicroscopyDataset(root_dir=tmp_path, split="train")
            dataset.prepare()

            img = dataset.load_image(0)
            assert img.mode == "L"

    def test_split_names(self) -> None:
        """split_names returns expected values."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = MicroscopyDataset(root_dir=tmp)
            assert dataset.split_names() == ["train", "val", "test"]


class TestDegradations:
    """Tests for degradation utilities."""

    def test_gaussian_noise_deterministic(self) -> None:
        """Same seed produces same noise."""
        img = create_test_image()

        noisy1 = add_gaussian_noise(img, sigma=25, seed=42)
        noisy2 = add_gaussian_noise(img, sigma=25, seed=42)

        arr1 = np.array(noisy1)
        arr2 = np.array(noisy2)
        assert np.array_equal(arr1, arr2)

    def test_gaussian_noise_different_seeds(self) -> None:
        """Different seeds produce different noise."""
        img = create_test_image()

        noisy1 = add_gaussian_noise(img, sigma=25, seed=42)
        noisy2 = add_gaussian_noise(img, sigma=25, seed=123)

        arr1 = np.array(noisy1)
        arr2 = np.array(noisy2)
        assert not np.array_equal(arr1, arr2)

    def test_generate_noisy_variants(self) -> None:
        """Generates variants for all supported sigmas."""
        img = create_test_image()
        variants = generate_noisy_variants(img, sigmas=SUPPORTED_SIGMAS, seed=42)

        assert set(variants.keys()) == set(SUPPORTED_SIGMAS)
        for sigma in SUPPORTED_SIGMAS:
            assert variants[sigma].mode == "L"

    def test_noisy_variants_deterministic(self) -> None:
        """Same seed produces same variants."""
        img = create_test_image()

        v1 = generate_noisy_variants(img, seed=42)
        v2 = generate_noisy_variants(img, seed=42)

        for sigma in SUPPORTED_SIGMAS:
            arr1 = np.array(v1[sigma])
            arr2 = np.array(v2[sigma])
            assert np.array_equal(arr1, arr2)

    def test_supported_sigmas(self) -> None:
        """SUPPORTED_SIGMAS contains expected values."""
        assert SUPPORTED_SIGMAS == (15, 25, 50)


class TestTransforms:
    """Tests for transform utilities."""

    def test_to_grayscale(self) -> None:
        """to_grayscale converts to mode L."""
        rgb = Image.new("RGB", (32, 32), color=(100, 150, 200))
        gray = to_grayscale(rgb)
        assert gray.mode == "L"

    def test_center_crop(self) -> None:
        """center_crop produces correct size."""
        img = create_test_image(size=(100, 100))
        cropped = center_crop(img, (50, 50))
        assert cropped.size == (50, 50)

    def test_normalize_to_uint8(self) -> None:
        """normalize_to_uint8 returns mode L."""
        img = Image.new("RGB", (32, 32))
        normalized = normalize_to_uint8(img)
        assert normalized.mode == "L"


class TestSamplePipeline:
    """Tests for sample generation pipeline."""

    def test_save_samples(self) -> None:
        """Sample pipeline can save output images."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            artifacts_dir = tmp_path / "artifacts"
            data_dir.mkdir()
            artifacts_dir.mkdir()

            # Create test data
            create_test_dataset(data_dir, num_images=3)

            # Load and process
            dataset = MicroscopyDataset(root_dir=data_dir, split="train")
            dataset.prepare()
            clean = dataset.load_image(0)
            variants = generate_noisy_variants(clean, seed=42)

            # Save outputs
            clean.save(artifacts_dir / "clean.png")
            for sigma, noisy in variants.items():
                noisy.save(artifacts_dir / f"noisy_sigma_{sigma}.png")

            # Verify files exist
            assert (artifacts_dir / "clean.png").exists()
            for sigma in SUPPORTED_SIGMAS:
                assert (artifacts_dir / f"noisy_sigma_{sigma}.png").exists()

    def test_save_samples_with_metadata(self) -> None:
        """Sample pipeline can save metadata.json."""
        import json

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            artifacts_dir = tmp_path / "artifacts"
            data_dir.mkdir()
            artifacts_dir.mkdir()

            # Create test data
            create_test_dataset(data_dir, num_images=3)

            # Load and process
            dataset = MicroscopyDataset(root_dir=data_dir, split="train")
            dataset.prepare()
            clean = dataset.load_image(0)
            source_path = dataset.image_path(0)
            variants = generate_noisy_variants(clean, seed=42)

            # Save outputs and metadata
            output_files = {}
            clean_path = artifacts_dir / "clean.png"
            clean.save(clean_path)
            output_files["clean"] = str(clean_path)

            for sigma, noisy in variants.items():
                noisy_path = artifacts_dir / f"noisy_sigma_{sigma}.png"
                noisy.save(noisy_path)
                output_files[f"noisy_sigma_{sigma}"] = str(noisy_path)

            metadata = {
                "source_image": str(source_path.name),
                "seed": 42,
                "sigma_values": list(SUPPORTED_SIGMAS),
                "output_files": output_files,
            }

            metadata_path = artifacts_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Verify metadata exists and is valid JSON
            assert metadata_path.exists()
            with open(metadata_path) as f:
                loaded = json.load(f)
                assert loaded["seed"] == 42
                assert loaded["sigma_values"] == list(SUPPORTED_SIGMAS)
                assert "clean" in loaded["output_files"]


class TestDatasetRobustness:
    """Tests for dataset robustness and edge cases."""

    def test_recursive_discovery_nested_folders(self) -> None:
        """Dataset discovers images in nested subdirectories."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create nested structure
            (tmp_path / "subdir1").mkdir()
            (tmp_path / "subdir2" / "nested").mkdir(parents=True)

            # Add images at different levels
            create_test_image().save(tmp_path / "root.png")
            create_test_image().save(tmp_path / "subdir1" / "img1.png")
            create_test_image().save(tmp_path / "subdir2" / "nested" / "img2.png")

            dataset = MicroscopyDataset(root_dir=tmp_path, split="train")
            dataset.prepare()

            # Should find all 3 images
            assert len(dataset._all_files) == 3

    def test_tiny_dataset_splits_disjoint(self) -> None:
        """Very small datasets produce disjoint splits."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create only 3 images
            create_test_dataset(tmp_path, num_images=3)

            train_ds = MicroscopyDataset(
                root_dir=tmp_path, split="train", val_fraction=0.3, test_fraction=0.3
            )
            val_ds = MicroscopyDataset(
                root_dir=tmp_path, split="val", val_fraction=0.3, test_fraction=0.3
            )
            test_ds = MicroscopyDataset(
                root_dir=tmp_path, split="test", val_fraction=0.3, test_fraction=0.3
            )

            train_ds.prepare()
            val_ds.prepare()
            test_ds.prepare()

            train_paths = {train_ds.image_path(i) for i in range(len(train_ds))}
            val_paths = {val_ds.image_path(i) for i in range(len(val_ds))}
            test_paths = {test_ds.image_path(i) for i in range(len(test_ds))}

            # Splits must be disjoint even for tiny datasets
            assert train_paths.isdisjoint(val_paths)
            assert train_paths.isdisjoint(test_paths)
            assert val_paths.isdisjoint(test_paths)

            # All files accounted for
            all_paths = train_paths | val_paths | test_paths
            assert len(all_paths) == 3

    def test_empty_directory_raises_clear_error(self) -> None:
        """Empty directory raises ValueError with clear message."""
        import pytest

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = MicroscopyDataset(root_dir=tmp_path)

            with pytest.raises(ValueError) as exc_info:
                dataset.prepare()

            error_msg = str(exc_info.value)
            assert "No image files found" in error_msg
            assert str(tmp_path) in error_msg
            assert "recursively" in error_msg

    def test_rgb_image_converted_to_grayscale(self) -> None:
        """RGB images are converted to grayscale when loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create an RGB image
            rgb_img = Image.new("RGB", (64, 64), color=(255, 100, 50))
            rgb_img.save(tmp_path / "color.png")

            dataset = MicroscopyDataset(root_dir=tmp_path, split="train")
            dataset.prepare()

            loaded = dataset.load_image(0)
            # Should be converted to grayscale
            assert loaded.mode == "L"

    def test_all_loaded_images_are_grayscale(self) -> None:
        """Verify all loaded images are in grayscale mode L."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Mix of formats
            Image.new("RGB", (32, 32), (255, 0, 0)).save(tmp_path / "rgb.png")
            Image.new("L", (32, 32), 128).save(tmp_path / "gray.png")
            Image.new("RGBA", (32, 32), (0, 255, 0, 255)).save(tmp_path / "rgba.png")

            dataset = MicroscopyDataset(root_dir=tmp_path, split="train")
            dataset.prepare()

            # All should be loaded as grayscale
            for i in range(len(dataset)):
                img = dataset.load_image(i)
                assert img.mode == "L"

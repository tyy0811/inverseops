"""Day 2 tests for data loading and degradation utilities."""

from pathlib import Path

import numpy as np
from PIL import Image

from inverseops.data.degradations import (
    SUPPORTED_SIGMAS,
    add_gaussian_noise,
    generate_noisy_variants,
)
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





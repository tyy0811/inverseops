"""Tests for evaluation metrics (PSNR and SSIM)."""

import numpy as np
import pytest
from PIL import Image

from inverseops.evaluation.metrics import compute_psnr, compute_ssim


def _create_grayscale_image(width: int, height: int, value: int = 128) -> Image.Image:
    """Create a solid grayscale image."""
    arr = np.full((height, width), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _create_random_image(width: int, height: int, seed: int = 42) -> Image.Image:
    """Create a random grayscale image."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


class TestPSNR:
    """Tests for compute_psnr function."""

    def test_identical_images_returns_inf(self) -> None:
        """PSNR of identical images should be infinity."""
        img = _create_grayscale_image(64, 64, value=100)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images_finite_psnr(self) -> None:
        """PSNR of different images should be finite and positive."""
        ref = _create_grayscale_image(64, 64, value=100)
        pred = _create_grayscale_image(64, 64, value=110)
        psnr = compute_psnr(ref, pred)
        assert psnr > 0
        assert psnr < float("inf")

    def test_psnr_is_symmetric(self) -> None:
        """PSNR should be symmetric: PSNR(a, b) == PSNR(b, a)."""
        img1 = _create_random_image(64, 64, seed=1)
        img2 = _create_random_image(64, 64, seed=2)
        assert compute_psnr(img1, img2) == compute_psnr(img2, img1)

    def test_higher_noise_lower_psnr(self) -> None:
        """More noise should result in lower PSNR."""
        ref = _create_grayscale_image(64, 64, value=128)
        small_diff = _create_grayscale_image(64, 64, value=130)
        large_diff = _create_grayscale_image(64, 64, value=150)

        psnr_small = compute_psnr(ref, small_diff)
        psnr_large = compute_psnr(ref, large_diff)

        assert psnr_small > psnr_large

    def test_mismatched_sizes_raises(self) -> None:
        """Should raise ValueError for different image sizes."""
        img1 = _create_grayscale_image(64, 64)
        img2 = _create_grayscale_image(32, 32)

        with pytest.raises(ValueError, match="size mismatch"):
            compute_psnr(img1, img2)


class TestSSIM:
    """Tests for compute_ssim function."""

    def test_identical_images_returns_one(self) -> None:
        """SSIM of identical images should be 1.0."""
        img = _create_grayscale_image(64, 64, value=100)
        ssim = compute_ssim(img, img)
        assert ssim == pytest.approx(1.0, abs=1e-6)

    def test_similar_images_high_ssim(self) -> None:
        """Similar images should have SSIM close to 1."""
        ref = _create_grayscale_image(64, 64, value=100)
        pred = _create_grayscale_image(64, 64, value=102)  # Small difference
        ssim = compute_ssim(ref, pred)
        assert ssim > 0.99

    def test_different_images_lower_ssim(self) -> None:
        """Different images should have lower SSIM."""
        img1 = _create_random_image(64, 64, seed=1)
        img2 = _create_random_image(64, 64, seed=2)
        ssim = compute_ssim(img1, img2)
        # SSIM can be negative for very different images (e.g., random noise)
        assert ssim < 1.0

    def test_ssim_is_symmetric(self) -> None:
        """SSIM should be symmetric: SSIM(a, b) == SSIM(b, a)."""
        img1 = _create_random_image(64, 64, seed=1)
        img2 = _create_random_image(64, 64, seed=2)
        assert compute_ssim(img1, img2) == pytest.approx(
            compute_ssim(img2, img1), abs=1e-10
        )

    def test_ssim_range(self) -> None:
        """SSIM should be in [-1, 1] range."""
        img1 = _create_random_image(64, 64, seed=10)
        img2 = _create_random_image(64, 64, seed=20)
        ssim = compute_ssim(img1, img2)
        # SSIM is in [-1, 1]; random images may yield negative values
        assert -1 <= ssim <= 1

    def test_mismatched_sizes_raises(self) -> None:
        """Should raise ValueError for different image sizes."""
        img1 = _create_grayscale_image(64, 64)
        img2 = _create_grayscale_image(32, 32)

        with pytest.raises(ValueError, match="size mismatch"):
            compute_ssim(img1, img2)

    def test_image_too_small_raises(self) -> None:
        """Should raise ValueError if image is smaller than window."""
        img = _create_grayscale_image(8, 8)

        with pytest.raises(ValueError, match="too small"):
            compute_ssim(img, img, window_size=11)


class TestColorConversion:
    """Test that metrics handle non-grayscale images correctly."""

    def test_psnr_converts_rgb(self) -> None:
        """PSNR should work with RGB images by converting to grayscale."""
        gray = _create_grayscale_image(64, 64, value=128)
        rgb = gray.convert("RGB")

        psnr = compute_psnr(gray, rgb)
        assert psnr == float("inf")  # Same image, different mode

    def test_ssim_converts_rgb(self) -> None:
        """SSIM should work with RGB images by converting to grayscale."""
        gray = _create_grayscale_image(64, 64, value=128)
        rgb = gray.convert("RGB")

        ssim = compute_ssim(gray, rgb)
        assert ssim == pytest.approx(1.0, abs=1e-6)

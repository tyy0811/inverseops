"""Image quality metrics for denoising evaluation.

Provides PSNR and SSIM computation for grayscale images.
Uses numpy for simplicity - no heavy metric libraries required.
"""

import numpy as np
from PIL import Image


def compute_psnr(reference: Image.Image, prediction: Image.Image) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        reference: Ground truth image (converted to grayscale if needed).
        prediction: Predicted/restored image (converted to grayscale if needed).

    Returns:
        PSNR value in dB. Higher is better.
        Returns float('inf') for identical images.

    Raises:
        ValueError: If images have different sizes.
    """
    ref = _to_grayscale_array(reference)
    pred = _to_grayscale_array(prediction)

    if ref.shape != pred.shape:
        raise ValueError(
            f"Image size mismatch: reference {ref.shape} vs prediction {pred.shape}"
        )

    mse = np.mean((ref.astype(np.float64) - pred.astype(np.float64)) ** 2)

    if mse == 0:
        return float("inf")

    # Max pixel value for 8-bit images
    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel**2) / mse)

    return float(psnr)


def compute_ssim(
    reference: Image.Image,
    prediction: Image.Image,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute Structural Similarity Index between two images.

    Implements the SSIM metric from Wang et al. (2004).
    Uses a sliding window approach with Gaussian weighting.

    Args:
        reference: Ground truth image (converted to grayscale if needed).
        prediction: Predicted/restored image (converted to grayscale if needed).
        window_size: Size of the sliding window (must be odd).
        k1: SSIM constant for luminance (default 0.01).
        k2: SSIM constant for contrast (default 0.03).

    Returns:
        SSIM value in [0, 1]. Higher is better.
        Returns 1.0 for identical images.

    Raises:
        ValueError: If images have different sizes or window_size is too large.
    """
    ref = _to_grayscale_array(reference).astype(np.float64)
    pred = _to_grayscale_array(prediction).astype(np.float64)

    if ref.shape != pred.shape:
        raise ValueError(
            f"Image size mismatch: reference {ref.shape} vs prediction {pred.shape}"
        )

    if min(ref.shape) < window_size:
        raise ValueError(
            f"Image too small ({ref.shape}) for window_size={window_size}"
        )

    # Constants
    L = 255.0  # Dynamic range for 8-bit images
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # Create Gaussian window
    window = _gaussian_window(window_size)

    # Compute local means
    mu1 = _filter2d(ref, window)
    mu2 = _filter2d(pred, window)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = _filter2d(ref**2, window) - mu1_sq
    sigma2_sq = _filter2d(pred**2, window) - mu2_sq
    sigma12 = _filter2d(ref * pred, window) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    # Return mean SSIM
    return float(np.mean(ssim_map))


def _to_grayscale_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale numpy array."""
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image, dtype=np.uint8)


def _gaussian_window(size: int, sigma: float | None = None) -> np.ndarray:
    """Create a 2D Gaussian window for SSIM computation.

    Args:
        size: Window size (must be odd).
        sigma: Standard deviation. If None, uses size / 6.

    Returns:
        2D Gaussian kernel normalized to sum to 1.
    """
    if sigma is None:
        sigma = size / 6.0

    coords = np.arange(size) - (size - 1) / 2.0
    gauss_1d = np.exp(-(coords**2) / (2 * sigma**2))
    gauss_2d = np.outer(gauss_1d, gauss_1d)

    return gauss_2d / gauss_2d.sum()


def _filter2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 2D convolution with 'valid' boundary handling.

    Uses numpy's stride tricks for efficient sliding window.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    kh, kw = kernel.shape
    windows = sliding_window_view(image, (kh, kw))

    # Convolve: sum over window dimensions
    return np.einsum("ijkl,kl->ij", windows, kernel)

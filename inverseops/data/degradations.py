"""Synthetic degradation utilities.

Provides deterministic Gaussian noise injection for microscopy images.
"""

import numpy as np
from PIL import Image

# Supported noise levels (sigma values)
SUPPORTED_SIGMAS: tuple[int, ...] = (15, 25, 50)


def numpy_rng(seed: int | None = None) -> np.random.Generator:
    """Create a numpy random generator with optional seed.

    Args:
        seed: Random seed for reproducibility. None for non-deterministic.

    Returns:
        Numpy random Generator instance.
    """
    return np.random.default_rng(seed)


def add_gaussian_noise(
    image: Image.Image,
    sigma: float,
    seed: int | None = None,
) -> Image.Image:
    """Add Gaussian noise to an image.

    Args:
        image: Input PIL image (will be converted to grayscale if needed).
        sigma: Standard deviation of the Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        Noisy PIL image in grayscale mode ('L').
    """
    # Ensure grayscale
    img_gray = image.convert("L")
    arr = np.array(img_gray, dtype=np.float32)

    # Generate noise
    rng = numpy_rng(seed)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)

    # Add noise and clip to valid range
    noisy = arr + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy, mode="L")


def generate_noisy_variants(
    image: Image.Image,
    sigmas: tuple[int, ...] = SUPPORTED_SIGMAS,
    seed: int = 42,
) -> dict[int, Image.Image]:
    """Generate noisy variants of an image at multiple sigma levels.

    Uses deterministic seeding: each sigma level gets seed + sigma_index
    to ensure reproducibility while producing different noise patterns.

    Args:
        image: Input PIL image.
        sigmas: Tuple of sigma values to use.
        seed: Base random seed.

    Returns:
        Dictionary mapping sigma values to noisy images.
    """
    variants: dict[int, Image.Image] = {}
    for i, sigma in enumerate(sigmas):
        # Use offset seed for each sigma to get different but reproducible noise
        variant_seed = seed + i
        variants[sigma] = add_gaussian_noise(image, sigma, seed=variant_seed)
    return variants

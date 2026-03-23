"""Minimal image transformation utilities.

Simple helpers for Day 2 data pipeline. Does not depend on torchvision.
"""

from PIL import Image


def to_grayscale(image: Image.Image) -> Image.Image:
    """Convert image to grayscale.

    Args:
        image: Input PIL image.

    Returns:
        Grayscale PIL image in mode 'L'.
    """
    return image.convert("L")


def center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center crop an image to the specified size.

    Args:
        image: Input PIL image.
        size: Target (width, height) tuple.

    Returns:
        Center-cropped PIL image.
    """
    width, height = image.size
    target_w, target_h = size

    left = (width - target_w) // 2
    top = (height - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return image.crop((left, top, right, bottom))


def normalize_to_uint8(image: Image.Image) -> Image.Image:
    """Ensure image is in uint8 format.

    Converts to grayscale mode 'L' which is uint8 by definition in PIL.

    Args:
        image: Input PIL image.

    Returns:
        PIL image in mode 'L' (uint8 grayscale).
    """
    return image.convert("L")

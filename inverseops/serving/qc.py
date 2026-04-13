"""Quality control layer for inference.

Input validation, output checks, and noise-level calibration logic.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Calibrated range around training sigmas (15, 25, 50)
CALIBRATED_RANGE = (10.0, 60.0)
MAX_RESOLUTION = 2048
SUPPORTED_MODES = {"L", "RGB", "I", "F"}


def validate_input(image: Image.Image) -> list[str]:
    """Check input image for issues. Returns list of issue strings (empty = valid)."""
    issues = []

    if image.mode not in SUPPORTED_MODES:
        issues.append(f"Unsupported image mode: {image.mode}")

    w, h = image.size
    if w > MAX_RESOLUTION or h > MAX_RESOLUTION:
        issues.append(
            f"Resolution {w}x{h} exceeds max "
            f"{MAX_RESOLUTION}x{MAX_RESOLUTION}"
        )

    if w < 8 or h < 8:
        issues.append(f"Resolution {w}x{h} too small (min 8x8)")

    return issues


def validate_output(output: np.ndarray) -> tuple[bool, list[str]]:
    """Check output array for NaN/Inf or other problems."""
    issues = []

    if np.any(np.isnan(output)):
        issues.append("Output contains NaN values")

    if np.any(np.isinf(output)):
        issues.append("Output contains Inf values")

    return len(issues) == 0, issues


def estimate_noise_level(image: Image.Image) -> float:
    """Estimate noise level using Median Absolute Deviation of wavelet coefficients.

    Uses the robust MAD estimator on the finest-scale wavelet detail coefficients
    (approximated by high-pass filtering). This is the standard approach from
    Donoho & Johnstone (1994).
    """
    arr = np.array(image.convert("L"), dtype=np.float64) / 255.0

    # High-pass filter (Laplacian-like) to approximate detail coefficients
    # Using the standard 3x3 kernel for noise estimation
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])

    from scipy.signal import convolve2d

    detail = convolve2d(arr, kernel, mode="valid")

    # MAD estimator: sigma = MAD / 0.6745
    mad = np.median(np.abs(detail - np.median(detail)))
    sigma_normalized = mad / 0.6745

    # Convert back to [0, 255] scale
    return float(sigma_normalized * 255.0)


def decide_denoise(
    noise_level: float | None,
    noise_level_estimated: float | None,
    output_valid: bool,
) -> str:
    """Denoise-task QC decision: 'good', 'review', or 'out_of_range'.

    Uses CALIBRATED_RANGE sigma check. Uses provided noise_level if
    available, otherwise falls back to the estimate.
    """
    if not output_valid:
        return "review"

    sigma = noise_level if noise_level is not None else noise_level_estimated
    if sigma is None:
        return "review"

    if sigma < CALIBRATED_RANGE[0] or sigma > CALIBRATED_RANGE[1]:
        return "out_of_range"

    return "good"


def decide_sr(output_valid: bool) -> str:
    """Super-resolution QC decision.

    SR has no noise-level calibration (model trained on clean LR), so QC
    checks input validity and output finiteness only. Input resolution
    bounds (8x8–2048x2048) are enforced by the shared validate_input
    path before this function runs.
    """
    if not output_valid:
        return "review"
    return "good"

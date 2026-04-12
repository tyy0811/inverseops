"""Sliding-window SR inference for 2x upscaling models.

Replicates the W2S `code/SR/test.py` assembly strategy: 128x128 LR patches
with stride 64, full-image output built from the interior 192x192 region
of each 256x256 HR patch (outer 64-pixel border discarded for interior
patches, kept for border patches). This matches the behavior of the W2S
pretrained RRDBNet test code exactly.

The function handles arbitrary LR shapes (including non-multiples of 64)
because the fully-convolutional SR model produces a proportionally smaller
output for border patches, so NumPy's read-side slice truncation on the
input lines up with the write-side truncation on the output.

Extracted from `scripts/modal_sr_calibration.py` so the stitching logic
can be unit-tested without requiring the Modal runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def sliding_window_sr(
    model: Any,
    lr_arr: np.ndarray,
    device: str | torch.device,
    *,
    clamp: bool = True,
) -> np.ndarray:
    """Run 2x SR inference over an LR image via a sliding window.

    Args:
        model: a fully-convolutional 2x SR model (PyTorch nn.Module or
            DataParallel wrapper). Expected input shape (1, 1, h, w),
            output (1, 1, 2h, 2w). Models returning a list (e.g. SRFBN)
            use the last element.
        lr_arr: low-resolution input array, shape (H, W), dtype float32.
            Interpretation of the input space (PNG/255, Z-score, etc.)
            is caller-controlled.
        device: torch device string or object to run inference on.
        clamp: if True (default), clamp model output to [0, 1] before
            assembly. Use False when the input/output space is not
            [0, 1] (e.g. Z-score inputs where the raw model output
            preserves the trained range).

    Returns:
        High-resolution output array, shape (2H, 2W), dtype float64.

    Stitching strategy (matches W2S code/SR/test.py):
        - Patches are 128x128 in LR space, stride 64 in LR space
          (equivalently 256x256 and 128 in HR space).
        - For each patch, the inner 192x192 region of the HR output
          (starting at offset 64) is written to the assembly buffer.
        - For patches at the top edge (x < 64), the top 64 rows are
          additionally kept. Same for left edge (y < 64) and corner.
        - Border patches (where the LR slice is smaller than 128x128)
          produce a proportionally smaller HR output, and the write-side
          slicing is truncated to match by NumPy's standard slice rules.
    """
    h, w = lr_arr.shape
    img_ans = np.zeros((h * 2, w * 2), dtype=np.float64)

    x = 0
    while x < h:
        y = 0
        while y < w:
            patch = lr_arr[x : x + 128, y : y + 128]
            inp = torch.from_numpy(patch.copy()).unsqueeze(0).unsqueeze(0)
            inp = inp.float().to(device)
            with torch.no_grad():
                sr_patch = model(inp)
            if isinstance(sr_patch, list):
                sr_patch = sr_patch[-1]
            sr_patch_np = sr_patch.cpu().numpy()
            if clamp:
                sr_patch_np = np.clip(sr_patch_np, 0.0, 1.0)
            sr_patch_np = sr_patch_np[0, 0]

            img_ans[x * 2 + 64 : x * 2 + 256, y * 2 + 64 : y * 2 + 256] = (
                sr_patch_np[64:, 64:]
            )
            if x < 64:
                img_ans[x * 2 : x * 2 + 64, y * 2 + 64 : y * 2 + 256] = (
                    sr_patch_np[:64, 64:]
                )
            if y < 64:
                img_ans[x * 2 + 64 : x * 2 + 256, y * 2 : y * 2 + 64] = (
                    sr_patch_np[64:, :64]
                )
            if x < 64 and y < 64:
                img_ans[x * 2 : x * 2 + 64, y * 2 : y * 2 + 64] = (
                    sr_patch_np[:64, :64]
                )

            y += 64
        x += 64

    return img_ans


__all__ = ["sliding_window_sr"]

"""Unit tests for the 2x SR sliding-window stitcher.

The stitching logic in `inverseops.evaluation.stitching.sliding_window_sr`
is used by the SR calibration script (Decision 19) and will be used by
the SwinIR SR eval path. These tests pin the invariants that previously
were only exercised by a Modal-only smoke test:

  1. Output shape is 2x input shape for representative LR sizes,
     including sizes not divisible by 64 (border patch handling).
  2. No zero pixels in the output for a non-zero input (no stitching
     gaps).
  3. For a nearest-neighbor 2x "fake model", the stitched output
     equals np.kron(lr, np.ones((2,2))) — the exact 2x NN upscale.
     This is a strong identity that catches off-by-one errors in the
     overlap assembly (every overlapping region must be consistent
     with the global upscale).
  4. The `clamp` flag correctly gates [0, 1] clipping.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from inverseops.evaluation.stitching import sliding_window_sr


class _FakeNearest2x(nn.Module):
    """Mock SR model: deterministic nearest-neighbor 2x upscale.

    Fully convolutional in the trivial sense — handles arbitrary input
    spatial shapes and always produces output of shape (B, C, 2H, 2W).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")


@pytest.fixture
def fake_model() -> _FakeNearest2x:
    m = _FakeNearest2x()
    m.eval()
    return m


@pytest.mark.parametrize(
    "shape",
    [(512, 512), (500, 500), (256, 256), (128, 128)],
)
def test_sliding_window_sr_output_shape(
    fake_model: _FakeNearest2x, shape: tuple[int, int]
) -> None:
    """Output shape is always exactly 2x the input shape."""
    lr = np.random.default_rng(0).random(shape).astype(np.float32)
    out = sliding_window_sr(fake_model, lr, device="cpu")
    assert out.shape == (shape[0] * 2, shape[1] * 2)
    assert out.dtype == np.float64


@pytest.mark.parametrize(
    "shape",
    [(512, 512), (500, 500), (256, 256)],
)
def test_sliding_window_sr_full_coverage(
    fake_model: _FakeNearest2x, shape: tuple[int, int]
) -> None:
    """Every output pixel is written (no stitching gaps).

    An all-positive input fed through a nearest-neighbor model cannot
    produce zero outputs, so any zero in the assembled image is a
    stitching gap.
    """
    lr = np.random.default_rng(0).random(shape).astype(np.float32) + 0.5
    out = sliding_window_sr(fake_model, lr, device="cpu", clamp=False)
    n_zero = int(np.sum(out == 0))
    assert n_zero == 0, (
        f"Stitching left {n_zero} zero pixels out of {out.size} "
        f"for LR shape {shape}"
    )


@pytest.mark.parametrize(
    "shape",
    [(512, 512), (500, 500), (256, 256), (128, 128)],
)
def test_sliding_window_sr_identity_with_nn_upscaler(
    fake_model: _FakeNearest2x, shape: tuple[int, int]
) -> None:
    """For a nearest-neighbor 2x model, stitching must reproduce the
    full-image 2x NN upscale exactly.

    This is the strongest stitching invariant: every overlapping region
    must be consistent with the global upscale regardless of which
    patch "owns" the region in the write-order. If any boundary or
    edge case is off by one pixel, this test fails.
    """
    lr = np.random.default_rng(0).random(shape).astype(np.float32)
    out = sliding_window_sr(fake_model, lr, device="cpu", clamp=False)
    expected = np.kron(lr.astype(np.float64), np.ones((2, 2), dtype=np.float64))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_sliding_window_sr_clamp_true_clips_to_unit_interval(
    fake_model: _FakeNearest2x,
) -> None:
    """With clamp=True, all output pixels are within [0, 1] even if
    the input (and therefore the fake model output) is not."""
    rng = np.random.default_rng(0)
    # Deliberately out-of-range input so the NN-2x model produces
    # out-of-range output that must be clipped by sliding_window_sr.
    lr = (rng.random((256, 256)).astype(np.float32) * 4.0) - 1.5
    assert lr.min() < 0.0 and lr.max() > 1.0  # sanity on the fixture
    out = sliding_window_sr(fake_model, lr, device="cpu", clamp=True)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_sliding_window_sr_clamp_false_preserves_out_of_range(
    fake_model: _FakeNearest2x,
) -> None:
    """With clamp=False, out-of-range values pass through unchanged."""
    rng = np.random.default_rng(0)
    lr = (rng.random((256, 256)).astype(np.float32) * 4.0) - 1.5
    assert lr.min() < 0.0 and lr.max() > 1.0
    out = sliding_window_sr(fake_model, lr, device="cpu", clamp=False)
    assert out.min() < 0.0 or out.max() > 1.0
    # And since the fake model is NN-2x, the range should closely
    # match the input range (no conv/activation compression).
    assert np.isclose(out.min(), lr.min(), atol=1e-5)
    assert np.isclose(out.max(), lr.max(), atol=1e-5)

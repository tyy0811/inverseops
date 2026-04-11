"""Tests for the evaluation harness metrics and aggregation."""

import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts.run_evaluation import (
    DATASET_DATA_RANGE,
    _aggregate_results,
    _get_sample_keys,
    psnr_tensor,
    ssim_tensor,
)


class TestPsnrTensor:

    def test_identical_tensors_return_inf(self):
        a = torch.randn(1, 64, 64)
        assert psnr_tensor(a, a) == float("inf")

    def test_known_mse(self):
        """PSNR = 10*log10(data_range^2 / MSE). For MSE=1, range=255: ~48.13 dB."""
        a = torch.zeros(1, 64, 64)
        b = torch.ones(1, 64, 64)  # MSE = 1.0
        p = psnr_tensor(a, b, data_range=255.0)
        expected = 10.0 * np.log10(255.0**2 / 1.0)
        assert abs(p - expected) < 0.01

    def test_higher_noise_lower_psnr(self):
        a = torch.full((1, 64, 64), 128.0)
        low_noise = a + torch.randn_like(a) * 5
        high_noise = a + torch.randn_like(a) * 50
        p_low = psnr_tensor(a, low_noise, data_range=255.0)
        p_high = psnr_tensor(a, high_noise, data_range=255.0)
        assert p_low > p_high

    def test_fixed_data_range_not_per_image(self):
        """Same noise, different signal levels — PSNR should differ with fixed range."""
        noise = torch.randn(1, 64, 64) * 10
        a_bright = torch.full((1, 64, 64), 200.0)
        a_dark = torch.full((1, 64, 64), 50.0)
        p_bright = psnr_tensor(a_bright, a_bright + noise, data_range=255.0)
        p_dark = psnr_tensor(a_dark, a_dark + noise, data_range=255.0)
        # Same noise, same fixed range — PSNR should be equal (within noise seed variation)
        assert abs(p_bright - p_dark) < 1.0


class TestSsimTensor:

    def test_identical_tensors_return_one(self):
        a = torch.randn(1, 64, 64) * 50 + 128
        s = ssim_tensor(a, a, data_range=255.0)
        assert abs(s - 1.0) < 0.001

    def test_uncorrelated_noise_low_ssim(self):
        a = torch.randn(1, 64, 64) * 50 + 128
        b = torch.randn(1, 64, 64) * 50 + 128
        s = ssim_tensor(a, b, data_range=255.0)
        assert s < 0.5

    def test_ssim_between_zero_and_one(self):
        a = torch.randn(1, 64, 64) * 50 + 128
        b = a + torch.randn_like(a) * 10
        s = ssim_tensor(a, b, data_range=255.0)
        assert 0.0 <= s <= 1.0


class TestAggregateResults:

    def test_per_fov_then_across_fovs(self):
        """Aggregation: per-FoV mean first, then mean +/- std across FoVs."""
        psnr = defaultdict(lambda: defaultdict(list))
        ssim = defaultdict(lambda: defaultdict(list))

        # Noise level 1, 2 FoVs, 3 wavelengths each
        psnr[1][10] = [30.0, 32.0, 31.0]  # FoV 10 mean = 31.0
        psnr[1][20] = [28.0, 29.0, 27.0]  # FoV 20 mean = 28.0
        ssim[1][10] = [0.90, 0.92, 0.91]  # FoV 10 mean = 0.91
        ssim[1][20] = [0.85, 0.86, 0.84]  # FoV 20 mean = 0.85

        results = _aggregate_results(psnr, ssim)

        assert results[1]["n_units"] == 2
        # Mean of FoV means: (31.0 + 28.0) / 2 = 29.5
        assert abs(results[1]["psnr_mean"] - 29.5) < 0.01
        # Std of FoV means: std([31.0, 28.0])
        expected_std = float(np.std([31.0, 28.0]))
        assert abs(results[1]["psnr_std"] - expected_std) < 0.01
        # SSIM: mean of (0.91, 0.85) = 0.88
        assert abs(results[1]["ssim_mean"] - 0.88) < 0.01

    def test_multiple_noise_levels(self):
        psnr = defaultdict(lambda: defaultdict(list))
        ssim = defaultdict(lambda: defaultdict(list))

        psnr[1][1] = [30.0]
        psnr[8][1] = [35.0]
        ssim[1][1] = [0.85]
        ssim[8][1] = [0.92]

        results = _aggregate_results(psnr, ssim)
        assert sorted(results.keys()) == [1, 8]
        assert results[1]["psnr_mean"] == 30.0
        assert results[8]["psnr_mean"] == 35.0


class TestDatasetDataRange:

    def test_w2s_range_is_255(self):
        assert DATASET_DATA_RANGE["w2s"] == 255.0

    def test_ixi_range_is_1(self):
        assert DATASET_DATA_RANGE["ixi"] == 1.0


class TestDatasetSampleKeys:

    def test_w2s_keys(self):
        group_key, unit_key = _get_sample_keys("w2s")
        assert group_key == "noise_level"
        assert unit_key == "fov_id"

    def test_ixi_keys(self):
        group_key, unit_key = _get_sample_keys("ixi")
        assert group_key == "sigma"
        assert unit_key == "subject_id"

    def test_unknown_dataset_falls_back(self):
        group_key, unit_key = _get_sample_keys("unknown")
        assert group_key == "noise_level"
        assert unit_key == "fov_id"

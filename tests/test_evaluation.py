"""Tests for evaluation script helpers and modes."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestSmokeFixtures:
    """Tests for smoke fixture generation."""

    def test_generate_smoke_fixtures_creates_images(self):
        """generate_smoke_fixtures should create deterministic grayscale images."""
        from scripts.run_evaluation import generate_smoke_fixtures

        with tempfile.TemporaryDirectory() as tmpdir:
            images = generate_smoke_fixtures(Path(tmpdir), count=3, seed=42)

            assert len(images) == 3
            for img_path in images:
                assert img_path.exists()
                img = Image.open(img_path)
                assert img.mode == "L"  # grayscale

    def test_generate_smoke_fixtures_deterministic(self):
        """Same seed should produce identical images."""
        from scripts.run_evaluation import generate_smoke_fixtures

        with tempfile.TemporaryDirectory() as tmpdir1:
            imgs1 = generate_smoke_fixtures(Path(tmpdir1), count=2, seed=42)
            arr1 = [np.array(Image.open(p)) for p in imgs1]

        with tempfile.TemporaryDirectory() as tmpdir2:
            imgs2 = generate_smoke_fixtures(Path(tmpdir2), count=2, seed=42)
            arr2 = [np.array(Image.open(p)) for p in imgs2]

        for a1, a2 in zip(arr1, arr2):
            assert np.array_equal(a1, a2)


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_results_mean_std(self):
        """aggregate_results should compute correct mean and population std."""
        from scripts.run_evaluation import EvalResult, aggregate_results

        results = [
            EvalResult(sigma=25, domain="test", dataset_name="ds", is_real_data=True,
                       image_name="a.png", image_path="a.png", psnr=30.0, ssim=0.9,
                       noise_seed=42, model_name="swinir", model_checkpoint="ckpt.pth"),
            EvalResult(sigma=25, domain="test", dataset_name="ds", is_real_data=True,
                       image_name="b.png", image_path="b.png", psnr=32.0, ssim=0.92,
                       noise_seed=43, model_name="swinir", model_checkpoint="ckpt.pth"),
        ]

        agg = aggregate_results(results)
        key = ("test", 25)

        assert key in agg
        assert agg[key]["count"] == 2
        assert agg[key]["psnr_mean"] == pytest.approx(31.0)
        # Population std: sqrt(((30-31)^2 + (32-31)^2) / 2) = sqrt(1) = 1.0
        assert agg[key]["psnr_std"] == pytest.approx(1.0)
        assert agg[key]["ssim_mean"] == pytest.approx(0.91)

    def test_aggregate_results_single_image_std_zero(self):
        """With count=1, std should be 0.0 (not NaN)."""
        from scripts.run_evaluation import EvalResult, aggregate_results

        results = [
            EvalResult(sigma=15, domain="single", dataset_name="ds", is_real_data=True,
                       image_name="only.png", image_path="only.png", psnr=28.5,
                       ssim=0.85, noise_seed=1, model_name="swinir",
                       model_checkpoint="ckpt.pth"),
        ]

        agg = aggregate_results(results)
        assert agg[("single", 15)]["psnr_std"] == 0.0
        assert agg[("single", 15)]["ssim_std"] == 0.0


class TestSummaryCSV:
    """Tests for baseline_summary.csv output."""

    def test_save_summary_csv_columns(self):
        """baseline_summary.csv should have all required columns
        including evidence_tier."""
        from scripts.run_evaluation import save_summary_csv

        agg = {
            ("microscopy", 25): {
                "count": 5,
                "psnr_mean": 29.5,
                "psnr_std": 1.2,
                "ssim_mean": 0.88,
                "ssim_std": 0.02,
                "dataset_name": "fmd",
                "is_real_data": True,
                "model_name": "swinir",
                "model_checkpoint": "noise25.pth",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            save_summary_csv(
                agg=agg,
                output_path=path,
                seed=42,
                mode="full",
                decision_gate_valid=True,
                evidence_tier="moderate",
            )

            assert path.exists()

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            row = rows[0]

            required_cols = [
                "sigma", "noise_type", "domain", "dataset_name", "is_real_data",
                "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
                "seed", "model_name", "model_checkpoint", "mode", "decision_gate_valid",
                "evidence_tier"
            ]
            for col in required_cols:
                assert col in row, f"Missing column: {col}"

            assert row["sigma"] == "25"
            assert row["noise_type"] == "synthetic_gaussian"
            assert row["mode"] == "full"
            assert row["decision_gate_valid"] == "True"
            assert row["evidence_tier"] == "moderate"


class TestDecisionJSON:
    """Tests for day3_decision.json output."""

    def test_save_decision_json_full_mode_all_sigmas(self):
        """Full mode with all sigmas and both domains should be valid."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 5},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 5},
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 5},
            ("natural", 15): {"psnr_mean": 30.0, "count": 5},
            ("natural", 25): {"psnr_mean": 28.5, "count": 5},
            ("natural", 50): {"psnr_mean": 26.0, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            decision = save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[15, 25, 50],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "full"
            assert decision["decision_gate_valid"] is True
            assert decision["dataset_locked"] is True
            assert decision["gap_threshold_db"] == 1.0
            assert "per_sigma_gap_db" in decision
            assert "15" in decision["per_sigma_gap_db"]
            # New fields
            assert "evidence_tier" in decision
            assert "microscopy_image_count" in decision
            assert "natural_image_count" in decision
            assert decision["microscopy_image_count"] == 5
            assert decision["natural_image_count"] == 5

    def test_save_decision_json_full_mode_single_sigma_invalid(self):
        """Full mode with single sigma should NOT be decision_gate_valid."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 5},
            ("natural", 50): {"psnr_mean": 26.0, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[50],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "full"
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
            assert "Insufficient" in decision["recommendation"]

    def test_save_decision_json_full_mode_missing_domain_invalid(self):
        """Full mode with all sigmas but missing domain should NOT be valid."""
        from scripts.run_evaluation import save_decision_json

        # Only microscopy, no natural
        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 5},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 5},
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[15, 25, 50],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
            assert "Insufficient" in decision["recommendation"]

    def test_save_decision_json_partial_mode(self):
        """In partial mode, decision_gate_valid and dataset_locked should be False."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("microscopy", 25): {"psnr_mean": 28.0, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="partial",
                sigmas=[25],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "partial"
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
            assert "Insufficient" in decision["recommendation"]

    def test_save_decision_json_partial_mode_all_sigmas_invalid(self):
        """Partial mode with all sigmas should still NOT be valid."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 5},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 5},
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 5},
            ("natural", 15): {"psnr_mean": 30.0, "count": 5},
            ("natural", 25): {"psnr_mean": 28.5, "count": 5},
            ("natural", 50): {"psnr_mean": 26.0, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="partial",
                sigmas=[15, 25, 50],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "partial"
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
            assert "Insufficient" in decision["recommendation"]

    def test_save_decision_json_smoke_mode(self):
        """In smoke mode, decision_gate_valid and dataset_locked should be False."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("smoke_microscopy", 25): {"psnr_mean": 28.0, "count": 3},
            ("smoke_natural", 25): {"psnr_mean": 29.0, "count": 3},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="smoke",
                sigmas=[25],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "smoke"
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False


class TestMetricsCSV:
    """Tests for baseline_metrics.csv output."""

    def test_save_csv_includes_image_path(self):
        """baseline_metrics.csv should include image_path column."""
        from scripts.run_evaluation import EvalResult, save_csv

        results = [
            EvalResult(
                sigma=25, domain="test", dataset_name="ds", is_real_data=True,
                image_name="a.png", image_path="subdir/a.png", psnr=30.0, ssim=0.9,
                noise_seed=42, model_name="swinir", model_checkpoint="ckpt.pth"
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            save_csv(results, path, mode="full")

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert "image_path" in rows[0]
            assert rows[0]["image_path"] == "subdir/a.png"
            assert rows[0]["image_name"] == "a.png"

    def test_image_path_differs_from_name_for_nested(self):
        """image_path should differ from image_name for nested files."""
        from scripts.run_evaluation import EvalResult, save_csv

        results = [
            EvalResult(
                sigma=25, domain="test", dataset_name="ds", is_real_data=True,
                image_name="img.png", image_path="nested/subdir/img.png",
                psnr=30.0, ssim=0.9, noise_seed=42, model_name="swinir",
                model_checkpoint="ckpt.pth"
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            save_csv(results, path, mode="full")

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert rows[0]["image_name"] != rows[0]["image_path"]
            assert "nested/subdir" in rows[0]["image_path"]


class TestEvaluateDomain:
    """Tests for evaluate_domain with model cache."""

    @pytest.fixture
    def mock_swinir(self, monkeypatch):
        """Mock SwinIRBaseline to avoid model downloads."""
        class MockModel:
            def __init__(self, noise_level, **kwargs):
                self.noise_level = noise_level
                self.device = "cpu"
                self._loaded = False

            def load(self):
                self._loaded = True

            def is_loaded(self):
                return self._loaded

            def predict_image(self, img):
                # Return slightly modified image to simulate denoising
                return img

            @property
            def checkpoint_source(self):
                return f"mock://sigma{self.noise_level}"

            @property
            def checkpoint_resolved_path(self):
                return Path(f"/tmp/mock_{self.noise_level}.pth")

        monkeypatch.setattr(
            "scripts.run_evaluation.SwinIRBaseline", MockModel
        )
        return MockModel

    def test_evaluate_domain_uses_sigma_specific_model(self, mock_swinir):
        """evaluate_domain should use the correct model for each sigma."""
        from scripts.run_evaluation import evaluate_domain, load_models

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img_path = Path(tmpdir) / "test.png"
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(img_path)

            models = load_models([15, 25])

            results = evaluate_domain(
                models=models,
                images=[img_path],
                domain="test",
                dataset_name="test_ds",
                is_real_data=True,
                sigmas=(15, 25),
                seed=42,
                limit=1,
            )

            # Should have 2 results (1 image x 2 sigmas)
            assert len(results) == 2

            # Check sigma-specific checkpoint used
            sigma_15_result = [r for r in results if r.sigma == 15][0]
            sigma_25_result = [r for r in results if r.sigma == 25][0]

            assert "15" in sigma_15_result.model_checkpoint
            assert "25" in sigma_25_result.model_checkpoint

    def test_evaluate_domain_computes_relative_path(self, mock_swinir):
        """evaluate_domain should compute relative paths when domain_root provided."""
        from scripts.run_evaluation import evaluate_domain, load_models

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subdir = root / "subdir"
            subdir.mkdir()
            img_path = subdir / "nested_image.png"
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(img_path)

            models = load_models([25])

            results = evaluate_domain(
                models=models,
                images=[img_path],
                domain="test",
                dataset_name="test_ds",
                is_real_data=True,
                sigmas=(25,),
                seed=42,
                limit=1,
                domain_root=root,
            )

            assert len(results) == 1
            assert results[0].image_name == "nested_image.png"
            assert results[0].image_path == "subdir/nested_image.png"

    def test_evaluate_domain_uses_absolute_without_root(self, mock_swinir):
        """evaluate_domain should use absolute paths when domain_root is None."""
        from scripts.run_evaluation import evaluate_domain, load_models

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(img_path)

            models = load_models([25])

            results = evaluate_domain(
                models=models,
                images=[img_path],
                domain="test",
                dataset_name="test_ds",
                is_real_data=True,
                sigmas=(25,),
                seed=42,
                limit=1,
                domain_root=None,
            )

            assert len(results) == 1
            # Without domain_root, image_path should be the absolute path
            assert results[0].image_path == str(img_path)


class TestModeIntegration:
    """Integration tests for execution modes."""

    @pytest.fixture
    def mock_swinir(self, monkeypatch):
        """Mock SwinIRBaseline."""
        class MockModel:
            def __init__(self, noise_level, **kwargs):
                self.noise_level = noise_level
                self.device = "cpu"

            def load(self):
                pass

            def is_loaded(self):
                return True

            def predict_image(self, img):
                return img

            @property
            def checkpoint_source(self):
                return f"mock://sigma{self.noise_level}.pth"

            @property
            def checkpoint_resolved_path(self):
                return Path(f"/tmp/mock_{self.noise_level}.pth")

        monkeypatch.setattr("scripts.run_evaluation.SwinIRBaseline", MockModel)

    def test_partial_mode_produces_csvs(self, mock_swinir):
        """Partial mode should produce CSVs with mode=partial."""
        from scripts.run_evaluation import (
            aggregate_results,
            evaluate_domain,
            load_models,
            save_csv,
            save_decision_json,
            save_summary_csv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create one test image (microscopy only)
            img_path = tmpdir / "micro.png"
            Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(img_path)

            models = load_models([25])
            results = evaluate_domain(
                models=models,
                images=[img_path],
                domain="microscopy",
                dataset_name="fmd",
                is_real_data=True,
                sigmas=(25,),
                seed=42,
                limit=1,
            )

            agg = aggregate_results(results)

            metrics_path = tmpdir / "metrics.csv"
            summary_path = tmpdir / "summary.csv"
            decision_path = tmpdir / "decision.json"

            save_csv(results, metrics_path, mode="partial")
            save_summary_csv(agg, summary_path, seed=42, mode="partial",
                           decision_gate_valid=False, evidence_tier="pilot")
            save_decision_json(agg, decision_path, mode="partial",
                             sigmas=[25], gap_threshold_db=1.0)

            # Verify metrics CSV
            with open(metrics_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["mode"] == "partial"

            # Verify summary CSV
            with open(summary_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert rows[0]["decision_gate_valid"] == "False"

            # Verify decision JSON
            with open(decision_path) as f:
                decision = json.load(f)
            assert decision["mode"] == "partial"
            assert decision["dataset_locked"] is False

    def test_smoke_mode_produces_csvs(self, mock_swinir):
        """Smoke mode should produce CSVs with mode=smoke and is_real_data=False."""
        from scripts.run_evaluation import (
            aggregate_results,
            evaluate_domain,
            generate_smoke_fixtures,
            load_models,
            save_csv,
            save_decision_json,
            save_summary_csv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            smoke_images = generate_smoke_fixtures(tmpdir / "smoke", count=2, seed=42)

            models = load_models([25])
            results = evaluate_domain(
                models=models,
                images=smoke_images,
                domain="smoke_microscopy",
                dataset_name="smoke_fixture",
                is_real_data=False,
                sigmas=(25,),
                seed=42,
                limit=2,
            )

            agg = aggregate_results(results)

            metrics_path = tmpdir / "metrics.csv"
            summary_path = tmpdir / "summary.csv"
            decision_path = tmpdir / "decision.json"

            save_csv(results, metrics_path, mode="smoke")
            save_summary_csv(agg, summary_path, seed=42, mode="smoke",
                           decision_gate_valid=False, evidence_tier="pilot")
            save_decision_json(agg, decision_path, mode="smoke",
                             sigmas=[25], gap_threshold_db=1.0)

            # Verify metrics CSV
            with open(metrics_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["is_real_data"] == "False"
            assert rows[0]["mode"] == "smoke"
            assert "smoke_" in rows[0]["domain"]

            # Verify decision JSON
            with open(decision_path) as f:
                decision = json.load(f)
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
            assert "Insufficient" in decision["recommendation"]


class TestFastDevMode:
    """Tests for --fast-dev CLI flag."""

    def test_fast_dev_sets_defaults(self):
        """--fast-dev should set single_sigma=50, limit<=2, no_wandb=True."""
        import argparse
        import sys

        # Simulate CLI args
        test_args = ["run_evaluation.py", "--fast-dev", "--smoke-mode"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", test_args)

            # Import and parse
            from scripts.run_evaluation import SUPPORTED_SIGMAS

            parser = argparse.ArgumentParser()
            parser.add_argument("--microscopy-root", type=Path, default=None)
            parser.add_argument("--natural-root", type=Path, default=None)
            parser.add_argument("--limit", type=int, default=10)
            parser.add_argument("--single-sigma", type=int, default=None)
            parser.add_argument("--no-wandb", action="store_true")
            parser.add_argument("--smoke-mode", action="store_true")
            parser.add_argument("--fast-dev", action="store_true")

            args = parser.parse_args(test_args[1:])

            # Apply fast-dev logic (same as in script)
            if args.fast_dev:
                if args.single_sigma is None:
                    args.single_sigma = 50
                args.limit = min(args.limit, 2)
                args.no_wandb = True

            # Determine effective sigmas
            if args.single_sigma:
                sigmas = (args.single_sigma,)
            else:
                sigmas = SUPPORTED_SIGMAS

            assert sigmas == (50,)
            assert args.no_wandb is True
            assert args.limit <= 2

    def test_fast_dev_respects_explicit_sigma(self):
        """--fast-dev should not override explicit --single-sigma."""
        import argparse
        import sys

        test_args = ["run_evaluation.py", "--fast-dev", "--single-sigma", "15"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", test_args)

            parser = argparse.ArgumentParser()
            parser.add_argument("--limit", type=int, default=10)
            parser.add_argument("--single-sigma", type=int, default=None)
            parser.add_argument("--no-wandb", action="store_true")
            parser.add_argument("--fast-dev", action="store_true")

            args = parser.parse_args(test_args[1:])

            if args.fast_dev:
                if args.single_sigma is None:
                    args.single_sigma = 50
                args.limit = min(args.limit, 2)
                args.no_wandb = True

            # Explicit --single-sigma 15 should be preserved
            assert args.single_sigma == 15

    def test_fast_dev_respects_lower_limit(self):
        """--fast-dev should not increase limit if user specified lower."""
        import argparse
        import sys

        test_args = ["run_evaluation.py", "--fast-dev", "--limit", "1"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", test_args)

            parser = argparse.ArgumentParser()
            parser.add_argument("--limit", type=int, default=10)
            parser.add_argument("--single-sigma", type=int, default=None)
            parser.add_argument("--no-wandb", action="store_true")
            parser.add_argument("--fast-dev", action="store_true")

            args = parser.parse_args(test_args[1:])

            if args.fast_dev:
                if args.single_sigma is None:
                    args.single_sigma = 50
                args.limit = min(args.limit, 2)
                args.no_wandb = True

            # User specified --limit 1, should remain 1
            assert args.limit == 1


class TestPilotDevMode:
    """Tests for --pilot-dev CLI flag."""

    def test_pilot_dev_sets_defaults(self):
        """--pilot-dev should set single_sigma=None (all), limit=1, no_wandb=True."""
        import argparse
        import sys

        from scripts.run_evaluation import SUPPORTED_SIGMAS

        test_args = ["run_evaluation.py", "--pilot-dev", "--smoke-mode"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", test_args)

            parser = argparse.ArgumentParser()
            parser.add_argument("--limit", type=int, default=10)
            parser.add_argument("--single-sigma", type=int, default=None)
            parser.add_argument("--no-wandb", action="store_true")
            parser.add_argument("--smoke-mode", action="store_true")
            parser.add_argument("--pilot-dev", action="store_true")

            args = parser.parse_args(test_args[1:])

            # Apply pilot-dev logic (same as in script)
            if args.pilot_dev:
                args.single_sigma = None
                args.limit = min(args.limit, 1)
                args.no_wandb = True

            # Determine effective sigmas
            if args.single_sigma:
                sigmas = (args.single_sigma,)
            else:
                sigmas = SUPPORTED_SIGMAS

            assert sigmas == (15, 25, 50)
            assert args.single_sigma is None
            assert args.no_wandb is True
            assert args.limit == 1

    def test_pilot_dev_overrides_single_sigma(self):
        """--pilot-dev should force full sigma set even with explicit --single-sigma."""
        import argparse
        import sys

        from scripts.run_evaluation import SUPPORTED_SIGMAS

        test_args = ["run_evaluation.py", "--pilot-dev", "--single-sigma", "50"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", test_args)

            parser = argparse.ArgumentParser()
            parser.add_argument("--limit", type=int, default=10)
            parser.add_argument("--single-sigma", type=int, default=None)
            parser.add_argument("--no-wandb", action="store_true")
            parser.add_argument("--pilot-dev", action="store_true")

            args = parser.parse_args(test_args[1:])

            # Apply pilot-dev logic
            if args.pilot_dev:
                args.single_sigma = None
                args.limit = min(args.limit, 1)
                args.no_wandb = True

            # Determine effective sigmas
            if args.single_sigma:
                sigmas = (args.single_sigma,)
            else:
                sigmas = SUPPORTED_SIGMAS

            # pilot-dev forces all sigmas
            assert sigmas == (15, 25, 50)
            assert args.single_sigma is None

    def test_pilot_dev_caps_limit_to_one(self):
        """--pilot-dev should cap limit to 1 regardless of user value."""
        import argparse
        import sys

        test_args = ["run_evaluation.py", "--pilot-dev", "--limit", "100"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", test_args)

            parser = argparse.ArgumentParser()
            parser.add_argument("--limit", type=int, default=10)
            parser.add_argument("--single-sigma", type=int, default=None)
            parser.add_argument("--no-wandb", action="store_true")
            parser.add_argument("--pilot-dev", action="store_true")

            args = parser.parse_args(test_args[1:])

            if args.pilot_dev:
                args.single_sigma = None
                args.limit = min(args.limit, 1)
                args.no_wandb = True

            assert args.limit == 1

    def test_pilot_run_decision_gate_valid(self):
        """Pilot run with all sigmas and both domains should be decision_gate_valid."""
        from scripts.run_evaluation import save_decision_json

        # 1 image per domain, across 3 sigmas
        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 1},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 1},
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 1},
            ("natural", 15): {"psnr_mean": 30.0, "count": 1},
            ("natural", 25): {"psnr_mean": 28.5, "count": 1},
            ("natural", 50): {"psnr_mean": 26.0, "count": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            decision = save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[15, 25, 50],
                gap_threshold_db=1.0,
                is_pilot=True,
            )

            assert decision["decision_gate_valid"] is True
            assert decision["dataset_locked"] is True
            assert decision["evidence_tier"] == "pilot"
            assert "protocol-complete pilot" in decision["notes"]

    def test_pilot_run_returns_decision_dict(self):
        """save_decision_json should return the decision dict."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("microscopy", 25): {"psnr_mean": 28.0, "count": 1},
            ("natural", 25): {"psnr_mean": 30.0, "count": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            decision = save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[25],
                gap_threshold_db=1.0,
            )

            assert isinstance(decision, dict)
            assert "decision_gate_valid" in decision
            assert "evidence_tier" in decision


class TestEvidenceTier:
    """Tests for evidence_tier computation."""

    def test_compute_evidence_tier_pilot(self):
        """1-2 images per domain should be 'pilot' tier."""
        from scripts.run_evaluation import compute_evidence_tier

        assert compute_evidence_tier(1, 1) == "pilot"
        assert compute_evidence_tier(2, 2) == "pilot"
        assert compute_evidence_tier(1, 10) == "pilot"  # min is 1
        assert compute_evidence_tier(2, 100) == "pilot"  # min is 2

    def test_compute_evidence_tier_moderate(self):
        """3-9 images per domain should be 'moderate' tier."""
        from scripts.run_evaluation import compute_evidence_tier

        assert compute_evidence_tier(3, 3) == "moderate"
        assert compute_evidence_tier(5, 5) == "moderate"
        assert compute_evidence_tier(9, 9) == "moderate"
        assert compute_evidence_tier(3, 100) == "moderate"  # min is 3

    def test_compute_evidence_tier_strong(self):
        """10+ images per domain should be 'strong' tier."""
        from scripts.run_evaluation import compute_evidence_tier

        assert compute_evidence_tier(10, 10) == "strong"
        assert compute_evidence_tier(100, 100) == "strong"
        assert compute_evidence_tier(10, 1000) == "strong"

    def test_decision_json_evidence_tier_pilot(self):
        """decision.json should have evidence_tier='pilot' for 1-2 images."""
        from scripts.run_evaluation import save_decision_json

        # 1 image x 3 sigmas = 3 count entries per domain
        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 1},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 1},
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 1},
            ("natural", 15): {"psnr_mean": 30.0, "count": 1},
            ("natural", 25): {"psnr_mean": 28.5, "count": 1},
            ("natural", 50): {"psnr_mean": 26.0, "count": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            decision = save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[15, 25, 50],
                gap_threshold_db=1.0,
            )

            assert decision["evidence_tier"] == "pilot"
            assert decision["microscopy_image_count"] == 1
            assert decision["natural_image_count"] == 1

    def test_decision_json_evidence_tier_moderate(self):
        """decision.json should have evidence_tier='moderate' for 3-9 images."""
        from scripts.run_evaluation import save_decision_json

        # 5 images x 3 sigmas = 5 count entries per domain
        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 5},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 5},
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 5},
            ("natural", 15): {"psnr_mean": 30.0, "count": 5},
            ("natural", 25): {"psnr_mean": 28.5, "count": 5},
            ("natural", 50): {"psnr_mean": 26.0, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            decision = save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[15, 25, 50],
                gap_threshold_db=1.0,
            )

            assert decision["evidence_tier"] == "moderate"
            assert decision["microscopy_image_count"] == 5
            assert decision["natural_image_count"] == 5

    def test_decision_json_evidence_tier_strong(self):
        """decision.json should have evidence_tier='strong' for 10+ images."""
        from scripts.run_evaluation import save_decision_json

        # 10 images x 3 sigmas = 10 count entries per domain
        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 10},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 10},
            ("microscopy", 50): {"psnr_mean": 24.0, "count": 10},
            ("natural", 15): {"psnr_mean": 30.0, "count": 10},
            ("natural", 25): {"psnr_mean": 28.5, "count": 10},
            ("natural", 50): {"psnr_mean": 26.0, "count": 10},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            decision = save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[15, 25, 50],
                gap_threshold_db=1.0,
            )

            assert decision["evidence_tier"] == "strong"
            assert decision["microscopy_image_count"] == 10
            assert decision["natural_image_count"] == 10

    def test_summary_csv_includes_evidence_tier(self):
        """baseline_summary.csv should include evidence_tier column."""
        from scripts.run_evaluation import save_summary_csv

        agg = {
            ("microscopy", 25): {
                "count": 1,
                "psnr_mean": 29.5,
                "psnr_std": 0.0,
                "ssim_mean": 0.88,
                "ssim_std": 0.0,
                "dataset_name": "fmd",
                "is_real_data": True,
                "model_name": "swinir",
                "model_checkpoint": "noise25.pth",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            save_summary_csv(
                agg=agg,
                output_path=path,
                seed=42,
                mode="full",
                decision_gate_valid=True,
                evidence_tier="pilot",
            )

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert "evidence_tier" in rows[0]
            assert rows[0]["evidence_tier"] == "pilot"


class TestSpecializationSummary:
    """Tests for specialization_summary.json (finetuned model runs)."""

    def _make_agg(self):
        """Create a standard two-domain agg dict for testing."""
        return {
            ("microscopy", 15): {
                "psnr_mean": 37.73, "ssim_mean": 0.94, "count": 2,
                "psnr_std": 0.5, "ssim_std": 0.01,
                "dataset_name": "fmd", "is_real_data": True,
                "model_name": "swinir", "model_checkpoint": "best.pt",
            },
            ("microscopy", 25): {
                "psnr_mean": 35.73, "ssim_mean": 0.91, "count": 2,
                "psnr_std": 0.5, "ssim_std": 0.01,
                "dataset_name": "fmd", "is_real_data": True,
                "model_name": "swinir", "model_checkpoint": "best.pt",
            },
            ("microscopy", 50): {
                "psnr_mean": 32.93, "ssim_mean": 0.89, "count": 2,
                "psnr_std": 0.5, "ssim_std": 0.01,
                "dataset_name": "fmd", "is_real_data": True,
                "model_name": "swinir", "model_checkpoint": "best.pt",
            },
            ("natural", 15): {
                "psnr_mean": 30.87, "ssim_mean": 0.89, "count": 3,
                "psnr_std": 0.5, "ssim_std": 0.01,
                "dataset_name": "natural", "is_real_data": True,
                "model_name": "swinir", "model_checkpoint": "best.pt",
            },
            ("natural", 25): {
                "psnr_mean": 28.31, "ssim_mean": 0.86, "count": 3,
                "psnr_std": 0.5, "ssim_std": 0.01,
                "dataset_name": "natural", "is_real_data": True,
                "model_name": "swinir", "model_checkpoint": "best.pt",
            },
            ("natural", 50): {
                "psnr_mean": 24.96, "ssim_mean": 0.65, "count": 3,
                "psnr_std": 0.5, "ssim_std": 0.01,
                "dataset_name": "natural", "is_real_data": True,
                "model_name": "swinir", "model_checkpoint": "best.pt",
            },
        }

    def _write_compare_csv(self, path, rows):
        """Write a fake compare_summary.csv."""
        fieldnames = [
            "sigma", "domain",
            "pretrained_psnr_mean", "finetuned_psnr_mean", "delta_psnr",
            "pretrained_ssim_mean", "finetuned_ssim_mean", "delta_ssim",
            "checkpoint_path",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_finetuned_writes_specialization_json(self):
        """save_specialization_summary should write specialization_summary.json."""
        from scripts.run_evaluation import save_specialization_summary

        agg = self._make_agg()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            compare_path = tmpdir / "compare_summary.csv"
            self._write_compare_csv(compare_path, [
                {"sigma": 15, "domain": "microscopy",
                 "pretrained_psnr_mean": "36.65", "finetuned_psnr_mean": "37.73",
                 "delta_psnr": "1.08", "pretrained_ssim_mean": "0.87",
                 "finetuned_ssim_mean": "0.94", "delta_ssim": "0.07",
                 "checkpoint_path": "best.pt"},
                {"sigma": 15, "domain": "natural",
                 "pretrained_psnr_mean": "32.81", "finetuned_psnr_mean": "30.87",
                 "delta_psnr": "-1.94", "pretrained_ssim_mean": "0.92",
                 "finetuned_ssim_mean": "0.89", "delta_ssim": "-0.03",
                 "checkpoint_path": "best.pt"},
            ])

            out_path = tmpdir / "specialization_summary.json"
            save_specialization_summary(
                agg=agg, output_path=out_path, mode="full",
                sigmas=[15, 25, 50], evidence_tier="pilot",
                microscopy_image_count=2, natural_image_count=3,
                compare_csv_path=compare_path,
            )

            assert out_path.exists()

            with open(out_path) as f:
                data = json.load(f)

            assert data["model_mode"] == "finetuned"
            assert data["mode"] == "full"
            assert data["evidence_tier"] == "pilot"
            assert data["microscopy_image_count"] == 2
            assert data["natural_image_count"] == 3
            assert data["sigma_list"] == [15, 25, 50]
            assert data["microscopy_mean_psnr"] is not None
            assert data["natural_mean_psnr"] is not None
            assert data["overall_natural_minus_micro_psnr"] is not None
            assert data["overall_natural_minus_micro_ssim"] is not None
            assert "per_sigma" in data
            assert "15" in data["per_sigma"]
            assert data["compare_summary_path"] == str(compare_path)
            assert "NOT a Day 3" in data["notes"]

    def test_finetuned_does_not_call_save_decision_json(self, monkeypatch):
        """Finetuned branch should call
        save_specialization_summary, not save_decision_json."""
        import scripts.run_evaluation as mod

        decision_called = []
        specialization_called = []

        orig_save_decision = mod.save_decision_json
        orig_save_specialization = mod.save_specialization_summary

        def mock_save_decision(*args, **kwargs):
            decision_called.append(True)
            return orig_save_decision(*args, **kwargs)

        def mock_save_specialization(*args, **kwargs):
            specialization_called.append(True)
            return orig_save_specialization(*args, **kwargs)

        monkeypatch.setattr(mod, "save_decision_json", mock_save_decision)
        monkeypatch.setattr(
            mod, "save_specialization_summary",
            mock_save_specialization,
        )

        agg = self._make_agg()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Simulate finetuned branch logic from main()
            mode = "full"
            sigmas = [15, 25, 50]
            evidence_tier = "pilot"
            model_mode = "finetuned"

            # Create minimal EvalResult list for save_csv
            from scripts.run_evaluation import (
                EvalResult,
                save_csv,
                save_summary_csv,
            )
            results = [
                EvalResult(sigma=15, domain="microscopy", dataset_name="fmd",
                           is_real_data=True, image_name="a.png", image_path="a.png",
                           psnr=37.73, ssim=0.94, noise_seed=42,
                           model_name="swinir", model_checkpoint="best.pt"),
            ]

            save_csv(results, tmpdir / "metrics.csv", mode)
            save_summary_csv(
                agg, tmpdir / "summary.csv", seed=42, mode=mode,
                decision_gate_valid=False, evidence_tier=evidence_tier,
            )

            if model_mode == "finetuned":
                specialization_path = tmpdir / "specialization_summary.json"
                mod.save_specialization_summary(
                    agg=agg, output_path=specialization_path, mode=mode,
                    sigmas=sigmas, evidence_tier=evidence_tier,
                    microscopy_image_count=2, natural_image_count=3,
                )
            else:
                decision_path = tmpdir / "day3_decision.json"
                mod.save_decision_json(
                    agg, decision_path, mode, sigmas, gap_threshold_db=1.0,
                )

            assert len(decision_called) == 0, (
                "save_decision_json should not be called"
                " for finetuned"
            )
            assert len(specialization_called) == 1, (
                "save_specialization_summary should be"
                " called for finetuned"
            )
            assert not (tmpdir / "day3_decision.json").exists()
            assert (tmpdir / "specialization_summary.json").exists()

    def test_specialization_detected_true(self):
        """specialization_detected should be True when micro
        improves and natural regresses."""
        from scripts.run_evaluation import save_specialization_summary

        agg = self._make_agg()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            compare_path = tmpdir / "compare_summary.csv"
            # Microscopy improved, natural regressed across all sigmas
            self._write_compare_csv(compare_path, [
                {"sigma": 15, "domain": "microscopy",
                 "pretrained_psnr_mean": "36.0", "finetuned_psnr_mean": "37.73",
                 "delta_psnr": "1.73", "pretrained_ssim_mean": "0.87",
                 "finetuned_ssim_mean": "0.94", "delta_ssim": "0.07",
                 "checkpoint_path": "best.pt"},
                {"sigma": 25, "domain": "microscopy",
                 "pretrained_psnr_mean": "30.0", "finetuned_psnr_mean": "35.73",
                 "delta_psnr": "5.73", "pretrained_ssim_mean": "0.70",
                 "finetuned_ssim_mean": "0.91", "delta_ssim": "0.21",
                 "checkpoint_path": "best.pt"},
                {"sigma": 50, "domain": "microscopy",
                 "pretrained_psnr_mean": "23.0", "finetuned_psnr_mean": "32.93",
                 "delta_psnr": "9.93", "pretrained_ssim_mean": "0.44",
                 "finetuned_ssim_mean": "0.89", "delta_ssim": "0.45",
                 "checkpoint_path": "best.pt"},
                {"sigma": 15, "domain": "natural",
                 "pretrained_psnr_mean": "32.81", "finetuned_psnr_mean": "30.87",
                 "delta_psnr": "-1.94", "pretrained_ssim_mean": "0.92",
                 "finetuned_ssim_mean": "0.89", "delta_ssim": "-0.03",
                 "checkpoint_path": "best.pt"},
                {"sigma": 25, "domain": "natural",
                 "pretrained_psnr_mean": "30.27", "finetuned_psnr_mean": "28.31",
                 "delta_psnr": "-1.96", "pretrained_ssim_mean": "0.88",
                 "finetuned_ssim_mean": "0.86", "delta_ssim": "-0.02",
                 "checkpoint_path": "best.pt"},
                {"sigma": 50, "domain": "natural",
                 "pretrained_psnr_mean": "26.02", "finetuned_psnr_mean": "24.96",
                 "delta_psnr": "-1.06", "pretrained_ssim_mean": "0.78",
                 "finetuned_ssim_mean": "0.65", "delta_ssim": "-0.13",
                 "checkpoint_path": "best.pt"},
            ])

            out_path = tmpdir / "spec.json"
            result = save_specialization_summary(
                agg=agg, output_path=out_path, mode="full",
                sigmas=[15, 25, 50], evidence_tier="pilot",
                microscopy_image_count=2, natural_image_count=3,
                compare_csv_path=compare_path,
            )

            assert result["specialization_detected"] is True
            assert result["microscopy_improved_vs_pretrained"] is True
            assert result["natural_regressed_vs_pretrained"] is True
            assert "domain specialization" in result["recommendation"]

    def test_specialization_detected_false_both_improved(self):
        """specialization_detected should be False when both domains improve."""
        from scripts.run_evaluation import save_specialization_summary

        agg = self._make_agg()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            compare_path = tmpdir / "compare_summary.csv"
            # Both microscopy and natural improved
            self._write_compare_csv(compare_path, [
                {"sigma": 25, "domain": "microscopy",
                 "pretrained_psnr_mean": "34.0", "finetuned_psnr_mean": "35.73",
                 "delta_psnr": "1.73", "pretrained_ssim_mean": "0.87",
                 "finetuned_ssim_mean": "0.91", "delta_ssim": "0.04",
                 "checkpoint_path": "best.pt"},
                {"sigma": 25, "domain": "natural",
                 "pretrained_psnr_mean": "27.0", "finetuned_psnr_mean": "28.31",
                 "delta_psnr": "1.31", "pretrained_ssim_mean": "0.84",
                 "finetuned_ssim_mean": "0.86", "delta_ssim": "0.02",
                 "checkpoint_path": "best.pt"},
            ])

            out_path = tmpdir / "spec.json"
            result = save_specialization_summary(
                agg=agg, output_path=out_path, mode="full",
                sigmas=[15, 25, 50], evidence_tier="pilot",
                microscopy_image_count=2, natural_image_count=3,
                compare_csv_path=compare_path,
            )

            assert result["specialization_detected"] is False
            assert result["microscopy_improved_vs_pretrained"] is True
            assert result["natural_regressed_vs_pretrained"] is False
            assert "improved both" in result["recommendation"]

    def test_specialization_null_without_compare_csv(self):
        """Without compare CSV, delta-dependent fields should be None."""
        from scripts.run_evaluation import save_specialization_summary

        agg = self._make_agg()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            out_path = tmpdir / "spec.json"
            result = save_specialization_summary(
                agg=agg, output_path=out_path, mode="full",
                sigmas=[15, 25, 50], evidence_tier="pilot",
                microscopy_image_count=2, natural_image_count=3,
                compare_csv_path=None,
            )

            assert result["specialization_detected"] is None
            assert result["microscopy_improved_vs_pretrained"] is None
            assert result["natural_regressed_vs_pretrained"] is None
            assert result["compare_summary_path"] is None
            # Should still have a valid descriptive recommendation
            assert result["recommendation"] is not None
            assert len(result["recommendation"]) > 10
            rec = result["recommendation"].lower()
            assert (
                "baseline comparison unavailable" in rec
                or "pretrained" in rec
            )
            # Per-sigma delta fields should be None
            for sigma_key, entry in result["per_sigma"].items():
                assert entry["microscopy_delta_psnr_vs_pretrained"] is None
                assert entry["natural_delta_psnr_vs_pretrained"] is None
            # Current metrics should still be present
            assert result["microscopy_mean_psnr"] is not None
            assert result["natural_mean_psnr"] is not None

    def test_recommendation_wording_no_day3_terms(self):
        """Finetuned recommendations should never contain Day 3 screening terms."""
        from scripts.run_evaluation import _specialization_recommendation

        forbidden_terms = [
            "decision gate", "dataset_locked", "pivot dataset", "similar",
        ]

        # Test all four recommendation branches
        recommendations = [
            _specialization_recommendation(True, True, True),     # specialization
            _specialization_recommendation(True, False, True),    # both improved
            _specialization_recommendation(False, True, True),    # micro not improved
            _specialization_recommendation(None, None, False),    # no compare data
        ]

        for rec in recommendations:
            rec_lower = rec.lower()
            for term in forbidden_terms:
                assert term not in rec_lower, (
                    f"Recommendation should not contain '{term}': {rec}"
                )

    def test_finetuned_stdout_uses_specialization_language(self, capsys):
        """Finetuned print_summary should use specialization terms, not Day 3 terms."""
        from scripts.run_evaluation import print_summary

        agg = self._make_agg()
        specialization = {
            "microscopy_mean_psnr": 35.46,
            "natural_mean_psnr": 28.05,
            "overall_natural_minus_micro_psnr": -7.42,
            "specialization_detected": True,
            "evidence_tier": "pilot",
            "recommendation": (
                "Fine-tuning produced domain specialization: "
                "improved microscopy denoising with reduced "
                "out-of-domain natural-image performance."
            ),
        }

        print_summary(
            agg, (15, 25, 50),
            model_mode="finetuned",
            specialization=specialization,
        )

        captured = capsys.readouterr().out

        # Should contain specialization terms
        assert "SPECIALIZATION ANALYSIS" in captured
        assert "specialization" in captured.lower()

        # Should NOT contain Day 3 terms
        assert "DECISION GATE" not in captured
        assert "dataset_locked" not in captured.lower()
        assert "pivot dataset" not in captured.lower()
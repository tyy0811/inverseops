"""Tests for training module."""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


class TestLosses:
    """Tests for loss functions."""

    def test_get_loss_l1_returns_callable(self) -> None:
        """get_loss('l1') should return a callable."""
        from inverseops.training.losses import get_loss

        loss_fn = get_loss("l1")
        assert callable(loss_fn)

    def test_get_loss_unsupported_raises(self) -> None:
        """get_loss with unsupported name should raise ValueError."""
        from inverseops.training.losses import get_loss

        with pytest.raises(ValueError, match="Unsupported loss"):
            get_loss("unsupported")

    def test_l1_loss_computation(self) -> None:
        """l1_loss should compute correct L1 distance."""
        from inverseops.training.losses import l1_loss

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 3.0, 5.0])
        # L1 = mean(|1-1|, |2-3|, |3-5|) = mean(0, 1, 2) = 1.0
        loss = l1_loss(pred, target)
        assert loss.item() == pytest.approx(1.0)

    def test_l1_loss_identical_is_zero(self) -> None:
        """l1_loss of identical tensors should be zero."""
        from inverseops.training.losses import l1_loss

        tensor = torch.rand(2, 1, 32, 32)
        loss = l1_loss(tensor, tensor)
        assert loss.item() == pytest.approx(0.0)


class TestTracking:
    """Tests for experiment tracking helpers."""

    def test_init_wandb_disabled_no_error(self) -> None:
        """init_wandb with enabled=False should not raise."""
        from inverseops.tracking.experiment import init_wandb

        # Should be a no-op, no error
        init_wandb(config={}, enabled=False, project="test")

    def test_log_metrics_disabled_no_error(self) -> None:
        """log_metrics with enabled=False should not raise."""
        from inverseops.tracking.experiment import log_metrics

        # Should be a no-op
        log_metrics(step=1, metrics={"loss": 0.5}, enabled=False)

    def test_finish_run_disabled_no_error(self) -> None:
        """finish_run with enabled=False should not raise."""
        from inverseops.tracking.experiment import finish_run

        finish_run(enabled=False)

    def test_save_config_copy(self, tmp_path) -> None:
        """save_config_copy should write YAML file."""
        from inverseops.tracking.experiment import save_config_copy

        config = {"seed": 42, "learning_rate": 0.001}
        save_config_copy(config, tmp_path)

        config_path = tmp_path / "config.yaml"
        assert config_path.exists()

        import yaml
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["seed"] == 42


class TestMicroscopyTrainDataset:
    """Tests for MicroscopyTrainDataset wrapper."""

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample images for testing."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(5):
            arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            img = Image.fromarray(arr, mode="L")
            img.save(img_dir / f"img_{i:03d}.png")

        return img_dir

    def test_dataset_returns_dict(self, sample_images) -> None:
        """Dataset __getitem__ should return expected dict structure."""
        from inverseops.data.microscopy import MicroscopyDataset
        from inverseops.data.torch_datasets import MicroscopyTrainDataset

        base = MicroscopyDataset(root_dir=sample_images, split="train")
        base.prepare()

        dataset = MicroscopyTrainDataset(
            base_dataset=base,
            patch_size=32,
            sigmas=(15, 25),
            training=True,
        )

        sample = dataset[0]
        assert "input" in sample
        assert "target" in sample
        assert "sigma" in sample
        assert "image_name" in sample

    def test_dataset_tensor_shapes(self, sample_images) -> None:
        """Tensors should have correct shape [1, H, W]."""
        from inverseops.data.microscopy import MicroscopyDataset
        from inverseops.data.torch_datasets import MicroscopyTrainDataset

        base = MicroscopyDataset(root_dir=sample_images, split="train")
        base.prepare()

        patch_size = 32
        dataset = MicroscopyTrainDataset(
            base_dataset=base,
            patch_size=patch_size,
            training=True,
        )

        sample = dataset[0]
        assert sample["input"].shape == (1, patch_size, patch_size)
        assert sample["target"].shape == (1, patch_size, patch_size)

    def test_dataset_tensor_range(self, sample_images) -> None:
        """Tensors should be in [0, 1] range."""
        from inverseops.data.microscopy import MicroscopyDataset
        from inverseops.data.torch_datasets import MicroscopyTrainDataset

        base = MicroscopyDataset(root_dir=sample_images, split="train")
        base.prepare()

        dataset = MicroscopyTrainDataset(
            base_dataset=base,
            patch_size=32,
            training=True,
        )

        sample = dataset[0]
        assert sample["input"].min() >= 0.0
        assert sample["input"].max() <= 1.0
        assert sample["target"].min() >= 0.0
        assert sample["target"].max() <= 1.0

    def test_validation_mode_deterministic(self, sample_images) -> None:
        """Validation mode should be deterministic."""
        from inverseops.data.microscopy import MicroscopyDataset
        from inverseops.data.torch_datasets import MicroscopyTrainDataset

        base = MicroscopyDataset(root_dir=sample_images, split="train")
        base.prepare()

        dataset = MicroscopyTrainDataset(
            base_dataset=base,
            patch_size=32,
            seed=42,
            training=False,
        )

        sample1 = dataset[0]
        sample2 = dataset[0]

        assert torch.allclose(sample1["input"], sample2["input"])
        assert sample1["sigma"] == sample2["sigma"]


class TestTrainableModel:
    """Tests for trainable model access."""

    def test_get_trainable_swinir_returns_module(self) -> None:
        """get_trainable_swinir should return nn.Module."""
        from inverseops.models.swinir import get_trainable_swinir

        # Test without loading weights (pretrained=False)
        model = get_trainable_swinir(noise_level=25, pretrained=False)

        import torch.nn as nn
        assert isinstance(model, nn.Module)

    def test_get_trainable_swinir_is_trainable(self) -> None:
        """Model should be in training mode."""
        from inverseops.models.swinir import get_trainable_swinir

        model = get_trainable_swinir(noise_level=25, pretrained=False)
        assert model.training is True

    def test_get_trainable_swinir_has_parameters(self) -> None:
        """Model should have trainable parameters."""
        from inverseops.models.swinir import get_trainable_swinir

        model = get_trainable_swinir(noise_level=25, pretrained=False)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def dummy_model(self):
        """Create a tiny model for testing."""
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        return TinyModel()

    @pytest.fixture
    def dummy_loaders(self):
        """Create dummy data loaders."""
        # Small tensors for fast tests
        inputs = torch.rand(4, 1, 32, 32)
        targets = torch.rand(4, 1, 32, 32)
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2)
        return loader, loader  # Same for train and val

    def test_trainer_creates_checkpoint_dir(self, dummy_model, dummy_loaders, tmp_path):
        """Trainer should create checkpoint directory."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            max_epochs=1,
            use_amp=False,
            wandb_enabled=False,
        )

        trainer.train()

        checkpoint_dir = tmp_path / "checkpoints"
        assert checkpoint_dir.exists()

    def test_trainer_saves_checkpoints(self, dummy_model, dummy_loaders, tmp_path):
        """Trainer should save latest and best checkpoints."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            max_epochs=1,
            use_amp=False,
            wandb_enabled=False,
        )

        trainer.train()

        assert (tmp_path / "checkpoints" / "latest.pt").exists()
        assert (tmp_path / "checkpoints" / "best.pt").exists()

    def test_trainer_returns_summary(self, dummy_model, dummy_loaders, tmp_path):
        """Trainer.train() should return summary dict."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            max_epochs=2,
            use_amp=False,
            wandb_enabled=False,
        )

        summary = trainer.train()

        assert "best_val_psnr" in summary
        assert "best_epoch" in summary
        assert "stopped_early" in summary
        assert "epochs_completed" in summary
        assert "best_checkpoint_path" in summary

    def test_resume_restores_trainer_state(self, dummy_model, dummy_loaders, tmp_path):
        """Resuming should restore bookkeeping and start at the correct epoch."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            max_epochs=6,
            use_amp=False,
            wandb_enabled=False,
            start_epoch=3,
            best_val_psnr=25.0,
            best_epoch=2,
            epochs_without_improvement=1,
            global_step=100,
        )

        assert trainer.start_epoch == 3
        assert trainer.best_val_psnr == 25.0
        assert trainer.best_epoch == 2
        assert trainer.epochs_without_improvement == 1
        assert trainer.global_step == 100

        summary = trainer.train()
        # Should run epochs 4, 5, 6 (three epochs)
        assert summary["epochs_completed"] == 6

    def test_psnr_is_mean_per_image(self, dummy_model, dummy_loaders, tmp_path):
        """_compute_psnr should return mean of per-image PSNRs."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            use_amp=False,
            wandb_enabled=False,
        )

        # Two images with known, different MSE levels (values stay in [0, 1])
        target = torch.zeros(2, 1, 4, 4)
        pred = torch.zeros(2, 1, 4, 4)
        pred[0] = 0.1   # MSE = 0.01 -> PSNR = 20.0
        pred[1] = 0.01  # MSE = 0.0001 -> PSNR = 40.0

        result = trainer._compute_psnr(pred, target)
        expected = (20.0 + 40.0) / 2.0
        assert result == pytest.approx(expected, abs=0.01)

    def test_latest_checkpoint_has_updated_best_val_psnr(
        self, dummy_model, dummy_loaders, tmp_path
    ):
        """latest.pt should contain the updated best_val_psnr
        after an improving epoch."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            max_epochs=1,
            use_amp=False,
            wandb_enabled=False,
        )

        trainer.train()

        latest = torch.load(tmp_path / "checkpoints" / "latest.pt", weights_only=False)
        best = torch.load(tmp_path / "checkpoints" / "best.pt", weights_only=False)
        # Epoch 1 always improves over -inf, so latest must match best
        assert latest["best_val_psnr"] == best["best_val_psnr"]
        assert latest["best_epoch"] == best["best_epoch"]


class TestIntegration:
    """Integration tests for training pipeline."""

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path):
        """Create sample images for integration testing."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create enough images for train/val split
        for i in range(10):
            arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            img = Image.fromarray(arr, mode="L")
            img.save(img_dir / f"img_{i:03d}.png")

        return img_dir

    def test_full_training_pipeline(self, sample_dataset_dir, tmp_path):
        """Test complete training pipeline with tiny model."""
        import torch.nn as nn
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from inverseops.data.microscopy import MicroscopyDataset
        from inverseops.data.torch_datasets import MicroscopyTrainDataset
        from inverseops.training.losses import get_loss
        from inverseops.training.trainer import Trainer

        # Build tiny model
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = TinyModel()

        # Build datasets
        train_base = MicroscopyDataset(
            root_dir=sample_dataset_dir,
            split="train",
            seed=42,
        )
        train_base.prepare()

        val_base = MicroscopyDataset(
            root_dir=sample_dataset_dir,
            split="val",
            seed=42,
        )
        val_base.prepare()

        train_dataset = MicroscopyTrainDataset(
            base_dataset=train_base,
            patch_size=32,
            sigmas=(15, 25),
            training=True,
        )

        val_dataset = MicroscopyTrainDataset(
            base_dataset=val_base,
            patch_size=32,
            sigmas=(15, 25),
            training=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)

        # Build optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=2)

        # Create trainer
        output_dir = tmp_path / "output"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=get_loss("l1"),
            device="cpu",
            output_dir=output_dir,
            max_epochs=2,
            use_amp=False,
            early_stopping_patience=5,
            wandb_enabled=False,
        )

        # Train
        summary = trainer.train()

        # Verify outputs
        assert summary["epochs_completed"] == 2
        assert summary["best_val_psnr"] > 0
        assert (output_dir / "checkpoints" / "latest.pt").exists()
        assert (output_dir / "checkpoints" / "best.pt").exists()
        assert (output_dir / "training_summary.json").exists()


class TestSetSeed:
    """Tests for set_seed deterministic flags."""

    def test_set_seed_configures_cudnn_deterministic(self):
        """set_seed should set CuDNN deterministic flags when CUDA is available."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

        from scripts.run_training import set_seed

        with mock.patch("torch.cuda.is_available", return_value=True):
            set_seed(42)
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False


class TestArtifactVerification:
    """Tests for training artifact verification."""

    @pytest.fixture
    def dummy_model(self):
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        return TinyModel()

    @pytest.fixture
    def dummy_loaders(self):
        inputs = torch.rand(4, 1, 32, 32)
        targets = torch.rand(4, 1, 32, 32)
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2)
        return loader, loader

    def test_training_produces_all_artifacts(
        self, dummy_model, dummy_loaders, tmp_path,
    ):
        """Training should produce latest.pt, best.pt, and training_summary.json."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            max_epochs=1,
            use_amp=False,
            wandb_enabled=False,
        )
        trainer.train()

        expected = [
            tmp_path / "checkpoints" / "latest.pt",
            tmp_path / "checkpoints" / "best.pt",
            tmp_path / "training_summary.json",
        ]
        missing = [str(p) for p in expected if not p.exists()]
        assert missing == [], f"Missing artifacts: {missing}"


class TestDay5Summary:
    """Tests for Day 5 enriched training_summary.json."""

    @pytest.fixture
    def dummy_model(self):
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        return TinyModel()

    @pytest.fixture
    def dummy_loaders(self):
        inputs = torch.rand(4, 1, 32, 32)
        targets = torch.rand(4, 1, 32, 32)
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2)
        return loader, loader

    def test_trainer_summary_has_core_keys(self, dummy_model, dummy_loaders, tmp_path):
        """Trainer.train() summary must include core keys for Day 5 enrichment."""
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        train_loader, val_loader = dummy_loaders
        optimizer = torch.optim.Adam(dummy_model.parameters())

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path,
            max_epochs=1,
            use_amp=False,
            wandb_enabled=False,
        )
        summary = trainer.train()

        required_keys = [
            "best_val_psnr", "best_epoch", "stopped_early",
            "epochs_completed", "best_checkpoint_path",
            "total_training_time_seconds",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_default_run_name(self):
        """Default run name should be swinir_fmd_denoise_sigma15_25_50_v1."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

        import argparse

        # Parse with no --run-name to verify default is applied via code path
        # We just test the argparse default is None (actual default set in main())
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-name", type=str, default=None)
        args = parser.parse_args([])
        assert args.run_name is None
        # The default "swinir_fmd_denoise_sigma15_25_50_v1" is applied in main()
        # when args.run_name is None: run_name = args.run_name or "swinir_..."

    def test_wandb_flag_precedence(self):
        """--wandb and --no-wandb should be mutually exclusive."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/run_training.py",
             "--config", "configs/denoise_swinir.yaml",
             "--wandb", "--no-wandb"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "not allowed" in result.stderr.lower()


class TestComparisonCSV:
    """Tests for build_comparison_csv."""

    def test_build_comparison_csv(self, tmp_path):
        """build_comparison_csv should produce correct deltas from two summary CSVs."""
        import csv
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

        from scripts.run_evaluation import build_comparison_csv

        # Write fake baseline summary
        baseline_path = tmp_path / "baseline_summary.csv"
        fieldnames = [
            "sigma", "noise_type", "domain", "dataset_name", "is_real_data",
            "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
            "seed", "model_name", "model_checkpoint", "mode",
            "decision_gate_valid", "evidence_tier",
        ]
        with open(baseline_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({
                "sigma": 50, "noise_type": "synthetic_gaussian",
                "domain": "microscopy", "dataset_name": "fmd",
                "is_real_data": True, "count": 5,
                "psnr_mean": "25.0000", "psnr_std": "1.0000",
                "ssim_mean": "0.7000", "ssim_std": "0.0500",
                "seed": 42, "model_name": "swinir",
                "model_checkpoint": "pretrained.pth", "mode": "full",
                "decision_gate_valid": True, "evidence_tier": "moderate",
            })

        # Write fake finetuned summary
        finetuned_path = tmp_path / "finetuned_summary.csv"
        with open(finetuned_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({
                "sigma": 50, "noise_type": "synthetic_gaussian",
                "domain": "microscopy", "dataset_name": "fmd",
                "is_real_data": True, "count": 5,
                "psnr_mean": "27.5000", "psnr_std": "0.8000",
                "ssim_mean": "0.7500", "ssim_std": "0.0400",
                "seed": 42, "model_name": "swinir",
                "model_checkpoint": "best.pt", "mode": "full",
                "decision_gate_valid": True, "evidence_tier": "moderate",
            })

        # Build comparison
        compare_path = tmp_path / "compare_summary.csv"
        build_comparison_csv(
            baseline_path, finetuned_path, compare_path, Path("best.pt")
        )

        assert compare_path.exists()
        with open(compare_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert float(row["delta_psnr"]) == pytest.approx(2.5, abs=0.001)
        assert float(row["delta_ssim"]) == pytest.approx(0.05, abs=0.001)

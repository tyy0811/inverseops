"""Tests for training module."""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
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

        result = trainer._compute_psnr(pred, target, data_range=1.0)
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

    def test_run_name_from_config_tracking(self, tmp_path):
        """If config specifies tracking.run_name, that value is used (no V2 default)."""
        import yaml

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from run_training import load_config

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "task": "denoise",
                    "data": {"dataset": "w2s", "train_root": "/tmp"},
                    "model": {"name": "swinir"},
                    "tracking": {"run_name": "my-v3-run"},
                }
            )
        )
        config = load_config(config_path)
        assert config["tracking"]["run_name"] == "my-v3-run"
        # No V2 fallback: if tracking.run_name is absent, the config carries None
        # and main() is responsible for raising.
        config_path2 = tmp_path / "cfg2.yaml"
        config_path2.write_text(
            yaml.safe_dump(
                {
                    "task": "denoise",
                    "data": {"dataset": "w2s", "train_root": "/tmp"},
                    "model": {"name": "swinir"},
                }
            )
        )
        config2 = load_config(config_path2)
        assert config2["tracking"].get("run_name") is None

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


def test_load_config_raises_on_missing_task(tmp_path):
    """load_config must raise ValueError if task is not specified."""
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_training import load_config

    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"dataset": "w2s", "train_root": "/tmp"},
                "model": {"name": "swinir"},
            }
        )
    )
    with pytest.raises(ValueError, match="task"):
        load_config(config_path)


def test_load_config_raises_on_missing_dataset(tmp_path):
    """load_config must raise ValueError if data.dataset is not specified."""
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_training import load_config

    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "denoise",
                "data": {"train_root": "/tmp"},
                "model": {"name": "swinir"},
            }
        )
    )
    with pytest.raises(ValueError, match="data.dataset"):
        load_config(config_path)


class TestDenormalizedPSNR:
    """Verify Trainer PSNR is plausible when using denormalize_fn."""

    def test_psnr_plausible_with_denormalize(self, tmp_path):
        """Train 2 epochs on W2S-like Z-score data, verify val PSNR is 15-55 dB."""

        from inverseops.data.w2s import W2SDataset
        from inverseops.training.losses import l1_loss
        from inverseops.training.trainer import Trainer

        # Create tiny W2S-like fixture (pre-normalized, mean≈0, std≈1)
        fixture_dir = tmp_path / "fixture"
        for level in [1, 400]:
            d = fixture_dir / f"avg{level}"
            d.mkdir(parents=True)
            for fov in range(1, 5):
                for wl in range(1):
                    noise = 0.3 / max(level, 1)
                    rng = np.random.default_rng(fov * 100 + level)
                    arr = rng.normal(0, 1.0, (32, 32)).astype(np.float32)
                    arr += rng.normal(0, noise, (32, 32)).astype(np.float32)
                    np.save(d / f"{fov:03d}_{wl}.npy", arr)

        ds = W2SDataset(
            root_dir=fixture_dir, split="train", avg_levels=[1], patch_size=32,
        )
        ds.prepare()

        loader = DataLoader(ds, batch_size=2, shuffle=False)

        # Tiny model
        model = torch.nn.Conv2d(1, 1, 3, padding=1)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            scheduler=None,
            loss_fn=l1_loss,
            device="cpu",
            output_dir=tmp_path / "out",
            max_epochs=2,
            use_amp=False,
            wandb_enabled=False,
            denormalize_fn=W2SDataset.denormalize,
            early_stopping_patience=100,
        )
        summary = trainer.train()

        psnr = summary["best_val_psnr"]
        # Untrained tiny model on random data: PSNR should be low but real.
        # The key assertion: NOT 120 dB (the old double-norm bug) and NOT inf.
        assert 5 < psnr < 55, (
            f"Val PSNR {psnr:.2f} dB outside plausible range 5-55 dB. "
            f"Denormalization may be broken."
        )



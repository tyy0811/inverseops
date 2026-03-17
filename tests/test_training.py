"""Tests for training module."""

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
        from inverseops.training.trainer import Trainer
        from inverseops.training.losses import l1_loss

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
        from inverseops.training.trainer import Trainer
        from inverseops.training.losses import l1_loss

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
        from inverseops.training.trainer import Trainer
        from inverseops.training.losses import l1_loss

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

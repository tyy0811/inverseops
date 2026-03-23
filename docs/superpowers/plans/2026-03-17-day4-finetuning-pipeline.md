# Day 4: Fine-Tuning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build minimal fine-tuning training loop for SwinIR on microscopy denoising.

**Architecture:** Thin torch Dataset wrapper around existing MicroscopyDataset, simple Trainer class with AMP/early-stopping, optional W&B integration, CLI entrypoint that wires everything together.

**Tech Stack:** PyTorch, wandb (optional), PyYAML, existing inverseops modules.

**Spec:** `docs/superpowers/specs/2026-03-17-day4-finetuning-pipeline-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `inverseops/data/torch_datasets.py` | Create | Torch Dataset wrapper for training |
| `inverseops/training/losses.py` | Create | Loss functions (L1 only) |
| `inverseops/training/trainer.py` | Create | Training loop, checkpointing, early stopping |
| `inverseops/tracking/experiment.py` | Create | W&B helpers (optional tracking) |
| `inverseops/models/swinir.py` | Modify | Add `get_trainable_swinir()` function |
| `inverseops/training/__init__.py` | Modify | Add exports |
| `inverseops/tracking/__init__.py` | Modify | Add exports |
| `inverseops/data/__init__.py` | Modify | Add export for torch dataset |
| `configs/denoise_swinir.yaml` | Modify | Expand with training config |
| `scripts/run_training.py` | Create | CLI entrypoint |
| `tests/test_training.py` | Create | Unit tests |

---

## Task 1: Losses Module

**Files:**
- Create: `inverseops/training/losses.py`
- Test: `tests/test_training.py`

- [ ] **Step 1.1: Create test file with loss function tests**

```python
# tests/test_training.py
"""Tests for training module."""

import pytest
import torch


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
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/test_training.py::TestLosses -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

- [ ] **Step 1.3: Create losses module**

```python
# inverseops/training/losses.py
"""Loss functions for training.

Day 4 supports L1 loss only. SSIM loss deferred to future.
"""

from typing import Callable

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute pixel-wise L1 (MAE) loss.

    Args:
        pred: Predicted tensor of any shape.
        target: Target tensor of same shape as pred.

    Returns:
        Scalar loss tensor.
    """
    return F.l1_loss(pred, target)


def get_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get loss function by name.

    Args:
        name: Loss function name. Only "l1" supported for Day 4.

    Returns:
        Loss function callable.

    Raises:
        ValueError: If loss name is not supported.
    """
    losses = {
        "l1": l1_loss,
    }
    if name not in losses:
        supported = list(losses.keys())
        raise ValueError(f"Unsupported loss: {name}. Supported: {supported}")
    return losses[name]
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest tests/test_training.py::TestLosses -v`
Expected: 4 tests PASS

- [ ] **Step 1.5: Update training __init__.py**

```python
# inverseops/training/__init__.py
"""Training loop and utilities."""

from inverseops.training.losses import get_loss, l1_loss

__all__ = ["get_loss", "l1_loss"]
```

- [ ] **Step 1.6: Commit**

```bash
git add inverseops/training/losses.py inverseops/training/__init__.py tests/test_training.py
git commit -m "feat(training): add L1 loss function

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Tracking Module

**Files:**
- Create: `inverseops/tracking/experiment.py`
- Modify: `inverseops/tracking/__init__.py`
- Test: `tests/test_training.py`

- [ ] **Step 2.1: Add tracking tests to test file**

Append to `tests/test_training.py`:

```python
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
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest tests/test_training.py::TestTracking -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 2.3: Create experiment tracking module**

```python
# inverseops/tracking/experiment.py
"""Experiment tracking helpers.

Provides optional W&B integration. All functions no-op when disabled.
"""

from pathlib import Path

import yaml


def init_wandb(
    config: dict,
    enabled: bool,
    project: str,
    run_name: str | None = None,
) -> None:
    """Initialize W&B run if enabled.

    Args:
        config: Configuration dict to log.
        enabled: Whether W&B is enabled.
        project: W&B project name.
        run_name: Optional run name.
    """
    if not enabled:
        return

    import wandb

    wandb.init(
        project=project,
        name=run_name,
        config=config,
    )


def log_metrics(step: int, metrics: dict, enabled: bool) -> None:
    """Log metrics to W&B if enabled.

    Args:
        step: Current step/iteration.
        metrics: Dict of metric names to values.
        enabled: Whether W&B is enabled.
    """
    if not enabled:
        return

    import wandb

    wandb.log(metrics, step=step)


def finish_run(enabled: bool) -> None:
    """Finish W&B run if enabled.

    Args:
        enabled: Whether W&B is enabled.
    """
    if not enabled:
        return

    import wandb

    wandb.finish()


def save_config_copy(config: dict, output_dir: Path) -> None:
    """Save a copy of the resolved config to output_dir/config.yaml.

    Args:
        config: Configuration dict to save.
        output_dir: Directory to save config to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `pytest tests/test_training.py::TestTracking -v`
Expected: 4 tests PASS

- [ ] **Step 2.5: Update tracking __init__.py**

```python
# inverseops/tracking/__init__.py
"""Experiment tracking integrations."""

from inverseops.tracking.experiment import (
    finish_run,
    init_wandb,
    log_metrics,
    save_config_copy,
)

__all__ = ["init_wandb", "log_metrics", "finish_run", "save_config_copy"]
```

- [ ] **Step 2.6: Commit**

```bash
git add inverseops/tracking/experiment.py inverseops/tracking/__init__.py tests/test_training.py
git commit -m "feat(tracking): add W&B experiment tracking helpers

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Torch Dataset Wrapper

**Files:**
- Create: `inverseops/data/torch_datasets.py`
- Modify: `inverseops/data/__init__.py`
- Test: `tests/test_training.py`

- [ ] **Step 3.1: Add dataset tests**

Append to `tests/test_training.py`:

```python
import numpy as np
from PIL import Image


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
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `pytest tests/test_training.py::TestMicroscopyTrainDataset -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3.3: Create torch dataset wrapper**

```python
# inverseops/data/torch_datasets.py
"""PyTorch Dataset wrappers for training.

Wraps existing dataset classes with training-time transforms.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from inverseops.data.microscopy import MicroscopyDataset


class MicroscopyTrainDataset(Dataset):
    """Torch Dataset wrapper for microscopy training.

    Wraps MicroscopyDataset with:
    - Tensor conversion
    - Random/center patch cropping
    - On-the-fly Gaussian noise generation

    Args:
        base_dataset: Prepared MicroscopyDataset instance.
        patch_size: Size of crops (patches are square).
        sigmas: Noise levels to sample from.
        seed: Random seed for reproducibility.
        training: If True, use random crops and random sigma.
                  If False, use center crops and deterministic behavior.
    """

    def __init__(
        self,
        base_dataset: MicroscopyDataset,
        patch_size: int = 128,
        sigmas: tuple[int, ...] = (15, 25, 50),
        seed: int = 42,
        training: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.sigmas = sigmas
        self.seed = seed
        self.training = training

        # Create RNG for training mode
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        # Load clean image as PIL
        clean_pil = self.base_dataset.load_image(index)
        image_name = self.base_dataset.image_path(index).name

        # Convert to numpy array
        clean_arr = np.array(clean_pil, dtype=np.float32) / 255.0

        # Crop
        if self.training:
            clean_crop = self._random_crop(clean_arr)
            sigma = self._rng.choice(self.sigmas)
            noise_seed = None  # Random noise each time
        else:
            clean_crop = self._center_crop(clean_arr)
            sigma = self.sigmas[index % len(self.sigmas)]
            noise_seed = self.seed + index

        # Add Gaussian noise
        noisy_crop = self._add_noise(clean_crop, sigma, seed=noise_seed)

        # Convert to tensors [1, H, W]
        target = torch.from_numpy(clean_crop).unsqueeze(0)
        input_tensor = torch.from_numpy(noisy_crop).unsqueeze(0)

        return {
            "input": input_tensor,
            "target": target,
            "sigma": int(sigma),
            "image_name": image_name,
        }

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        """Random crop to patch_size x patch_size."""
        h, w = arr.shape
        if h < self.patch_size or w < self.patch_size:
            # Pad if needed
            arr = self._pad_to_size(arr, self.patch_size)
            h, w = arr.shape

        top = self._rng.integers(0, h - self.patch_size + 1)
        left = self._rng.integers(0, w - self.patch_size + 1)
        return arr[top : top + self.patch_size, left : left + self.patch_size]

    def _center_crop(self, arr: np.ndarray) -> np.ndarray:
        """Center crop to patch_size x patch_size."""
        h, w = arr.shape
        if h < self.patch_size or w < self.patch_size:
            arr = self._pad_to_size(arr, self.patch_size)
            h, w = arr.shape

        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        return arr[top : top + self.patch_size, left : left + self.patch_size]

    def _pad_to_size(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Pad array to at least size x size using reflection."""
        h, w = arr.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(
                arr,
                ((0, pad_h), (0, pad_w)),
                mode="reflect",
            )
        return arr

    def _add_noise(
        self, arr: np.ndarray, sigma: float, seed: int | None = None
    ) -> np.ndarray:
        """Add Gaussian noise to normalized [0, 1] array."""
        rng = np.random.default_rng(seed)
        # sigma is in [0, 255] scale, normalize to [0, 1]
        sigma_normalized = sigma / 255.0
        noise = rng.normal(0, sigma_normalized, arr.shape).astype(np.float32)
        noisy = arr + noise
        return np.clip(noisy, 0, 1).astype(np.float32)
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `pytest tests/test_training.py::TestMicroscopyTrainDataset -v`
Expected: 5 tests PASS

- [ ] **Step 3.5: Update data __init__.py**

Read current file first, then append export:

```python
# inverseops/data/__init__.py
# Add to existing file:
from inverseops.data.torch_datasets import MicroscopyTrainDataset

# Update __all__ if present to include MicroscopyTrainDataset
```

- [ ] **Step 3.6: Commit**

```bash
git add inverseops/data/torch_datasets.py inverseops/data/__init__.py tests/test_training.py
git commit -m "feat(data): add MicroscopyTrainDataset torch wrapper

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Model Extension

**Files:**
- Modify: `inverseops/models/swinir.py`
- Test: `tests/test_training.py`

- [ ] **Step 4.1: Add model test**

Append to `tests/test_training.py`:

```python
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
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `pytest tests/test_training.py::TestTrainableModel -v`
Expected: FAIL with "ImportError" or "AttributeError"

- [ ] **Step 4.3: Add get_trainable_swinir to swinir.py**

Append to `inverseops/models/swinir.py`:

```python
def get_trainable_swinir(
    noise_level: int = 25,
    pretrained: bool = True,
    device: str | None = None,
    cache_dir: Path | str | None = None,
) -> SwinIR:
    """Return trainable SwinIR model for grayscale denoising.

    Args:
        noise_level: Target noise level (15, 25, or 50).
        pretrained: If True, load pretrained weights.
        device: Device to place model on. None for auto.
        cache_dir: Directory for cached weights.

    Returns:
        SwinIR nn.Module in training mode.
    """
    if noise_level not in MODEL_URLS:
        raise ValueError(f"noise_level must be 15, 25, or 50, got {noise_level}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    # Build model with grayscale denoising architecture
    model = SwinIR(
        upscale=1,
        in_chans=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )

    if pretrained:
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = MODEL_URLS[noise_level]
        filename = Path(url).name
        weight_path = cache_dir / filename

        if not weight_path.exists():
            import urllib.request
            print(f"Downloading SwinIR weights from {url}...")
            urllib.request.urlretrieve(url, weight_path)
            print(f"Saved to {weight_path}")

        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        state_dict = state_dict.get("params", state_dict)
        model.load_state_dict(state_dict, strict=True)

    model.train()
    model.to(device)
    return model
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `pytest tests/test_training.py::TestTrainableModel -v`
Expected: 3 tests PASS

- [ ] **Step 4.5: Commit**

```bash
git add inverseops/models/swinir.py tests/test_training.py
git commit -m "feat(models): add get_trainable_swinir for training

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Trainer Class

**Files:**
- Create: `inverseops/training/trainer.py`
- Modify: `inverseops/training/__init__.py`
- Test: `tests/test_training.py`

- [ ] **Step 5.1: Add trainer tests**

Append to `tests/test_training.py`:

```python
from torch.utils.data import DataLoader, TensorDataset


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
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `pytest tests/test_training.py::TestTrainer -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 5.3: Create trainer module**

```python
# inverseops/training/trainer.py
"""Training loop for image restoration models."""

import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from inverseops.tracking.experiment import log_metrics


class Trainer:
    """Training loop with checkpointing and early stopping.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer instance.
        scheduler: Optional learning rate scheduler.
        loss_fn: Loss function callable.
        device: Device to train on.
        output_dir: Directory for checkpoints and logs.
        use_amp: Whether to use automatic mixed precision.
        max_epochs: Maximum number of epochs.
        early_stopping_patience: Epochs to wait for improvement.
        log_every_n_steps: Log training loss every N steps.
        wandb_enabled: Whether W&B logging is enabled.
        config: Optional config dict to save in checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str,
        output_dir: Path,
        use_amp: bool = True,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        log_every_n_steps: int = 100,
        wandb_enabled: bool = False,
        config: dict | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.log_every_n_steps = log_every_n_steps
        self.wandb_enabled = wandb_enabled
        self.config = config or {}

        # AMP setup - only use on CUDA
        self.use_amp = use_amp and device != "cpu" and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_val_psnr = float("-inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.global_step = 0

    def train(self) -> dict:
        """Run training loop.

        Returns:
            Summary dict with training results.
        """
        start_time = time.time()
        stopped_early = False

        for epoch in range(1, self.max_epochs + 1):
            # Training
            train_loss = self._train_epoch(epoch)

            # Validation
            val_loss, val_psnr = self._validate_epoch()

            # Get current learning rate
            lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/psnr": val_psnr,
                "learning_rate": lr,
            }
            log_metrics(step=epoch, metrics=metrics, enabled=self.wandb_enabled)

            # Print epoch summary
            is_best = val_psnr > self.best_val_psnr
            best_marker = " [best]" if is_best else ""
            print(
                f"Epoch {epoch}/{self.max_epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_psnr: {val_psnr:.2f} dB | "
                f"lr: {lr:.2e}{best_marker}"
            )

            # Save checkpoints
            self._save_checkpoint(epoch, val_psnr, is_latest=True)

            if is_best:
                self.best_val_psnr = val_psnr
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_psnr, is_latest=False)
            else:
                self.epochs_without_improvement += 1

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"Early stopping: no improvement for "
                    f"{self.early_stopping_patience} epochs"
                )
                stopped_early = True
                break

        total_time = time.time() - start_time
        epochs_completed = epoch

        # Print final summary
        print(f"\nTraining complete. Best PSNR: {self.best_val_psnr:.2f} dB at epoch {self.best_epoch}")

        # Create summary
        best_checkpoint_path = str(self.checkpoint_dir / "best.pt")
        summary = {
            "best_val_psnr": self.best_val_psnr,
            "best_epoch": self.best_epoch,
            "stopped_early": stopped_early,
            "epochs_completed": epochs_completed,
            "best_checkpoint_path": best_checkpoint_path,
            "total_training_time_seconds": total_time,
        }

        # Save summary JSON
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both dict batches and tuple batches
            if isinstance(batch, dict):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log every N steps
            if self.global_step % self.log_every_n_steps == 0:
                log_metrics(
                    step=self.global_step,
                    metrics={"train/step_loss": loss.item()},
                    enabled=self.wandb_enabled,
                )

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def _validate_epoch(self) -> tuple[float, float]:
        """Validate model.

        Returns:
            Tuple of (mean_loss, mean_psnr).
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if isinstance(batch, dict):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Compute PSNR
            psnr = self._compute_psnr(outputs, targets)

            total_loss += loss.item()
            total_psnr += psnr
            num_batches += 1

        mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
        mean_psnr = total_psnr / num_batches if num_batches > 0 else 0.0

        return mean_loss, mean_psnr

    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR for tensors in [0, 1] range.

        Args:
            pred: Predicted tensor [B, C, H, W].
            target: Target tensor [B, C, H, W].

        Returns:
            Mean PSNR across batch.
        """
        pred = pred.clamp(0, 1)
        mse = torch.nn.functional.mse_loss(pred, target)
        if mse == 0:
            return float("inf")
        return (10 * torch.log10(1.0 / mse)).item()

    def _save_checkpoint(self, epoch: int, val_psnr: float, is_latest: bool) -> None:
        """Save checkpoint.

        Args:
            epoch: Current epoch.
            val_psnr: Validation PSNR.
            is_latest: If True, save as latest.pt, else save as best.pt.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_psnr": self.best_val_psnr,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        filename = "latest.pt" if is_latest else "best.pt"
        torch.save(checkpoint, self.checkpoint_dir / filename)
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `pytest tests/test_training.py::TestTrainer -v`
Expected: 3 tests PASS

- [ ] **Step 5.5: Update training __init__.py**

```python
# inverseops/training/__init__.py
"""Training loop and utilities."""

from inverseops.training.losses import get_loss, l1_loss
from inverseops.training.trainer import Trainer

__all__ = ["get_loss", "l1_loss", "Trainer"]
```

- [ ] **Step 5.6: Commit**

```bash
git add inverseops/training/trainer.py inverseops/training/__init__.py tests/test_training.py
git commit -m "feat(training): add Trainer class with early stopping and AMP

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Config

**Files:**
- Modify: `configs/denoise_swinir.yaml`

- [ ] **Step 6.1: Read current config**

Run: `cat configs/denoise_swinir.yaml`

- [ ] **Step 6.2: Expand config with training settings**

```yaml
# configs/denoise_swinir.yaml
# Day 4: SwinIR denoising configuration

seed: 42
task: denoise

data:
  train_root: data/raw/fmd
  val_root: data/raw/fmd
  batch_size: 4
  num_workers: 2
  patch_size: 128
  sigmas: [15, 25, 50]

model:
  name: swinir
  pretrained: true
  noise_level: 25

training:
  epochs: 100
  learning_rate: 2.0e-4
  weight_decay: 0.0
  loss: l1
  amp: true
  early_stopping_patience: 10
  log_every_n_steps: 100

scheduler:
  name: cosine
  t_max: 100
  min_lr: 1.0e-6

output_dir: outputs/training

tracking:
  wandb_project: inverseops-training
  enabled: false
```

- [ ] **Step 6.3: Commit**

```bash
git add configs/denoise_swinir.yaml
git commit -m "config: expand denoise_swinir.yaml for training

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: CLI Entrypoint

**Files:**
- Create: `scripts/run_training.py`
- Test: Integration test (run script with smoke data)

- [ ] **Step 7.1: Create run_training.py**

```python
#!/usr/bin/env python3
"""Day 4: Training script for SwinIR microscopy denoising.

Usage:
    python scripts/run_training.py --config configs/denoise_swinir.yaml
    python scripts/run_training.py --config configs/denoise_swinir.yaml --no-wandb
    python scripts/run_training.py --config configs/denoise_swinir.yaml --resume outputs/training/checkpoints/latest.pt
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from inverseops.data.microscopy import MicroscopyDataset
from inverseops.data.torch_datasets import MicroscopyTrainDataset
from inverseops.models.swinir import get_trainable_swinir
from inverseops.tracking.experiment import (
    finish_run,
    init_wandb,
    save_config_copy,
)
from inverseops.training.losses import get_loss
from inverseops.training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    """Load YAML config with defaults."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply defaults
    config.setdefault("seed", 42)
    config.setdefault("data", {})
    config["data"].setdefault("batch_size", 4)
    config["data"].setdefault("num_workers", 0)
    config["data"].setdefault("patch_size", 128)
    config["data"].setdefault("sigmas", [15, 25, 50])

    config.setdefault("model", {})
    config["model"].setdefault("pretrained", True)
    config["model"].setdefault("noise_level", 25)

    config.setdefault("training", {})
    config["training"].setdefault("epochs", 100)
    config["training"].setdefault("learning_rate", 2e-4)
    config["training"].setdefault("weight_decay", 0.0)
    config["training"].setdefault("loss", "l1")
    config["training"].setdefault("amp", True)
    config["training"].setdefault("early_stopping_patience", 10)
    config["training"].setdefault("log_every_n_steps", 100)

    config.setdefault("scheduler", {})
    config["scheduler"].setdefault("name", "cosine")
    config["scheduler"].setdefault("t_max", config["training"]["epochs"])
    config["scheduler"].setdefault("min_lr", 1e-6)

    config.setdefault("output_dir", "outputs/training")

    config.setdefault("tracking", {})
    config["tracking"].setdefault("enabled", False)
    config["tracking"].setdefault("wandb_project", "inverseops-training")

    return config


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    data_cfg = config["data"]
    seed = config["seed"]

    # Build train dataset
    train_base = MicroscopyDataset(
        root_dir=data_cfg["train_root"],
        split="train",
        seed=seed,
    )
    train_base.prepare()

    train_dataset = MicroscopyTrainDataset(
        base_dataset=train_base,
        patch_size=data_cfg["patch_size"],
        sigmas=tuple(data_cfg["sigmas"]),
        seed=seed,
        training=True,
    )

    # Build validation dataset
    val_base = MicroscopyDataset(
        root_dir=data_cfg["val_root"],
        split="val",
        seed=seed,
    )
    val_base.prepare()

    val_dataset = MicroscopyTrainDataset(
        base_dataset=val_base,
        patch_size=data_cfg["patch_size"],
        sigmas=tuple(data_cfg["sigmas"]),
        seed=seed,
        training=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    return train_loader, val_loader


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train SwinIR for microscopy denoising."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    config = load_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.no_wandb:
        config["tracking"]["enabled"] = False

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(config["seed"])
    print(f"Seed: {config['seed']}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Save config copy
    save_config_copy(config, output_dir)

    # Build data loaders
    print("\nBuilding datasets...")
    train_loader, val_loader = build_dataloaders(config)

    # Build model
    print("\nBuilding model...")
    model_cfg = config["model"]
    model = get_trainable_swinir(
        noise_level=model_cfg["noise_level"],
        pretrained=model_cfg["pretrained"],
        device=device,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build optimizer
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Build scheduler
    sched_cfg = config["scheduler"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["t_max"],
        eta_min=sched_cfg["min_lr"],
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if not args.resume.exists():
            print(f"Error: Checkpoint not found: {args.resume}")
            return 1

        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Get loss function
    loss_fn = get_loss(train_cfg["loss"])

    # Initialize W&B
    wandb_enabled = config["tracking"]["enabled"]
    init_wandb(
        config=config,
        enabled=wandb_enabled,
        project=config["tracking"]["wandb_project"],
        run_name=args.run_name,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        use_amp=train_cfg["amp"],
        max_epochs=train_cfg["epochs"],
        early_stopping_patience=train_cfg["early_stopping_patience"],
        log_every_n_steps=train_cfg["log_every_n_steps"],
        wandb_enabled=wandb_enabled,
        config=config,
    )

    # Train
    print("\nStarting training...")
    print("=" * 70)
    summary = trainer.train()
    print("=" * 70)

    # Finish W&B
    finish_run(enabled=wandb_enabled)

    # Print summary
    print(f"\nCheckpoints saved to: {output_dir / 'checkpoints'}")
    print(f"Training summary saved to: {output_dir / 'training_summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.2: Make script executable**

Run: `chmod +x scripts/run_training.py`

- [ ] **Step 7.3: Verify script syntax**

Run: `python -m py_compile scripts/run_training.py`
Expected: No output (no syntax errors)

- [ ] **Step 7.4: Commit**

```bash
git add scripts/run_training.py
git commit -m "feat(scripts): add run_training.py CLI entrypoint

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integration Test

**Files:**
- Test: `tests/test_training.py` (add integration test)

- [ ] **Step 8.1: Add integration test**

Append to `tests/test_training.py`:

```python
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
```

- [ ] **Step 8.2: Run integration test**

Run: `pytest tests/test_training.py::TestIntegration -v`
Expected: PASS

- [ ] **Step 8.3: Run all training tests**

Run: `pytest tests/test_training.py -v`
Expected: All tests PASS

- [ ] **Step 8.4: Commit**

```bash
git add tests/test_training.py
git commit -m "test(training): add integration test for full pipeline

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Final Verification

- [ ] **Step 9.1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 9.2: Verify script runs with smoke data**

Run (with sample images):
```bash
python scripts/run_training.py \
    --config configs/denoise_swinir.yaml \
    --output-dir outputs/smoke_test \
    --no-wandb
```

Expected:
- Script starts without errors
- Epoch summaries printed
- Checkpoints created in `outputs/smoke_test/checkpoints/`
- `training_summary.json` created

- [ ] **Step 9.3: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address any final issues from verification

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

**Files created:**
- `inverseops/data/torch_datasets.py`
- `inverseops/training/losses.py`
- `inverseops/training/trainer.py`
- `inverseops/tracking/experiment.py`
- `scripts/run_training.py`
- `tests/test_training.py`

**Files modified:**
- `inverseops/models/swinir.py` (added `get_trainable_swinir`)
- `inverseops/training/__init__.py`
- `inverseops/tracking/__init__.py`
- `inverseops/data/__init__.py`
- `configs/denoise_swinir.yaml`

**Launch training:**
```bash
python scripts/run_training.py --config configs/denoise_swinir.yaml
```

**Checkpoints saved to:** `{output_dir}/checkpoints/`

**W&B:** Disabled by default (`tracking.enabled: false` in config). Enable with `tracking.enabled: true` or remove `--no-wandb` flag.

# Day 4: Minimal Fine-Tuning Pipeline Design

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Build minimal fine-tuning training loop for SwinIR on microscopy denoising

## Overview

Day 4 adds training capabilities to InverseOps. The goal is a minimal, working fine-tuning pipeline that:

- Trains SwinIR on microscopy images with synthetic Gaussian noise
- Supports AMP, early stopping, and checkpoint saving
- Integrates with W&B for experiment tracking (optional)
- Maintains consistency with existing codebase patterns

### Non-Goals (Explicitly Out of Scope)

- SSIM loss
- ONNX export
- Serving logic
- Additional models
- Multi-GPU / DDP
- Mixed task support beyond denoising
- Complex callback frameworks
- MLflow integration

## File Structure

```
inverseops/
├── data/
│   └── torch_datasets.py      # NEW: MicroscopyTrainDataset wrapper
├── training/
│   ├── __init__.py            # UPDATE: exports
│   ├── losses.py              # NEW: l1_loss, get_loss
│   └── trainer.py             # NEW: Trainer class
├── tracking/
│   ├── __init__.py            # UPDATE: exports
│   └── experiment.py          # NEW: W&B helpers
└── models/
    └── swinir.py              # UPDATE: add get_trainable_swinir()

scripts/
└── run_training.py            # NEW: CLI entrypoint

configs/
└── denoise_swinir.yaml        # UPDATE: expand for training

tests/
└── test_training.py           # NEW: minimal tests
```

## Component Specifications

### 1. Data Pipeline (`inverseops/data/torch_datasets.py`)

#### Class: `MicroscopyTrainDataset`

A thin torch Dataset wrapper around the existing `MicroscopyDataset` class.

```python
class MicroscopyTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: MicroscopyDataset,
        patch_size: int = 128,
        sigmas: tuple[int, ...] = (15, 25, 50),
        seed: int = 42,
        training: bool = True,
    ) -> None:
        """
        Args:
            base_dataset: Prepared MicroscopyDataset instance.
            patch_size: Size of random/center crops.
            sigmas: Noise levels to sample from.
            seed: Random seed for reproducibility.
            training: If True, use random crops and random sigma.
                      If False, use center crops and deterministic behavior.
        """
```

#### Returns

Each `__getitem__` returns a dict:

```python
{
    "input": torch.Tensor,    # Noisy image [1, H, W] in [0, 1]
    "target": torch.Tensor,   # Clean image [1, H, W] in [0, 1]
    "sigma": int,             # Noise level used
    "image_name": str,        # Source image filename
}
```

#### Behavior

**Training mode (`training=True`):**
- Random crop to `patch_size x patch_size`
- Random sigma sampled uniformly from `sigmas` list
- On-the-fly Gaussian noise generation with random seed

**Validation mode (`training=False`):**
- Center crop to `patch_size x patch_size`
- Deterministic sigma cycling (sigma = sigmas[index % len(sigmas)])
- Deterministic noise seed per sample (seed + index)

#### Design Rationale

Wrapping `MicroscopyDataset` rather than creating a standalone loader:
- Preserves existing train/val/test split logic from Day 2
- Keeps responsibilities separated (discovery vs. training transforms)
- Reduces risk of semantic drift between evaluation and training pipelines

---

### 2. Losses (`inverseops/training/losses.py`)

#### Functions

```python
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute pixel-wise L1 loss.

    Args:
        pred: Predicted tensor of shape [B, C, H, W].
        target: Target tensor of shape [B, C, H, W].

    Returns:
        Scalar loss tensor.
    """

def get_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get loss function by name.

    Args:
        name: Loss function name. Only "l1" supported for Day 4.

    Returns:
        Loss function callable.

    Raises:
        ValueError: If loss name is not supported.
    """
```

#### Supported Losses

| Name | Description |
|------|-------------|
| `l1` | Pixel-wise L1 (MAE) loss |

SSIM loss is explicitly deferred to a future day.

---

### 3. Tracking (`inverseops/tracking/experiment.py`)

#### Functions

```python
def init_wandb(
    config: dict,
    enabled: bool,
    project: str,
    run_name: str | None = None,
) -> None:
    """Initialize W&B run if enabled, no-op otherwise."""

def log_metrics(step: int, metrics: dict, enabled: bool) -> None:
    """Log metrics to W&B if enabled, no-op otherwise.

    Expected metrics keys:
        - train/loss
        - val/psnr
        - val/loss
        - learning_rate
        - epoch
    """

def finish_run(enabled: bool) -> None:
    """Finish W&B run if enabled, no-op otherwise."""

def save_config_copy(config: dict, output_dir: Path) -> None:
    """Save a copy of the resolved config to output_dir/config.yaml."""
```

#### Design Principles

- W&B is strictly optional; all functions accept `enabled` parameter
- When `enabled=False`, functions return immediately without side effects
- No MLflow or other tracking backends for Day 4

---

### 4. Trainer (`inverseops/training/trainer.py`)

#### Class: `Trainer`

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
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
```

#### Core Method: `train()`

```python
def train(self) -> dict:
    """Run training loop.

    Returns:
        Summary dict with keys:
            - best_val_psnr: float
            - best_epoch: int
            - stopped_early: bool
            - epochs_completed: int
            - best_checkpoint_path: str
    """
```

#### Training Loop Behavior

1. **Per-epoch training:**
   - Iterate over train_loader
   - Forward pass with optional AMP autocast
   - Compute L1 loss
   - Backward pass with GradScaler if AMP enabled
   - Log train/loss every `log_every_n_steps`

2. **Per-epoch validation:**
   - Compute mean validation loss
   - Compute mean PSNR (clamping predictions to [0, 1])
   - Log val/loss, val/psnr, learning_rate, epoch

3. **Checkpointing:**
   - Save `latest.pt` every epoch
   - Save `best.pt` when validation PSNR improves

4. **Early stopping:**
   - Monitor validation PSNR
   - Stop when no improvement for `patience` epochs

#### Checkpoint Format

```python
{
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict | None,
    "epoch": int,
    "best_val_psnr": float,
    "config": dict,
    "scaler_state_dict": dict | None,  # If AMP used
}
```

#### PSNR Computation

Validation PSNR uses torch tensors directly:

```python
def compute_psnr_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR for tensors in [0, 1] range.

    Assumes inputs are [B, C, H, W] tensors.
    Returns mean PSNR across batch.
    """
    pred = pred.clamp(0, 1)
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()
```

This is consistent with the existing numpy PSNR formula but operates on normalized [0, 1] tensors.

#### AMP Handling

- Use `torch.amp.autocast('cuda')` and `GradScaler` when CUDA available and `use_amp=True`
- Fall back gracefully on CPU (AMP disabled)
- Save/restore scaler state in checkpoints

---

### 5. Model Access (`inverseops/models/swinir.py`)

#### New Function

```python
def get_trainable_swinir(
    noise_level: int = 25,
    pretrained: bool = True,
    device: str | None = None,
    cache_dir: Path | str | None = None,
) -> nn.Module:
    """Return trainable SwinIR model for grayscale denoising.

    Args:
        noise_level: Target noise level (15, 25, or 50).
        pretrained: If True, load pretrained weights.
        device: Device to place model on.
        cache_dir: Directory for cached weights.

    Returns:
        SwinIR nn.Module in training mode.
    """
```

#### Design Rationale

- Reuses existing architecture definition from `_swinir_arch.py`
- Reuses weight download/caching logic from `SwinIRBaseline`
- Does not modify `SwinIRBaseline` class behavior
- Returns model in training mode (not eval mode)

---

### 6. Configuration (`configs/denoise_swinir.yaml`)

#### Full Schema

```yaml
# Project metadata
seed: 42
task: denoise

# Data configuration
data:
  train_root: data/raw/fmd
  val_root: data/raw/fmd
  batch_size: 4
  num_workers: 2
  patch_size: 128
  sigmas: [15, 25, 50]

# Model configuration
model:
  name: swinir
  pretrained: true

# Training configuration
training:
  epochs: 100
  learning_rate: 2e-4
  weight_decay: 0.0
  loss: l1
  amp: true
  early_stopping_patience: 10
  log_every_n_steps: 100

# Learning rate scheduler
scheduler:
  name: cosine
  t_max: 100
  min_lr: 1e-6

# Output configuration
output_dir: outputs/training

# Experiment tracking
tracking:
  wandb_project: inverseops-training
  enabled: true
```

#### Defaults Handling

The training script applies sensible defaults for missing keys:
- `seed`: 42
- `data.batch_size`: 4
- `data.num_workers`: 0
- `training.amp`: True
- `tracking.enabled`: False

---

### 7. CLI Entrypoint (`scripts/run_training.py`)

#### Arguments

```
python scripts/run_training.py \
    --config PATH           # Required: YAML config file
    [--output-dir PATH]     # Optional: override output directory
    [--no-wandb]            # Disable W&B logging
    [--run-name NAME]       # W&B run name
    [--resume PATH]         # Resume from checkpoint
```

#### Execution Flow

1. Parse CLI arguments
2. Load and merge YAML config with CLI overrides
3. Set deterministic seeds:
   - `random.seed(seed)`
   - `np.random.seed(seed)`
   - `torch.manual_seed(seed)`
   - `torch.cuda.manual_seed_all(seed)`
4. Build train/val datasets:
   - Create `MicroscopyDataset` for train and val splits
   - Wrap with `MicroscopyTrainDataset`
   - Create DataLoaders
5. Build model via `get_trainable_swinir()`
6. Create optimizer (AdamW) and scheduler (CosineAnnealingLR)
7. Initialize W&B if enabled
8. Optionally load checkpoint if `--resume`
9. Run training via `Trainer.train()`
10. Save `training_summary.json`
11. Print final summary to stdout

#### Stdout Output

Concise epoch summaries:

```
Epoch 1/100 | train_loss: 0.0234 | val_loss: 0.0198 | val_psnr: 28.45 dB | lr: 2.00e-04
Epoch 2/100 | train_loss: 0.0187 | val_loss: 0.0165 | val_psnr: 29.12 dB | lr: 1.99e-04 [best]
...
Training complete. Best PSNR: 31.24 dB at epoch 47
```

---

### 8. Output Artifacts

```
{output_dir}/
├── checkpoints/
│   ├── latest.pt          # Latest epoch checkpoint
│   └── best.pt            # Best validation PSNR checkpoint
├── training_summary.json  # Final training summary
└── config.yaml            # Resolved config copy
```

#### `training_summary.json` Format

```json
{
  "best_val_psnr": 31.24,
  "best_epoch": 47,
  "stopped_early": true,
  "epochs_completed": 57,
  "best_checkpoint_path": "outputs/training/checkpoints/best.pt",
  "total_training_time_seconds": 3847.2
}
```

---

### 9. Tests (`tests/test_training.py`)

#### Test Cases

1. **Loss function tests:**
   - `test_get_loss_l1_returns_callable` - verify `get_loss("l1")` works
   - `test_get_loss_unsupported_raises` - verify `ValueError` for unknown losses

2. **Checkpoint tests:**
   - `test_checkpoint_dir_created` - verify output directories created

3. **Trainer smoke test:**
   - `test_trainer_one_epoch_dummy_data` - run 1 epoch on small random tensors
   - No weight downloads required
   - Uses tiny model or mocked forward pass

#### Test Principles

- No network dependencies (no weight downloads)
- Fast execution (< 5 seconds total)
- Uses pytest fixtures for temp directories

---

## Reproducibility

### Seed Setting

```python
def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Deterministic Flags

When practical, enable:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

Note: Full determinism may impact performance and is optional.

---

## Error Handling

### Data Errors
- Missing `train_root` or `val_root`: clear error message with path
- Empty dataset split: raise `ValueError` with split name and count

### Training Errors
- CUDA OOM: not caught (let it fail fast)
- Invalid loss name: `ValueError` with supported options

### Checkpoint Errors
- Resume file not found: `FileNotFoundError` with path
- Incompatible checkpoint: `RuntimeError` with details

---

## Definition of Done

The following must work:

```bash
# Start training
python scripts/run_training.py --config configs/denoise_swinir.yaml

# Expected behavior:
# - Checkpoints written to output_dir/checkpoints/
# - Validation PSNR computed and printed each epoch
# - W&B logs training loss and validation PSNR (when enabled)
# - Training completes or stops early
# - training_summary.json written
```

Tests must pass:
```bash
pytest tests/test_training.py -v
```

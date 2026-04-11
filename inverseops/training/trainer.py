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
        denormalize_fn: Callable to reverse dataset normalization before PSNR.
            If None, PSNR is computed on raw model output (assumes [0,1] range).
        data_range: Peak signal value for PSNR/SSIM (255.0 for W2S, 1.0 for IXI).
            Must match the range of denormalized data. If denormalize_fn is None,
            defaults to 1.0 (raw [0,1] output).
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
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        data_range: float = 1.0,
        start_epoch: int = 0,
        best_val_psnr: float = float("-inf"),
        best_epoch: int = 0,
        epochs_without_improvement: int = 0,
        global_step: int = 0,
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
        self.denormalize_fn = denormalize_fn
        self.data_range = data_range

        # AMP setup - only use on CUDA
        self.use_amp = use_amp and device != "cpu" and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None  # type: ignore[attr-defined]

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state (restored from checkpoint on resume)
        self.start_epoch = start_epoch
        self.best_val_psnr = best_val_psnr
        self.best_epoch = best_epoch
        self.epochs_without_improvement = epochs_without_improvement
        self.global_step = global_step

    def train(self) -> dict:
        """Run training loop.

        Returns:
            Summary dict with training results.
        """
        start_time = time.time()
        stopped_early = False

        for epoch in range(self.start_epoch + 1, self.max_epochs + 1):
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

            # Update best tracking before saving so latest.pt has fresh state
            if is_best:
                self.best_val_psnr = val_psnr
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                log_metrics(
                    step=epoch,
                    metrics={"best_val_psnr": self.best_val_psnr},
                    enabled=self.wandb_enabled,
                )
                self._save_checkpoint(epoch, val_psnr, is_latest=False)
            else:
                self.epochs_without_improvement += 1

            # Save latest checkpoint (state already updated)
            self._save_checkpoint(epoch, val_psnr, is_latest=True)

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
        print(
            f"\nTraining complete. Best PSNR: "
            f"{self.best_val_psnr:.2f} dB at epoch {self.best_epoch}"
        )

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

                assert self.scaler is not None
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

            loss_val = loss.item()
            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"WARNING: NaN/Inf loss at step {self.global_step}, "
                    f"batch {batch_idx}. Skipping batch."
                )
                continue

            total_loss += loss_val
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

            # Denormalize and clamp before PSNR — same protocol as
            # run_evaluation.py so checkpoint selection optimizes the
            # same metric that gets reported.
            if self.denormalize_fn is not None:
                outputs_for_psnr = self.denormalize_fn(outputs).clamp(
                    0, self.data_range
                )
                targets_for_psnr = self.denormalize_fn(targets).clamp(
                    0, self.data_range
                )
            else:
                outputs_for_psnr = outputs.clamp(0, self.data_range)
                targets_for_psnr = targets.clamp(0, self.data_range)

            psnr = self._compute_psnr(
                outputs_for_psnr, targets_for_psnr, self.data_range
            )

            total_loss += loss.item()
            total_psnr += psnr
            num_batches += 1

        mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
        mean_psnr = total_psnr / num_batches if num_batches > 0 else 0.0

        # Sanity guard: only when denormalize_fn is set (real PSNR in dB).
        # Without denormalize_fn, PSNR is on raw model output and bounds
        # depend on the data distribution, so we can't assert a range.
        if self.denormalize_fn is not None:
            if mean_psnr > 60:
                raise RuntimeError(
                    f"Val PSNR {mean_psnr:.2f} dB is suspiciously high (>60 dB). "
                    f"Check denormalization — this likely indicates a data range or "
                    f"normalization bug. Aborting to prevent wasting GPU compute."
                )
            if mean_psnr < 5 and num_batches > 0:
                raise RuntimeError(
                    f"Val PSNR {mean_psnr:.2f} dB is implausibly low (<5 dB). "
                    f"Check data loading and model forward pass. "
                    f"Aborting to prevent wasting GPU compute."
                )

        return mean_loss, mean_psnr

    def _compute_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 255.0,
    ) -> float:
        """Compute PSNR between prediction and target tensors.

        Args:
            pred: Predicted tensor [B, C, H, W].
            target: Target tensor [B, C, H, W].
            data_range: Peak signal value (255.0 for denormalized, 1.0 for [0,1]).

        Returns:
            Mean PSNR across batch in dB.
        """
        mse_per_image = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
        eps = 1e-12
        psnr_per_image = 10.0 * torch.log10(
            data_range**2 / torch.clamp(mse_per_image, min=eps)
        )
        return psnr_per_image.mean().item()

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
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "global_step": self.global_step,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        filename = "latest.pt" if is_latest else "best.pt"
        torch.save(checkpoint, self.checkpoint_dir / filename)

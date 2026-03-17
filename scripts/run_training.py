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

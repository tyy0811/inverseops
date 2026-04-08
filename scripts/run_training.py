#!/usr/bin/env python3
"""Day 4: Training script for SwinIR microscopy denoising.

Usage:
    python scripts/run_training.py --config configs/denoise_swinir.yaml
    python scripts/run_training.py --config configs/denoise_swinir.yaml --no-wandb
    python scripts/run_training.py --config configs/denoise_swinir.yaml \\
        --resume outputs/training/checkpoints/latest.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from inverseops.config import validate_config
from inverseops.data.microscopy import MicroscopyDataset
from inverseops.data.torch_datasets import MicroscopyTrainDataset
from inverseops.models import build_model
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    config["data"].setdefault("noise_source", "synthetic")

    config.setdefault("task", "denoise")

    config.setdefault("model", {})
    config["model"].setdefault("name", "swinir")
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


def build_dataloaders(
    config: dict, preload: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    data_cfg = config["data"]
    seed = config["seed"]
    noise_source = data_cfg.get("noise_source", "synthetic")

    if noise_source == "real":
        from inverseops.data.microscopy_real import RealNoiseMicroscopyDataset

        from inverseops.data.torch_datasets import RealNoiseTrainDataset

        train_base = RealNoiseMicroscopyDataset(
            root_dir=data_cfg["train_root"],
            split="train",
            seed=seed,
        )
        train_base.prepare()
        train_dataset = RealNoiseTrainDataset(
            base_dataset=train_base,
            patch_size=data_cfg["patch_size"],
            seed=seed,
            training=True,
        )

        val_base = RealNoiseMicroscopyDataset(
            root_dir=data_cfg["val_root"],
            split="val",
            seed=seed,
        )
        val_base.prepare()
        val_dataset = RealNoiseTrainDataset(
            base_dataset=val_base,
            patch_size=data_cfg["patch_size"],
            seed=seed,
            training=False,
        )
    else:
        # Existing synthetic pipeline
        train_base = MicroscopyDataset(
            root_dir=data_cfg["train_root"],
            split="train",
            seed=seed,
        )
        train_base.prepare()
        if preload:
            train_base.preload()

        train_dataset = MicroscopyTrainDataset(
            base_dataset=train_base,
            patch_size=data_cfg["patch_size"],
            sigmas=tuple(data_cfg.get("sigmas", [15, 25, 50])),
            seed=seed,
            training=True,
        )

        val_base = MicroscopyDataset(
            root_dir=data_cfg["val_root"],
            split="val",
            seed=seed,
        )
        val_base.prepare()
        if preload:
            val_base.preload()

        val_dataset = MicroscopyTrainDataset(
            base_dataset=val_base,
            patch_size=data_cfg["patch_size"],
            sigmas=tuple(data_cfg.get("sigmas", [15, 25, 50])),
            seed=seed,
            training=False,
        )

    # Apply sample limits
    limit_train = data_cfg.get("limit_train_samples")
    if limit_train is not None and limit_train < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(limit_train)))  # type: ignore[assignment]

    limit_val = data_cfg.get("limit_val_samples")
    if limit_val is not None and limit_val < len(val_dataset):
        val_dataset = Subset(val_dataset, list(range(limit_val)))  # type: ignore[assignment]

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
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Force-enable W&B logging",
    )
    wandb_group.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Force-disable W&B logging",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Run name for W&B and metadata"
            " (default: swinir_fmd_denoise_sigma15_25_50_v1)"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--limit-train-samples",
        type=int,
        default=None,
        help="Limit number of training samples",
    )
    parser.add_argument(
        "--limit-val-samples",
        type=int,
        default=None,
        help="Limit number of validation samples",
    )
    parser.add_argument(
        "--train-sigma",
        type=int,
        default=None,
        help="Override sigma to a single value",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=False,
        help="Preload all images into RAM before training",
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

    # W&B precedence: --wandb forces on, --no-wandb forces off, else config default
    if args.wandb:
        config["tracking"]["enabled"] = True
    elif args.no_wandb:
        config["tracking"]["enabled"] = False

    # Run name: CLI > config > default
    run_name = (
        args.run_name
        or config["tracking"].get("run_name")
        or "swinir_fmd_denoise_sigma15_25_50_v1"
    )
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["scheduler"]["t_max"] = args.epochs
    if args.train_sigma is not None:
        config["data"]["sigmas"] = [args.train_sigma]
    if args.limit_train_samples is not None:
        config["data"]["limit_train_samples"] = args.limit_train_samples
    if args.limit_val_samples is not None:
        config["data"]["limit_val_samples"] = args.limit_val_samples
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    # Validate config before any side effects
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Config validation error: {e}")
        return 1

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
    train_loader, val_loader = build_dataloaders(config, preload=args.preload)

    # Build model
    print("\nBuilding model...")
    model = build_model(config, device=device)
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
    resume_kwargs: dict = {}
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
        resume_kwargs["start_epoch"] = checkpoint.get("epoch", 0)
        resume_kwargs["best_val_psnr"] = checkpoint.get("best_val_psnr", float("-inf"))
        resume_kwargs["best_epoch"] = checkpoint.get("best_epoch", 0)
        resume_kwargs["epochs_without_improvement"] = checkpoint.get(
            "epochs_without_improvement", 0
        )
        resume_kwargs["global_step"] = checkpoint.get("global_step", 0)
        print(f"Resumed from epoch {resume_kwargs['start_epoch']}")

    # Get loss function
    loss_fn = get_loss(train_cfg["loss"])

    # Initialize W&B
    wandb_enabled = config["tracking"]["enabled"]
    wandb_project = config["tracking"]["wandb_project"]
    tags = config["tracking"].get("tags")
    init_wandb(
        config=config,
        enabled=wandb_enabled,
        project=wandb_project,
        run_name=run_name,
        tags=tags,
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
        **resume_kwargs,
    )

    # Reset GPU peak memory stats before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Train
    print(f"\nStarting training (run: {run_name})...")
    print("=" * 70)
    wall_start = time.time()
    trainer.train()
    wall_seconds = time.time() - wall_start
    print("=" * 70)

    # Capture GPU peak memory
    if torch.cuda.is_available():
        max_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        max_gpu_memory_mb = None

    # Enrich summary with Day 5 metadata (merge, don't replace)
    summary_path = output_dir / "training_summary.json"
    with open(summary_path) as f:
        saved_summary = json.load(f)

    saved_summary.update({
        "run_name": run_name,
        "config_path": str(args.config),
        "resolved_output_dir": str(output_dir),
        "best_checkpoint_path": str(output_dir / "checkpoints" / "best.pt"),
        "latest_checkpoint_path": str(output_dir / "checkpoints" / "latest.pt"),
        "train_time_seconds": round(wall_seconds, 2),
        "train_time_minutes": round(wall_seconds / 60, 2),
        "device": device,
        "amp_enabled": trainer.use_amp,
        "seed": config["seed"],
        "batch_size": config["data"]["batch_size"],
        "train_sigmas": config["data"]["sigmas"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
        "scheduler_name": config["scheduler"]["name"],
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_run_name": run_name,
        "max_gpu_memory_mb": (
            round(max_gpu_memory_mb, 1)
            if max_gpu_memory_mb is not None
            else None
        ),
    })

    with open(summary_path, "w") as f:
        json.dump(saved_summary, f, indent=2)

    # Log summary to W&B before finishing
    if wandb_enabled:
        import wandb
        for k, v in saved_summary.items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                wandb.run.summary[k] = v  # type: ignore[union-attr]

    # Finish W&B
    finish_run(enabled=wandb_enabled)

    # Verify artifacts
    expected_artifacts = [
        output_dir / "checkpoints" / "latest.pt",
        output_dir / "checkpoints" / "best.pt",
        summary_path,
    ]
    missing = [str(p) for p in expected_artifacts if not p.exists()]
    if missing:
        print("\nWARNING: Missing expected artifacts:")
        for m in missing:
            print(f"  - {m}")
        return 1

    # Print success summary
    gpu_str = (
        f"{max_gpu_memory_mb:.1f} MB"
        if max_gpu_memory_mb is not None
        else "N/A (CPU)"
    )
    print(f"\nRun: {run_name}")
    print(f"Best checkpoint: {output_dir / 'checkpoints' / 'best.pt'}")
    best_psnr = saved_summary['best_val_psnr']
    best_ep = saved_summary['best_epoch']
    print(f"Best val PSNR: {best_psnr:.2f} dB (epoch {best_ep})")
    print(f"Training time: {wall_seconds / 60:.1f} min")
    print(f"Peak GPU memory: {gpu_str}")
    print(f"W&B: {'enabled' if wandb_enabled else 'disabled'}")
    print("Training complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

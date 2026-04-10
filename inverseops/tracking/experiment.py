# inverseops/tracking/experiment.py
"""Experiment tracking helpers.

Provides optional W&B integration. All functions no-op when disabled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# Valid tag values for W&B experiment organization
VALID_TAGS: dict[str, set[str]] = {
    "task": {"denoising", "sr"},
    "model": {"swinir", "nafnet"},
    "noise": {"synthetic"},
}


def make_run_name(base_name: str, git_sha: str | None = None) -> str:
    """Create a unique run name with git SHA or timestamp suffix.

    Args:
        base_name: Base name for the run (e.g., 'swinir_denoise_realnoise').
        git_sha: Git commit SHA. If None, uses UTC timestamp.

    Returns:
        Unique run name like 'swinir_denoise_realnoise_abc1234'.
    """
    if git_sha:
        return f"{base_name}_{git_sha[:7]}"
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{ts}"


def init_wandb(
    config: dict,
    enabled: bool,
    project: str,
    run_name: Optional[str] = None,
    tags: list[str] | None = None,
) -> None:
    """Initialize W&B run if enabled.

    Args:
        config: Configuration dict to log.
        enabled: Whether W&B is enabled.
        project: W&B project name.
        run_name: Optional run name.
        tags: Optional list of tags for the run.
    """
    if not enabled:
        return

    import wandb

    wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags,
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

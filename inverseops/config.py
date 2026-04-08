"""Configuration validation for training configs.

Validates config dicts at load time to catch errors before
expensive setup (dataset loading, model construction).
"""

from __future__ import annotations

from inverseops.models import MODEL_REGISTRY

VALID_TASKS = {"denoise", "sr"}
VALID_LOSSES = {"l1"}
VALID_NOISE_SOURCES = {"synthetic", "real"}

# Which tasks each model supports
MODEL_TASK_SUPPORT: dict[str, set[str]] = {
    "swinir": {"denoise", "sr"},
    "nafnet": {"denoise"},
}


def validate_config(config: dict) -> None:
    """Validate a training config dict. Raises ValueError on invalid config."""
    # Task
    task = config.get("task", "denoise")
    if task not in VALID_TASKS:
        raise ValueError(
            f"Unknown task: {task!r}. Valid: {sorted(VALID_TASKS)}"
        )

    # Model
    model_name = config.get("model", {}).get("name", "")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    # Model-task compatibility
    supported_tasks = MODEL_TASK_SUPPORT.get(model_name, VALID_TASKS)
    if task not in supported_tasks:
        raise ValueError(
            f"Model {model_name!r} does not support task {task!r}. "
            f"Supported tasks: {sorted(supported_tasks)}"
        )

    # Data roots
    data_cfg = config.get("data", {})
    if not data_cfg.get("train_root"):
        raise ValueError("Config missing data.train_root")
    if not data_cfg.get("val_root"):
        raise ValueError("Config missing data.val_root")

    # Noise source (optional, defaults to synthetic)
    noise_source = data_cfg.get("noise_source", "synthetic")
    if noise_source not in VALID_NOISE_SOURCES:
        raise ValueError(
            f"Unknown noise_source: {noise_source!r}. "
            f"Valid: {sorted(VALID_NOISE_SOURCES)}"
        )

    # Loss
    loss = config.get("training", {}).get("loss", "l1")
    if loss not in VALID_LOSSES:
        raise ValueError(
            f"Unknown loss: {loss!r}. Valid: {sorted(VALID_LOSSES)}"
        )

    # Output dir
    if not config.get("output_dir"):
        raise ValueError("Config missing output_dir")

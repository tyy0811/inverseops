"""Model definitions, interfaces, and registry."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from inverseops.models.base import RestorationModel

__all__ = ["RestorationModel", "SwinIRBaseline", "MODEL_REGISTRY", "build_model"]

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str, factory: Callable[..., nn.Module]) -> None:
    """Register a model factory function."""
    MODEL_REGISTRY[name] = factory


def build_model(config: dict, device: str = "cpu") -> nn.Module:
    """Build a model from config using the registry.

    Config must have config["model"]["name"] matching a registered key.
    Additional keys in config["model"] are passed as kwargs to the factory.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "")
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name!r}. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    factory = MODEL_REGISTRY[name]
    # Pass model config kwargs (excluding 'name') to the factory
    kwargs: dict[str, Any] = {k: v for k, v in model_cfg.items() if k != "name"}
    kwargs.setdefault("device", device)
    return factory(**kwargs)


def __getattr__(name: str):
    """Lazy import for SwinIRBaseline to avoid import errors in minimal environments."""
    if name == "SwinIRBaseline":
        from inverseops.models.swinir import SwinIRBaseline
        return SwinIRBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register built-in models (lazy — imported only when build_model is called)
def _register_builtins() -> None:
    from inverseops.models.swinir import get_trainable_swinir
    register_model("swinir", get_trainable_swinir)


_register_builtins()

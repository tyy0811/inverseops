"""Model definitions and interfaces."""

from inverseops.models.base import RestorationModel

__all__ = ["RestorationModel", "SwinIRBaseline"]


def __getattr__(name: str):
    """Lazy import for SwinIRBaseline to avoid import errors in minimal environments."""
    if name == "SwinIRBaseline":
        from inverseops.models.swinir import SwinIRBaseline
        return SwinIRBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

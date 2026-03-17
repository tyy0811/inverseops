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

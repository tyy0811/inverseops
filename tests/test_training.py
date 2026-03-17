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

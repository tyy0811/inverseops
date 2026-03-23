"""Base protocol for restoration models.

Defines the minimal interface that all restoration models must implement.
This protocol is intentionally lightweight and does not depend on torch,
allowing for flexible backend implementations.
"""

from typing import Protocol


class RestorationModel(Protocol):
    """Protocol defining the interface for image restoration models.

    All restoration models (e.g., SwinIR variants) should implement this
    interface to ensure consistent usage across the codebase.
    """

    def load(self) -> None:
        """Load model weights and prepare for inference.

        This method should handle loading pretrained weights, moving the
        model to the appropriate device, and any other setup required
        before running predictions.
        """
        ...

    def predict(self, input_path: str) -> str:
        """Run restoration on a single image.

        Args:
            input_path: Path to the input image file.

        Returns:
            Path to the restored output image.
        """
        ...

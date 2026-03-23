"""Base protocol for datasets.

Defines the minimal interface that all datasets must implement.
This protocol is intentionally lightweight to allow flexibility
in how datasets are prepared and split.
"""

from typing import Protocol


class Dataset(Protocol):
    """Protocol defining the interface for datasets.

    All dataset implementations should conform to this interface
    to ensure consistent usage in training and evaluation pipelines.
    """

    def prepare(self) -> None:
        """Prepare the dataset for use.

        This method should handle any required preprocessing, downloading,
        or validation steps needed before the dataset can be used.
        """
        ...

    def split_names(self) -> list[str]:
        """Return the names of available data splits.

        Returns:
            List of split names (e.g., ["train", "val", "test"]).
        """
        ...

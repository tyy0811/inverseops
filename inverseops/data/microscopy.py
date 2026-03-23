"""Microscopy dataset loader.

Provides a minimal, torch-independent dataset loader for microscopy images.
Supports deterministic train/val/test splitting based on sorted file list and seed.
"""

from pathlib import Path

from PIL import Image


class MicroscopyDataset:
    """Dataset loader for microscopy images.

    Recursively discovers image files under root_dir (including nested folders),
    sorts them deterministically, and splits into train/val/test sets based on
    the provided seed.

    Tiny dataset behavior:
        For very small datasets, some splits may be empty. Splits are always
        disjoint and deterministic for a given seed. The splitting algorithm
        allocates test first, then val, then train from shuffled indices.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: Root directory containing image files (supports nested dirs).
            split: Which split to use ('train', 'val', or 'test').
            val_fraction: Fraction of data for validation.
            test_fraction: Fraction of data for testing.
            seed: Random seed for deterministic splitting.
            extensions: Tuple of valid image file extensions.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.extensions = tuple(ext.lower() for ext in extensions)

        self._all_files: list[Path] = []
        self._split_files: list[Path] = []
        self._prepared = False

    def prepare(self) -> None:
        """Discover image files and compute splits.

        Uses Path.rglob() to recursively find all image files, including those
        in nested subdirectories. Splits are computed deterministically and are
        guaranteed to be disjoint.

        Raises:
            FileNotFoundError: If root_dir does not exist.
            ValueError: If no image files are found in root_dir or subdirectories.
        """
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # Recursively find all image files (supports nested folders)
        self._all_files = sorted(
            f
            for f in self.root_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.extensions
        )

        if not self._all_files:
            raise ValueError(
                f"No image files found in {self.root_dir} (searched recursively).\n"
                f"Supported extensions: {self.extensions}"
            )

        # Compute split indices deterministically
        train_indices, val_indices, test_indices = self._compute_split_indices()

        # Select files for the requested split
        if self.split == "test":
            split_indices = test_indices
        elif self.split == "val":
            split_indices = val_indices
        else:
            split_indices = train_indices

        self._split_files = [self._all_files[i] for i in sorted(split_indices)]
        self._prepared = True

    def _compute_split_indices(self) -> tuple[set[int], set[int], set[int]]:
        """Compute deterministic train/val/test split indices.

        Returns:
            Tuple of (train_indices, val_indices, test_indices) as sets.
        """
        import random

        rng = random.Random(self.seed)
        indices = list(range(len(self._all_files)))
        rng.shuffle(indices)

        n_total = len(indices)
        n_test = int(n_total * self.test_fraction)
        n_val = int(n_total * self.val_fraction)

        # Allocate test, then val, then train
        test_indices = set(indices[:n_test])
        val_indices = set(indices[n_test : n_test + n_val])
        train_indices = set(indices[n_test + n_val :])

        return train_indices, val_indices, test_indices

    def split_names(self) -> list[str]:
        """Return available split names."""
        return ["train", "val", "test"]

    def __len__(self) -> int:
        """Return number of images in the current split."""
        if not self._prepared:
            raise RuntimeError("Dataset not prepared. Call prepare() first.")
        return len(self._split_files)

    def image_path(self, index: int) -> Path:
        """Return the file path for the image at the given index."""
        if not self._prepared:
            raise RuntimeError("Dataset not prepared. Call prepare() first.")
        return self._split_files[index]

    def load_image(self, index: int) -> Image.Image:
        """Load and return the image at the given index as grayscale.

        Always converts to grayscale mode 'L' regardless of source format.

        Args:
            index: Index of the image to load.

        Returns:
            PIL Image in grayscale mode ('L').
        """
        path = self.image_path(index)
        img = Image.open(path)
        # Explicitly convert to grayscale for consistency
        return img.convert("L")

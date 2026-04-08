"""Real-noise microscopy dataset loader.

Reads FMD's actual noisy/clean pairs (raw captures + averaged ground truth)
instead of applying synthetic noise to clean references.

Key design decisions:
- Splits by specimen (not by image) to prevent evaluation contamination
- All captures per specimen are training pairs (model learns noise, not structure)
- Ground truth is the averaged image (avg50.png), not Noise2Noise
- Same tensor format as MicroscopyTrainDataset: {input, target} both float32 [0,1]
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class NoisyCleanPair:
    """A single noisy capture paired with its averaged ground truth."""

    noisy_path: Path
    target_path: Path
    specimen_id: int
    capture_name: str


class RealNoiseMicroscopyDataset:
    """Dataset loader for FMD real noisy/clean pairs.

    Directory structure expected:
        root_dir/
            raw/{specimen_id}/capture_*.png   (noisy captures)
            gt/{specimen_id}/avg50.png         (averaged ground truth)

    Splits are by specimen, not by image, to prevent data leakage.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        microscope_filter: str | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.microscope_filter = microscope_filter
        self.extensions = tuple(ext.lower() for ext in extensions)

        self.pairs: list[NoisyCleanPair] = []
        self._prepared = False

    def prepare(self) -> None:
        """Discover noisy/clean pairs and compute specimen-level splits."""
        raw_dir = self.root_dir / "raw"
        gt_dir = self.root_dir / "gt"

        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        if not gt_dir.exists():
            raise FileNotFoundError(f"GT directory not found: {gt_dir}")

        # Discover specimens (numbered directories under raw/)
        specimen_ids = sorted(
            int(d.name)
            for d in raw_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if not specimen_ids:
            raise ValueError(f"No specimen directories found in {raw_dir}")

        # Split specimens into train/val/test
        train_ids, val_ids, test_ids = self._split_specimens(specimen_ids)

        if self.split == "train":
            split_ids = train_ids
        elif self.split == "val":
            split_ids = val_ids
        elif self.split == "test":
            split_ids = test_ids
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Build pairs for the selected split
        self.pairs = []
        for specimen_id in sorted(split_ids):
            specimen_raw = raw_dir / str(specimen_id)
            specimen_gt = gt_dir / str(specimen_id)

            # Find the ground truth file
            gt_files = [
                f
                for f in specimen_gt.iterdir()
                if f.is_file()
                and f.suffix.lower() in self.extensions
                and f.name != "desktop.ini"
            ]
            if not gt_files:
                continue
            gt_file = gt_files[0]  # avg50.png

            # Find all noisy captures
            for capture_path in sorted(specimen_raw.iterdir()):
                if (
                    capture_path.is_file()
                    and capture_path.suffix.lower() in self.extensions
                ):
                    self.pairs.append(
                        NoisyCleanPair(
                            noisy_path=capture_path,
                            target_path=gt_file,
                            specimen_id=specimen_id,
                            capture_name=capture_path.name,
                        )
                    )

        self._prepared = True

    def _split_specimens(
        self, specimen_ids: list[int]
    ) -> tuple[set[int], set[int], set[int]]:
        """Split specimen IDs into train/val/test sets deterministically."""
        rng = random.Random(self.seed)
        ids = list(specimen_ids)
        rng.shuffle(ids)

        n = len(ids)
        n_test = max(1, int(n * self.test_fraction))
        n_val = max(1, int(n * self.val_fraction))

        test_ids = set(ids[:n_test])
        val_ids = set(ids[n_test : n_test + n_val])
        train_ids = set(ids[n_test + n_val :])

        return train_ids, val_ids, test_ids

    def split_names(self) -> list[str]:
        return ["train", "val", "test"]

    def __len__(self) -> int:
        if not self._prepared:
            raise RuntimeError("Dataset not prepared. Call prepare() first.")
        return len(self.pairs)

    def load_pair(self, index: int) -> tuple[Image.Image, Image.Image]:
        """Load a noisy/clean pair as PIL images.

        Returns:
            Tuple of (noisy_image, clean_image) both in grayscale mode 'L'.
        """
        if not self._prepared:
            raise RuntimeError("Dataset not prepared. Call prepare() first.")

        pair = self.pairs[index]
        noisy = Image.open(pair.noisy_path).convert("L")
        clean = Image.open(pair.target_path).convert("L")
        return noisy, clean

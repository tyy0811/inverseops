"""Tests for evaluation script helpers and modes."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestSmokeFixtures:
    """Tests for smoke fixture generation."""

    def test_generate_smoke_fixtures_creates_images(self):
        """generate_smoke_fixtures should create deterministic grayscale images."""
        from scripts.run_evaluation import generate_smoke_fixtures

        with tempfile.TemporaryDirectory() as tmpdir:
            images = generate_smoke_fixtures(Path(tmpdir), count=3, seed=42)

            assert len(images) == 3
            for img_path in images:
                assert img_path.exists()
                img = Image.open(img_path)
                assert img.mode == "L"  # grayscale

    def test_generate_smoke_fixtures_deterministic(self):
        """Same seed should produce identical images."""
        from scripts.run_evaluation import generate_smoke_fixtures

        with tempfile.TemporaryDirectory() as tmpdir1:
            imgs1 = generate_smoke_fixtures(Path(tmpdir1), count=2, seed=42)
            arr1 = [np.array(Image.open(p)) for p in imgs1]

        with tempfile.TemporaryDirectory() as tmpdir2:
            imgs2 = generate_smoke_fixtures(Path(tmpdir2), count=2, seed=42)
            arr2 = [np.array(Image.open(p)) for p in imgs2]

        for a1, a2 in zip(arr1, arr2):
            assert np.array_equal(a1, a2)

# Evaluation Script Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the sigma/model mismatch in run_evaluation.py and add robust execution modes (full/partial/smoke) with proper output artifacts.

**Architecture:** Minimal changes to three files. Add checkpoint metadata properties to SwinIRBaseline. Refactor run_evaluation.py to use sigma-specific model cache and support three execution modes. Add comprehensive tests with mocked models.

**Tech Stack:** Python, pytest, PIL, numpy (existing dependencies only)

**Spec:** `docs/superpowers/specs/2026-03-16-evaluation-script-fixes-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `inverseops/models/swinir.py` | Add `checkpoint_source` and `checkpoint_resolved_path` properties |
| `scripts/run_evaluation.py` | Main evaluation logic with modes, model cache, output artifacts |
| `tests/test_evaluation.py` | New test file with mocked SwinIR |

---

## Chunk 1: Model Metadata Properties

### Task 1: Add checkpoint_source property to SwinIRBaseline

**Files:**
- Modify: `inverseops/models/swinir.py:80-82`
- Test: `tests/test_swinir_metadata.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_swinir_metadata.py`:

```python
"""Tests for SwinIR model metadata properties."""

import pytest


def test_checkpoint_source_returns_url():
    """checkpoint_source should return the download URL for the noise level."""
    # Import here to allow test collection even if torch unavailable
    try:
        from inverseops.models.swinir import SwinIRBaseline, MODEL_URLS
    except (ImportError, AttributeError):
        pytest.skip("SwinIR import failed (likely env issue)")

    model = SwinIRBaseline(noise_level=25)
    assert model.checkpoint_source == MODEL_URLS[25]
    assert "noise25" in model.checkpoint_source


def test_checkpoint_source_all_sigmas():
    """checkpoint_source should work for all supported noise levels."""
    try:
        from inverseops.models.swinir import SwinIRBaseline, MODEL_URLS
    except (ImportError, AttributeError):
        pytest.skip("SwinIR import failed (likely env issue)")

    for sigma in [15, 25, 50]:
        model = SwinIRBaseline(noise_level=sigma)
        assert model.checkpoint_source == MODEL_URLS[sigma]


def test_checkpoint_resolved_path_before_load():
    """checkpoint_resolved_path should return None before load() is called."""
    try:
        from inverseops.models.swinir import SwinIRBaseline
    except (ImportError, AttributeError):
        pytest.skip("SwinIR import failed (likely env issue)")

    model = SwinIRBaseline(noise_level=25)
    assert model.checkpoint_resolved_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_swinir_metadata.py -v`
Expected: FAIL - `checkpoint_source` attribute does not exist

- [ ] **Step 3: Implement checkpoint_source and checkpoint_resolved_path**

Add to `inverseops/models/swinir.py` after `is_loaded()` method (around line 82):

```python
    @property
    def checkpoint_source(self) -> str:
        """Return the download URL for this checkpoint."""
        return MODEL_URLS[self.noise_level]

    @property
    def checkpoint_resolved_path(self) -> Path | None:
        """Return the local cached path if weights are loaded."""
        if not self.is_loaded():
            return None
        return self.cache_dir / Path(MODEL_URLS[self.noise_level]).name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_swinir_metadata.py -v`
Expected: PASS (or SKIP if torch unavailable)

- [ ] **Step 5: Commit**

```bash
git add inverseops/models/swinir.py tests/test_swinir_metadata.py
git commit -m "feat(swinir): add checkpoint_source and checkpoint_resolved_path properties"
```

---

## Chunk 2: Core Evaluation Refactoring

### Task 2: Add smoke fixture generation helper

**Files:**
- Modify: `scripts/run_evaluation.py` (add helper function)
- Test: `tests/test_evaluation.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::TestSmokeFixtures -v`
Expected: FAIL - `generate_smoke_fixtures` not found

- [ ] **Step 3: Implement generate_smoke_fixtures**

Add to `scripts/run_evaluation.py` after imports (around line 35):

```python
def generate_smoke_fixtures(output_dir: Path, count: int = 5, seed: int = 42) -> list[Path]:
    """Generate deterministic synthetic grayscale images for smoke testing.

    Creates simple patterns: gradient, checkerboard, circles, noise.

    Args:
        output_dir: Directory to save images.
        count: Number of images to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of paths to generated images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    size = 128
    paths = []

    patterns = ["gradient", "checkerboard", "circles", "noise", "stripes"]

    for i in range(count):
        pattern = patterns[i % len(patterns)]

        if pattern == "gradient":
            arr = np.tile(np.linspace(0, 255, size), (size, 1)).astype(np.uint8)
        elif pattern == "checkerboard":
            block = 16
            arr = np.zeros((size, size), dtype=np.uint8)
            for y in range(0, size, block):
                for x in range(0, size, block):
                    if ((y // block) + (x // block)) % 2 == 0:
                        arr[y : y + block, x : x + block] = 255
        elif pattern == "circles":
            y, x = np.ogrid[:size, :size]
            center = size // 2
            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            arr = ((np.sin(dist / 5) + 1) * 127.5).astype(np.uint8)
        elif pattern == "noise":
            arr = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
        else:  # stripes
            arr = np.zeros((size, size), dtype=np.uint8)
            for y in range(0, size, 8):
                arr[y : y + 4, :] = 255

        img = Image.fromarray(arr, mode="L")
        path = output_dir / f"smoke_{pattern}_{i:03d}.png"
        img.save(path)
        paths.append(path)

    return paths
```

Also add numpy import at the top if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py::TestSmokeFixtures -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_evaluation.py tests/test_evaluation.py
git commit -m "feat(eval): add generate_smoke_fixtures helper"
```

---

### Task 3: Add aggregate_results with std calculation

**Files:**
- Modify: `scripts/run_evaluation.py` (update aggregate_results)
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation.py`:

```python
class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_results_mean_std(self):
        """aggregate_results should compute correct mean and population std."""
        from scripts.run_evaluation import EvalResult, aggregate_results

        results = [
            EvalResult(sigma=25, domain="test", dataset_name="ds", is_real_data=True,
                       image_name="a.png", psnr=30.0, ssim=0.9, noise_seed=42,
                       model_name="swinir", model_checkpoint="ckpt.pth"),
            EvalResult(sigma=25, domain="test", dataset_name="ds", is_real_data=True,
                       image_name="b.png", psnr=32.0, ssim=0.92, noise_seed=43,
                       model_name="swinir", model_checkpoint="ckpt.pth"),
        ]

        agg = aggregate_results(results)
        key = ("test", 25)

        assert key in agg
        assert agg[key]["count"] == 2
        assert agg[key]["psnr_mean"] == pytest.approx(31.0)
        # Population std: sqrt(((30-31)^2 + (32-31)^2) / 2) = sqrt(1) = 1.0
        assert agg[key]["psnr_std"] == pytest.approx(1.0)
        assert agg[key]["ssim_mean"] == pytest.approx(0.91)

    def test_aggregate_results_single_image_std_zero(self):
        """With count=1, std should be 0.0 (not NaN)."""
        from scripts.run_evaluation import EvalResult, aggregate_results

        results = [
            EvalResult(sigma=15, domain="single", dataset_name="ds", is_real_data=True,
                       image_name="only.png", psnr=28.5, ssim=0.85, noise_seed=1,
                       model_name="swinir", model_checkpoint="ckpt.pth"),
        ]

        agg = aggregate_results(results)
        assert agg[("single", 15)]["psnr_std"] == 0.0
        assert agg[("single", 15)]["ssim_std"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::TestAggregateResults -v`
Expected: FAIL - EvalResult missing new fields

- [ ] **Step 3: Update EvalResult dataclass**

Replace `EvalResult` in `scripts/run_evaluation.py`:

```python
@dataclass
class EvalResult:
    """Single evaluation result row."""

    sigma: int
    domain: str
    dataset_name: str
    is_real_data: bool
    image_name: str
    psnr: float
    ssim: float
    noise_seed: int
    model_name: str
    model_checkpoint: str
```

- [ ] **Step 4: Update aggregate_results function**

Replace `aggregate_results` in `scripts/run_evaluation.py`:

```python
def aggregate_results(results: list[EvalResult]) -> dict:
    """Compute aggregate statistics by domain and sigma.

    Uses population std (ddof=0). For count=1, std=0.0.
    """
    from collections import defaultdict

    groups: dict[tuple[str, int], list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[(r.domain, r.sigma)].append(r)

    agg = {}
    for (domain, sigma), group in groups.items():
        psnrs = [r.psnr for r in group]
        ssims = [r.ssim for r in group]
        n = len(group)

        psnr_mean = sum(psnrs) / n
        ssim_mean = sum(ssims) / n

        # Population std (ddof=0)
        if n == 1:
            psnr_std = 0.0
            ssim_std = 0.0
        else:
            psnr_std = (sum((p - psnr_mean) ** 2 for p in psnrs) / n) ** 0.5
            ssim_std = (sum((s - ssim_mean) ** 2 for s in ssims) / n) ** 0.5

        agg[(domain, sigma)] = {
            "count": n,
            "psnr_mean": psnr_mean,
            "psnr_std": psnr_std,
            "ssim_mean": ssim_mean,
            "ssim_std": ssim_std,
            "dataset_name": group[0].dataset_name,
            "is_real_data": group[0].is_real_data,
            "model_name": group[0].model_name,
            "model_checkpoint": group[0].model_checkpoint,
        }

    return agg
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py::TestAggregateResults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_evaluation.py tests/test_evaluation.py
git commit -m "feat(eval): update EvalResult and aggregate_results with new fields"
```

---

### Task 4: Add save_summary_csv function

**Files:**
- Modify: `scripts/run_evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation.py`:

```python
class TestSummaryCSV:
    """Tests for baseline_summary.csv output."""

    def test_save_summary_csv_columns(self):
        """baseline_summary.csv should have all required columns."""
        from scripts.run_evaluation import save_summary_csv

        agg = {
            ("microscopy", 25): {
                "count": 5,
                "psnr_mean": 29.5,
                "psnr_std": 1.2,
                "ssim_mean": 0.88,
                "ssim_std": 0.02,
                "dataset_name": "fmd",
                "is_real_data": True,
                "model_name": "swinir",
                "model_checkpoint": "noise25.pth",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.csv"
            save_summary_csv(
                agg=agg,
                output_path=path,
                seed=42,
                mode="full",
                decision_gate_valid=True,
            )

            assert path.exists()

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            row = rows[0]

            required_cols = [
                "sigma", "noise_type", "domain", "dataset_name", "is_real_data",
                "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
                "seed", "model_name", "model_checkpoint", "mode", "decision_gate_valid"
            ]
            for col in required_cols:
                assert col in row, f"Missing column: {col}"

            assert row["sigma"] == "25"
            assert row["noise_type"] == "synthetic_gaussian"
            assert row["mode"] == "full"
            assert row["decision_gate_valid"] == "True"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::TestSummaryCSV -v`
Expected: FAIL - `save_summary_csv` not found

- [ ] **Step 3: Implement save_summary_csv**

Add to `scripts/run_evaluation.py`:

```python
def save_summary_csv(
    agg: dict,
    output_path: Path,
    seed: int,
    mode: str,
    decision_gate_valid: bool,
) -> None:
    """Save aggregate statistics to baseline_summary.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sigma", "noise_type", "domain", "dataset_name", "is_real_data",
        "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
        "seed", "model_name", "model_checkpoint", "mode", "decision_gate_valid"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (domain, sigma), stats in sorted(agg.items()):
            writer.writerow({
                "sigma": sigma,
                "noise_type": "synthetic_gaussian",
                "domain": domain,
                "dataset_name": stats["dataset_name"],
                "is_real_data": stats["is_real_data"],
                "count": stats["count"],
                "psnr_mean": f"{stats['psnr_mean']:.4f}",
                "psnr_std": f"{stats['psnr_std']:.4f}",
                "ssim_mean": f"{stats['ssim_mean']:.4f}",
                "ssim_std": f"{stats['ssim_std']:.4f}",
                "seed": seed,
                "model_name": stats["model_name"],
                "model_checkpoint": stats["model_checkpoint"],
                "mode": mode,
                "decision_gate_valid": decision_gate_valid,
            })

    print(f"Summary saved to: {output_path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py::TestSummaryCSV -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_evaluation.py tests/test_evaluation.py
git commit -m "feat(eval): add save_summary_csv function"
```

---

### Task 5: Add save_decision_json function

**Files:**
- Modify: `scripts/run_evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation.py`:

```python
class TestDecisionJSON:
    """Tests for day3_decision.json output."""

    def test_save_decision_json_full_mode(self):
        """In full mode with gap > threshold, dataset_locked should be True."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("microscopy", 15): {"psnr_mean": 28.0, "count": 5},
            ("microscopy", 25): {"psnr_mean": 26.0, "count": 5},
            ("natural", 15): {"psnr_mean": 30.0, "count": 5},
            ("natural", 25): {"psnr_mean": 28.5, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="full",
                sigmas=[15, 25],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "full"
            assert decision["decision_gate_valid"] is True
            assert decision["dataset_locked"] is True
            assert decision["gap_threshold_db"] == 1.0
            assert "per_sigma_gap_db" in decision
            assert "15" in decision["per_sigma_gap_db"]

    def test_save_decision_json_partial_mode(self):
        """In partial mode, decision_gate_valid and dataset_locked should be False."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("microscopy", 25): {"psnr_mean": 28.0, "count": 5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="partial",
                sigmas=[25],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "partial"
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
            assert "Insufficient" in decision["recommendation"]

    def test_save_decision_json_smoke_mode(self):
        """In smoke mode, decision_gate_valid and dataset_locked should be False."""
        from scripts.run_evaluation import save_decision_json

        agg = {
            ("smoke_microscopy", 25): {"psnr_mean": 28.0, "count": 3},
            ("smoke_natural", 25): {"psnr_mean": 29.0, "count": 3},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            save_decision_json(
                agg=agg,
                output_path=path,
                mode="smoke",
                sigmas=[25],
                gap_threshold_db=1.0,
            )

            with open(path) as f:
                decision = json.load(f)

            assert decision["mode"] == "smoke"
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::TestDecisionJSON -v`
Expected: FAIL - `save_decision_json` not found

- [ ] **Step 3: Implement save_decision_json**

Add to `scripts/run_evaluation.py`:

```python
def save_decision_json(
    agg: dict,
    output_path: Path,
    mode: str,
    sigmas: list[int],
    gap_threshold_db: float,
) -> None:
    """Save day3_decision.json with decision gate analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count images per domain type
    microscopy_count = sum(
        stats["count"] for (domain, _), stats in agg.items()
        if "microscopy" in domain.lower()
    )
    natural_count = sum(
        stats["count"] for (domain, _), stats in agg.items()
        if "natural" in domain.lower()
    )

    # Compute per-sigma gaps
    per_sigma_gap_db = {}
    for sigma in sigmas:
        micro_key = None
        nat_key = None
        for (domain, s) in agg.keys():
            if s == sigma:
                if "microscopy" in domain.lower():
                    micro_key = (domain, s)
                elif "natural" in domain.lower():
                    nat_key = (domain, s)

        if micro_key and nat_key:
            gap = agg[nat_key]["psnr_mean"] - agg[micro_key]["psnr_mean"]
            per_sigma_gap_db[str(sigma)] = round(gap, 2)

    # Compute overall means
    micro_psnrs = [
        stats["psnr_mean"] for (domain, _), stats in agg.items()
        if "microscopy" in domain.lower()
    ]
    nat_psnrs = [
        stats["psnr_mean"] for (domain, _), stats in agg.items()
        if "natural" in domain.lower()
    ]

    overall_micro_psnr = sum(micro_psnrs) / len(micro_psnrs) if micro_psnrs else None
    overall_natural_psnr = sum(nat_psnrs) / len(nat_psnrs) if nat_psnrs else None

    if overall_micro_psnr is not None and overall_natural_psnr is not None:
        overall_gap_db = round(overall_natural_psnr - overall_micro_psnr, 2)
    else:
        overall_gap_db = None

    # Decision logic
    decision_gate_valid = mode == "full"
    dataset_locked = mode == "full"

    if decision_gate_valid and overall_gap_db is not None:
        if overall_gap_db > gap_threshold_db:
            recommendation = (
                f"Retain FMD as primary dataset; microscopy underperforms natural "
                f"images by {overall_gap_db:.1f} dB under matched synthetic Gaussian corruption."
            )
        elif overall_gap_db > 0.5:
            recommendation = (
                f"Modest domain gap detected ({overall_gap_db:.1f} dB). "
                "Fine-tuning may provide incremental gains."
            )
        else:
            recommendation = (
                "Microscopy performance is similar to natural images. "
                "Consider using a more challenging microscopy subset (e.g., EM)."
            )
    else:
        recommendation = (
            "Insufficient real microscopy-vs-natural evidence for Day 3 decision gate."
        )

    decision = {
        "mode": mode,
        "decision_gate_valid": decision_gate_valid,
        "dataset_locked": dataset_locked,
        "microscopy_count": microscopy_count,
        "natural_count": natural_count,
        "sigma_list": sigmas,
        "gap_threshold_db": gap_threshold_db,
        "per_sigma_gap_db": per_sigma_gap_db,
        "overall_micro_psnr": round(overall_micro_psnr, 2) if overall_micro_psnr else None,
        "overall_natural_psnr": round(overall_natural_psnr, 2) if overall_natural_psnr else None,
        "overall_gap_db": overall_gap_db,
        "recommendation": recommendation,
        "notes": "Day 3 uses synthetic Gaussian noise on clean reference images.",
    }

    with open(output_path, "w") as f:
        json.dump(decision, f, indent=2)

    print(f"Decision saved to: {output_path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py::TestDecisionJSON -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_evaluation.py tests/test_evaluation.py
git commit -m "feat(eval): add save_decision_json function"
```

---

## Chunk 3: Main Evaluation Flow Refactoring

### Task 6: Refactor evaluate_domain to use model cache

**Files:**
- Modify: `scripts/run_evaluation.py`
- Test: `tests/test_evaluation.py`

**Prerequisites**: The following functions must exist in `scripts/run_evaluation.py`:
- `add_gaussian_noise(image, sigma, seed)` - adds Gaussian noise to PIL image
- `compute_psnr(ref, pred)` - computes PSNR between two images
- `compute_ssim(ref, pred)` - computes SSIM between two images

These should already exist from the original Day 3 implementation. Verify with:

Run: `grep -n "def add_gaussian_noise\|from inverseops.evaluation.metrics import" scripts/run_evaluation.py`

If missing, add the import:
```python
from inverseops.evaluation.metrics import compute_psnr, compute_ssim
```

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation.py`:

```python
class TestEvaluateDomain:
    """Tests for evaluate_domain with model cache."""

    @pytest.fixture
    def mock_swinir(self, monkeypatch):
        """Mock SwinIRBaseline to avoid model downloads."""
        class MockModel:
            def __init__(self, noise_level, **kwargs):
                self.noise_level = noise_level
                self.device = "cpu"
                self._loaded = False

            def load(self):
                self._loaded = True

            def is_loaded(self):
                return self._loaded

            def predict_image(self, img):
                # Return slightly modified image to simulate denoising
                return img

            @property
            def checkpoint_source(self):
                return f"mock://sigma{self.noise_level}"

            @property
            def checkpoint_resolved_path(self):
                return Path(f"/tmp/mock_{self.noise_level}.pth")

        monkeypatch.setattr(
            "scripts.run_evaluation.SwinIRBaseline", MockModel
        )
        return MockModel

    def test_evaluate_domain_uses_sigma_specific_model(self, mock_swinir):
        """evaluate_domain should use the correct model for each sigma."""
        from scripts.run_evaluation import evaluate_domain, load_models

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img_path = Path(tmpdir) / "test.png"
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(img_path)

            models = load_models([15, 25])

            results = evaluate_domain(
                models=models,
                images=[img_path],
                domain="test",
                dataset_name="test_ds",
                is_real_data=True,
                sigmas=(15, 25),
                seed=42,
                limit=1,
            )

            # Should have 2 results (1 image x 2 sigmas)
            assert len(results) == 2

            # Check sigma-specific checkpoint used
            sigma_15_result = [r for r in results if r.sigma == 15][0]
            sigma_25_result = [r for r in results if r.sigma == 25][0]

            assert "15" in sigma_15_result.model_checkpoint
            assert "25" in sigma_25_result.model_checkpoint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::TestEvaluateDomain -v`
Expected: FAIL - `load_models` and new `evaluate_domain` signature not found

- [ ] **Step 3: Implement load_models function**

Add to `scripts/run_evaluation.py`:

```python
def load_models(sigmas: list[int]) -> dict[int, SwinIRBaseline]:
    """Load SwinIR models for each sigma level.

    Models are cached to avoid redundant loading.

    Args:
        sigmas: List of sigma values to load models for.

    Returns:
        Dictionary mapping sigma to loaded model.
    """
    models: dict[int, SwinIRBaseline] = {}
    for sigma in sigmas:
        print(f"Loading SwinIR model (sigma={sigma})...")
        model = SwinIRBaseline(noise_level=sigma)
        model.load()
        models[sigma] = model
        print(f"  Loaded on device: {model.device}")
    return models
```

- [ ] **Step 4: Refactor evaluate_domain signature**

Replace `evaluate_domain` in `scripts/run_evaluation.py`:

```python
def evaluate_domain(
    models: dict[int, SwinIRBaseline],
    images: list[Path],
    domain: str,
    dataset_name: str,
    is_real_data: bool,
    sigmas: tuple[int, ...],
    seed: int,
    limit: int,
) -> list[EvalResult]:
    """Evaluate models on a set of images across all sigma levels.

    Args:
        models: Dictionary mapping sigma to loaded SwinIR model.
        images: List of image paths to evaluate.
        domain: Domain name (e.g., 'microscopy', 'natural').
        dataset_name: Dataset identifier (e.g., 'fmd', 'smoke_fixture').
        is_real_data: Whether this is real data (True) or synthetic (False).
        sigmas: Noise levels to test.
        seed: Random seed for noise generation.
        limit: Maximum number of images to evaluate.

    Returns:
        List of EvalResult for each (image, sigma) combination.
    """
    results = []
    images = images[:limit]

    for i, img_path in enumerate(images):
        clean = Image.open(img_path).convert("L")

        for sigma_idx, sigma in enumerate(sigmas):
            model = models[sigma]

            noise_seed = seed + i * 100 + sigma_idx
            noisy = add_gaussian_noise(clean, sigma, seed=noise_seed)

            denoised = model.predict_image(noisy)

            psnr = compute_psnr(clean, denoised)
            ssim = compute_ssim(clean, denoised)

            checkpoint_name = Path(model.checkpoint_source).name

            results.append(
                EvalResult(
                    sigma=sigma,
                    domain=domain,
                    dataset_name=dataset_name,
                    is_real_data=is_real_data,
                    image_name=img_path.name,
                    psnr=psnr,
                    ssim=ssim,
                    noise_seed=noise_seed,
                    model_name="swinir",
                    model_checkpoint=checkpoint_name,
                )
            )

            print(
                f"  [{domain}] {img_path.name} sigma={sigma}: "
                f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}"
            )

    return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py::TestEvaluateDomain -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_evaluation.py tests/test_evaluation.py
git commit -m "refactor(eval): use sigma-specific model cache in evaluate_domain"
```

---

### Task 7: Update CLI arguments and constants

**Files:**
- Modify: `scripts/run_evaluation.py`

- [ ] **Step 1: Add SUPPORTED_SIGMAS constant**

Add at module level (after imports, around line 25):

```python
# Supported noise levels for SwinIR denoising
SUPPORTED_SIGMAS: tuple[int, ...] = (15, 25, 50)
```

- [ ] **Step 2: Verify required helper imports**

Ensure these imports exist at the top of `scripts/run_evaluation.py`:

```python
import csv
import json
```

If `discover_images`, `MicroscopyDataset`, or `print_summary` don't exist, they should already be present from the original script. Verify with:

Run: `grep -n "def discover_images\|class MicroscopyDataset\|def print_summary" scripts/run_evaluation.py`

If any are missing, check the existing implementation and either import from the correct location or implement them.

- [ ] **Step 3: Remove old --noise-level argument (if present)**

Search for and remove any existing `--noise-level` argument:

Run: `grep -n "noise-level" scripts/run_evaluation.py`

If found, delete that argument block. The new `--single-sigma` argument replaces it.

- [ ] **Step 4: Update argparse arguments**

Replace the argument parser section in `main()`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained SwinIR on microscopy and natural images."
    )
    parser.add_argument(
        "--microscopy-root",
        type=Path,
        default=Path("data/raw/fmd"),
        help="Root directory for microscopy images",
    )
    parser.add_argument(
        "--natural-root",
        type=Path,
        default=None,
        help="Root directory for natural reference images (optional)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to use for microscopy images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of images per domain to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/baseline/baseline_metrics.csv"),
        help="Output CSV path for per-image metrics",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV path for summary (default: next to output-csv)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="inverseops-baseline",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--single-sigma",
        type=int,
        choices=[15, 25, 50],
        default=None,
        help="Evaluate only a single sigma (default: all)",
    )
    parser.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help="Allow partial/smoke mode instead of error when datasets missing",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Force smoke mode with synthetic images",
    )
    parser.add_argument(
        "--smoke-count",
        type=int,
        default=5,
        help="Number of synthetic images per domain in smoke mode",
    )
    parser.add_argument(
        "--gap-threshold-db",
        type=float,
        default=1.0,
        help="PSNR gap threshold (dB) for domain difference (default: 1.0)",
    )

    args = parser.parse_args()
```

- [ ] **Step 5: Run linter to verify syntax**

Run: `ruff check scripts/run_evaluation.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add scripts/run_evaluation.py
git commit -m "feat(eval): add SUPPORTED_SIGMAS constant and update CLI arguments"
```

---

### Task 8: Implement mode determination and main evaluation flow

**Files:**
- Modify: `scripts/run_evaluation.py`

- [ ] **Step 1: Add determine_mode helper**

Add to `scripts/run_evaluation.py`:

```python
def determine_mode(
    microscopy_count: int,
    natural_count: int,
    force_smoke: bool,
) -> str:
    """Determine execution mode based on available data.

    Args:
        microscopy_count: Number of microscopy images found.
        natural_count: Number of natural images found.
        force_smoke: Whether --smoke-mode was specified.

    Returns:
        Mode string: 'full', 'partial', or 'smoke'.
    """
    if force_smoke:
        return "smoke"
    if microscopy_count > 0 and natural_count > 0:
        return "full"
    if microscopy_count > 0 or natural_count > 0:
        return "partial"
    return "smoke"
```

- [ ] **Step 2: Rewrite main function body**

Replace the main function body after argument parsing:

```python
    # Determine sigmas to evaluate
    if args.single_sigma:
        sigmas = (args.single_sigma,)
    else:
        sigmas = SUPPORTED_SIGMAS  # (15, 25, 50)

    # Set default summary CSV path
    if args.summary_csv is None:
        args.summary_csv = args.output_csv.parent / "baseline_summary.csv"

    decision_json_path = args.output_csv.parent / "day3_decision.json"

    # Collect images from each domain
    microscopy_images: list[Path] = []
    natural_images: list[Path] = []
    microscopy_dataset_name = "fmd"
    natural_dataset_name = "natural"

    if not args.smoke_mode:
        # Try to load microscopy images
        if args.microscopy_root and args.microscopy_root.exists():
            try:
                microscopy_ds = MicroscopyDataset(
                    root_dir=args.microscopy_root,
                    split=args.split,
                    seed=args.seed,
                )
                microscopy_ds.prepare()
                microscopy_images = [
                    microscopy_ds.image_path(i) for i in range(len(microscopy_ds))
                ]
                print(f"Found {len(microscopy_images)} microscopy images ({args.split} split)")
            except ValueError as e:
                print(f"Warning: Could not load microscopy images: {e}")

        # Try to load natural images
        if args.natural_root and args.natural_root.exists():
            natural_images = discover_images(args.natural_root)
            if natural_images:
                print(f"Found {len(natural_images)} natural images")
            else:
                print(f"Warning: No images found in {args.natural_root}")

    # Determine execution mode
    mode = determine_mode(
        microscopy_count=len(microscopy_images),
        natural_count=len(natural_images),
        force_smoke=args.smoke_mode,
    )

    # Check if we should fail on missing data
    if mode != "full" and not args.allow_missing_datasets and not args.smoke_mode:
        print("Error: Missing datasets and --allow-missing-datasets not set.")
        print("  Microscopy images: ", len(microscopy_images))
        print("  Natural images: ", len(natural_images))
        print("Use --allow-missing-datasets for partial mode or --smoke-mode for smoke testing.")
        return 1

    decision_gate_valid = mode == "full"

    # Print mode warning
    if mode == "smoke":
        print("\n" + "=" * 80)
        print("WARNING: SMOKE MODE - RESULTS ARE NOT SCIENTIFICALLY MEANINGFUL")
        print("These outputs verify pipeline execution only.")
        print("DO NOT use for Day 3 dataset decision.")
        print("=" * 80 + "\n")
    elif mode == "partial":
        print("\n" + "=" * 80)
        print("WARNING: PARTIAL MODE - Only one domain available")
        print("Day 3 decision gate cannot be concluded with one domain.")
        print("=" * 80 + "\n")

    # Generate smoke fixtures if needed
    if mode == "smoke":
        import tempfile
        smoke_dir = Path(tempfile.mkdtemp(prefix="inverseops_smoke_"))
        smoke_micro_dir = smoke_dir / "microscopy"
        smoke_nat_dir = smoke_dir / "natural"

        microscopy_images = generate_smoke_fixtures(
            smoke_micro_dir, count=args.smoke_count, seed=args.seed
        )
        natural_images = generate_smoke_fixtures(
            smoke_nat_dir, count=args.smoke_count, seed=args.seed + 1000
        )
        microscopy_dataset_name = "smoke_fixture"
        natural_dataset_name = "smoke_fixture"
        print(f"Generated {args.smoke_count} smoke fixtures per domain")

    # Load models
    print(f"\nLoading SwinIR models for sigmas: {sigmas}")
    models = load_models(list(sigmas))

    # Initialize W&B
    if not args.no_wandb:
        checkpoint_sources = {
            sigma: models[sigma].checkpoint_source for sigma in sigmas
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "mode": mode,
                "decision_gate_valid": decision_gate_valid,
                "microscopy_root": str(args.microscopy_root) if args.microscopy_root else None,
                "natural_root": str(args.natural_root) if args.natural_root else None,
                "split": args.split,
                "limit": args.limit,
                "seed": args.seed,
                "sigmas": list(sigmas),
                "gap_threshold_db": args.gap_threshold_db,
                "checkpoint_sources": checkpoint_sources,
                "device": models[sigmas[0]].device,
                "output_csv": str(args.output_csv),
                "summary_csv": str(args.summary_csv),
            },
        )

    all_results: list[EvalResult] = []

    # Evaluate microscopy
    if microscopy_images:
        domain_name = "smoke_microscopy" if mode == "smoke" else "microscopy"
        print(f"\nEvaluating {domain_name} images...")
        micro_results = evaluate_domain(
            models=models,
            images=microscopy_images,
            domain=domain_name,
            dataset_name=microscopy_dataset_name,
            is_real_data=(mode != "smoke"),
            sigmas=sigmas,
            seed=args.seed,
            limit=args.limit,
        )
        all_results.extend(micro_results)

    # Evaluate natural
    if natural_images:
        domain_name = "smoke_natural" if mode == "smoke" else "natural"
        print(f"\nEvaluating {domain_name} images...")
        nat_results = evaluate_domain(
            models=models,
            images=natural_images,
            domain=domain_name,
            dataset_name=natural_dataset_name,
            is_real_data=(mode != "smoke"),
            sigmas=sigmas,
            seed=args.seed,
            limit=args.limit,
        )
        all_results.extend(nat_results)

    # Aggregate and display results
    agg = aggregate_results(all_results)
    print_summary(agg, sigmas)

    # Save outputs
    save_csv(all_results, args.output_csv, mode)
    save_summary_csv(agg, args.summary_csv, args.seed, mode, decision_gate_valid)
    save_decision_json(agg, decision_json_path, mode, list(sigmas), args.gap_threshold_db)

    # Log to W&B
    if not args.no_wandb:
        for (domain, sigma), stats in agg.items():
            wandb.log({
                f"{domain}/sigma_{sigma}/psnr_mean": stats["psnr_mean"],
                f"{domain}/sigma_{sigma}/ssim_mean": stats["ssim_mean"],
                f"{domain}/sigma_{sigma}/count": stats["count"],
            })

        table_data = [
            [r.sigma, r.domain, r.image_name, r.psnr, r.ssim, r.model_checkpoint]
            for r in all_results
        ]
        wandb_table = wandb.Table(
            columns=["sigma", "domain", "image_name", "psnr", "ssim", "checkpoint"],
            data=table_data,
        )
        wandb.log({"results_table": wandb_table})

        artifact = wandb.Artifact("baseline_metrics", type="evaluation")
        artifact.add_file(str(args.output_csv))
        artifact.add_file(str(args.summary_csv))
        artifact.add_file(str(decision_json_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("\nW&B run complete.")

    return 0
```

- [ ] **Step 3: Update save_csv to include mode**

Update `save_csv` signature and implementation:

```python
def save_csv(results: list[EvalResult], output_path: Path, mode: str) -> None:
    """Save per-image results to baseline_metrics.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sigma", "noise_type", "domain", "dataset_name", "is_real_data",
        "image_name", "psnr", "ssim", "seed", "model_name", "model_checkpoint", "mode"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "sigma": r.sigma,
                "noise_type": "synthetic_gaussian",
                "domain": r.domain,
                "dataset_name": r.dataset_name,
                "is_real_data": r.is_real_data,
                "image_name": r.image_name,
                "psnr": f"{r.psnr:.4f}",
                "ssim": f"{r.ssim:.4f}",
                "seed": r.noise_seed,
                "model_name": r.model_name,
                "model_checkpoint": r.model_checkpoint,
                "mode": mode,
            })

    print(f"\nResults saved to: {output_path}")
```

- [ ] **Step 4: Run linter**

Run: `ruff check scripts/run_evaluation.py`
Expected: No errors (or fix any issues)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_evaluation.py
git commit -m "feat(eval): implement mode determination and refactored main flow"
```

---

## Chunk 4: Integration Tests

### Task 9: Add integration tests for modes

**Files:**
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Add partial mode integration test**

Add to `tests/test_evaluation.py`:

```python
class TestModeIntegration:
    """Integration tests for execution modes."""

    @pytest.fixture
    def mock_swinir(self, monkeypatch):
        """Mock SwinIRBaseline."""
        class MockModel:
            def __init__(self, noise_level, **kwargs):
                self.noise_level = noise_level
                self.device = "cpu"

            def load(self):
                pass

            def is_loaded(self):
                return True

            def predict_image(self, img):
                return img

            @property
            def checkpoint_source(self):
                return f"mock://sigma{self.noise_level}.pth"

            @property
            def checkpoint_resolved_path(self):
                return Path(f"/tmp/mock_{self.noise_level}.pth")

        monkeypatch.setattr("scripts.run_evaluation.SwinIRBaseline", MockModel)

    def test_partial_mode_produces_csvs(self, mock_swinir):
        """Partial mode should produce CSVs with mode=partial."""
        from scripts.run_evaluation import (
            evaluate_domain, aggregate_results, save_csv,
            save_summary_csv, save_decision_json, load_models,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create one test image (microscopy only)
            img_path = tmpdir / "micro.png"
            Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(img_path)

            models = load_models([25])
            results = evaluate_domain(
                models=models,
                images=[img_path],
                domain="microscopy",
                dataset_name="fmd",
                is_real_data=True,
                sigmas=(25,),
                seed=42,
                limit=1,
            )

            agg = aggregate_results(results)

            metrics_path = tmpdir / "metrics.csv"
            summary_path = tmpdir / "summary.csv"
            decision_path = tmpdir / "decision.json"

            save_csv(results, metrics_path, mode="partial")
            save_summary_csv(agg, summary_path, seed=42, mode="partial",
                           decision_gate_valid=False)
            save_decision_json(agg, decision_path, mode="partial",
                             sigmas=[25], gap_threshold_db=1.0)

            # Verify metrics CSV
            with open(metrics_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["mode"] == "partial"

            # Verify summary CSV
            with open(summary_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert rows[0]["decision_gate_valid"] == "False"

            # Verify decision JSON
            with open(decision_path) as f:
                decision = json.load(f)
            assert decision["mode"] == "partial"
            assert decision["dataset_locked"] is False

    def test_smoke_mode_produces_csvs(self, mock_swinir):
        """Smoke mode should produce CSVs with mode=smoke and is_real_data=False."""
        from scripts.run_evaluation import (
            generate_smoke_fixtures, evaluate_domain, aggregate_results,
            save_csv, save_summary_csv, save_decision_json, load_models,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            smoke_images = generate_smoke_fixtures(tmpdir / "smoke", count=2, seed=42)

            models = load_models([25])
            results = evaluate_domain(
                models=models,
                images=smoke_images,
                domain="smoke_microscopy",
                dataset_name="smoke_fixture",
                is_real_data=False,
                sigmas=(25,),
                seed=42,
                limit=2,
            )

            agg = aggregate_results(results)

            metrics_path = tmpdir / "metrics.csv"
            summary_path = tmpdir / "summary.csv"
            decision_path = tmpdir / "decision.json"

            save_csv(results, metrics_path, mode="smoke")
            save_summary_csv(agg, summary_path, seed=42, mode="smoke",
                           decision_gate_valid=False)
            save_decision_json(agg, decision_path, mode="smoke",
                             sigmas=[25], gap_threshold_db=1.0)

            # Verify metrics CSV
            with open(metrics_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["is_real_data"] == "False"
            assert rows[0]["mode"] == "smoke"
            assert "smoke_" in rows[0]["domain"]

            # Verify decision JSON
            with open(decision_path) as f:
                decision = json.load(f)
            assert decision["decision_gate_valid"] is False
            assert decision["dataset_locked"] is False
            assert "Insufficient" in decision["recommendation"]
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_evaluation.py::TestModeIntegration -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_evaluation.py
git commit -m "test(eval): add integration tests for partial and smoke modes"
```

---

### Task 10: Final verification and cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (some may skip due to torch)

- [ ] **Step 2: Run linter**

Run: `ruff check inverseops scripts tests`
Expected: No errors

- [ ] **Step 3: Run smoke mode to verify pipeline**

Run: `python scripts/run_evaluation.py --smoke-mode --smoke-count 2 --no-wandb`
Expected: Completes successfully with smoke warnings

- [ ] **Step 4: Verify output artifacts**

Check:
- `artifacts/baseline/baseline_metrics.csv` exists with correct columns
- `artifacts/baseline/baseline_summary.csv` exists with correct columns
- `artifacts/baseline/day3_decision.json` exists with mode=smoke

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "feat(eval): complete Day 3 evaluation script fixes

- Fix sigma/model mismatch: each sigma uses matching checkpoint
- Add execution modes: full, partial, smoke
- Add baseline_summary.csv with all required columns
- Add day3_decision.json for decision gate
- Add smoke fixtures for pipeline testing
- Update CLI with new flags
- Add comprehensive tests with mocked models"
```

---

## Summary

### How to run each mode:

**Full mode (default, strict):**
```bash
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd \
    --natural-root data/raw/natural
```

**Partial mode:**
```bash
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd \
    --allow-missing-datasets
```

**Smoke mode:**
```bash
python scripts/run_evaluation.py \
    --smoke-mode \
    --smoke-count 3
```

# Day 3 Evaluation Script Fixes

**Date**: 2026-03-16
**Status**: Draft
**Goal**: Fix scientific flaws in `run_evaluation.py` and add robust execution modes

## Problem Statement

The current `scripts/run_evaluation.py` has four issues:

1. **Sigma/model mismatch**: Loads one SwinIR checkpoint via `--noise-level` but evaluates all sigmas (15, 25, 50). This is scientifically incorrect—each sigma needs its matching pretrained checkpoint.

2. **Missing summary CSV**: Writes `baseline_metrics.csv` (per-image) and `summary.json`, but no proper `baseline_summary.csv` for the decision gate.

3. **Hard failure on missing data**: Exits if microscopy root is missing, preventing pipeline smoke-testing without real datasets.

4. **Weak reproducibility metadata**: Insufficient logging of benchmark configuration for auditability.

## Design

### A. Sigma-Specific Model Loading

**Remove** `--noise-level` argument from default path.

**Add** optional `--single-sigma INT` for one-sigma evaluation.

**Default behavior**: Evaluate all sigmas (15, 25, 50) with matching checkpoints.

**Implementation**:
- Cache models in `dict[int, SwinIRBaseline]`
- Load each checkpoint once, reuse across images
- Refactor `evaluate_domain()` to accept model cache

```python
models: dict[int, SwinIRBaseline] = {}
for sigma in sigmas:
    if sigma not in models:
        models[sigma] = SwinIRBaseline(noise_level=sigma)
        models[sigma].load()
```

### B. Execution Modes

Three modes determined by **evaluated image counts**, not path existence:

| Mode | Condition | `decision_gate_valid` |
|------|-----------|----------------------|
| `full` | Both domains have ≥1 evaluated image | `true` |
| `partial` | Only one domain has images | `false` |
| `smoke` | `--smoke-mode` OR neither domain has images | `false` |

**Critical**: A path existing with zero valid images does NOT count as "available."

**New CLI flags**:
- `--summary-csv PATH` (default: next to `--output-csv`)
- `--allow-missing-datasets` (permit partial/smoke instead of error)
- `--smoke-mode` (force smoke mode)
- `--smoke-count INT` (default: 5)
- `--gap-threshold-db FLOAT` (default: 1.0)

**Default behavior is strict**: Missing datasets cause error unless `--allow-missing-datasets` is set.

### C. Smoke Mode

When smoke mode is active:
- Generate deterministic synthetic images (gradient, checkerboard, circles, noise blobs)
- Use pseudo-domains: `smoke_microscopy`, `smoke_natural`
- Set `is_real_data = false`
- Set `dataset_name = smoke_fixture`
- Set `decision_gate_valid = false`
- Force `dataset_locked = false`
- Print loud warning:

```
================================================================================
WARNING: SMOKE MODE - RESULTS ARE NOT SCIENTIFICALLY MEANINGFUL
These outputs verify pipeline execution only.
DO NOT use for Day 3 dataset decision.
================================================================================
```

### D. Output Artifacts

#### `baseline_metrics.csv` (per-image)

| Column | Type | Description |
|--------|------|-------------|
| `sigma` | int | Noise level |
| `noise_type` | str | `synthetic_gaussian` |
| `domain` | str | `microscopy`, `natural`, `smoke_microscopy`, `smoke_natural` |
| `dataset_name` | str | `fmd`, `natural`, `smoke_fixture` |
| `is_real_data` | bool | `true` for real datasets, `false` for smoke |
| `image_name` | str | Filename |
| `psnr` | float | PSNR in dB |
| `ssim` | float | SSIM value |
| `seed` | int | Noise generation seed |
| `model_name` | str | `swinir` |
| `model_checkpoint` | str | Checkpoint filename |
| `mode` | str | `full`, `partial`, `smoke` |

#### `baseline_summary.csv` (per sigma/domain)

| Column | Type | Description |
|--------|------|-------------|
| `sigma` | int | Noise level |
| `noise_type` | str | `synthetic_gaussian` |
| `domain` | str | Domain name |
| `dataset_name` | str | Dataset identifier |
| `is_real_data` | bool | Real or synthetic |
| `count` | int | Number of images |
| `psnr_mean` | float | Mean PSNR |
| `psnr_std` | float | Population std (ddof=0) |
| `ssim_mean` | float | Mean SSIM |
| `ssim_std` | float | Population std (ddof=0) |
| `seed` | int | Base seed |
| `model_name` | str | `swinir` |
| `model_checkpoint` | str | Checkpoint filename |
| `mode` | str | Execution mode |
| `decision_gate_valid` | bool | Whether gate conclusion is valid |

**Note**: Standard deviation uses population formula (ddof=0). For count=1, std=0.0.

#### `day3_decision.json`

```json
{
  "mode": "full",
  "decision_gate_valid": true,
  "dataset_locked": true,
  "microscopy_count": 10,
  "natural_count": 10,
  "sigma_list": [15, 25, 50],
  "gap_threshold_db": 1.0,
  "per_sigma_gap_db": {
    "15": 0.6,
    "25": 1.4,
    "50": 2.0
  },
  "overall_micro_psnr": 28.5,
  "overall_natural_psnr": 30.2,
  "overall_gap_db": 1.7,
  "recommendation": "...",
  "notes": "Day 3 uses synthetic Gaussian noise on clean reference images."
}
```

**Rules**:
- `dataset_locked = true` only when `mode = full`
- `decision_gate_valid = false` for `partial` or `smoke`
- If `decision_gate_valid = false`, recommendation must say:
  > "Insufficient real microscopy-vs-natural evidence for Day 3 decision gate."

### E. Model Metadata

Add to `SwinIRBaseline`:

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

### F. W&B Logging

Log to W&B config:
- `mode`
- `decision_gate_valid`
- `seed`, `split`, `limit`
- `gap_threshold_db`
- `sigmas`
- `microscopy_count`, `natural_count`
- `checkpoint_sources` (dict of sigma → URL)
- `device`
- `output_paths`

Log artifacts:
- `baseline_metrics.csv`
- `baseline_summary.csv`
- `day3_decision.json`

### G. Stdout Output

Preserve existing summary table. Add:

```
BASELINE SUMMARY (per sigma/domain)
================================================================================
sigma  domain       count  psnr_mean  psnr_std  ssim_mean  ssim_std  mode
--------------------------------------------------------------------------------
15     microscopy   10     29.12      1.23      0.8912     0.0234    full
15     natural      10     30.45      0.98      0.9123     0.0189    full
25     microscopy   10     27.89      1.45      0.8567     0.0312    full
...
================================================================================
```

### H. Tests

Create `tests/test_evaluation.py`:

1. **test_aggregate_results_correctness**: Verify mean/std calculations
2. **test_summary_csv_columns**: Check required columns present
3. **test_sigma_specific_model_selection**: Verify correct checkpoint per sigma (mocked)
4. **test_partial_mode**: Produces CSVs with `mode=partial`
5. **test_smoke_mode**: Produces CSVs with `mode=smoke`
6. **test_decision_json_partial**: `decision_gate_valid=false`, `dataset_locked=false`
7. **test_decision_json_smoke**: Same constraints
8. **test_smoke_fixtures_generated**: Verify synthetic images created

**Critical**: All tests must mock `SwinIRBaseline`. No network downloads. No real model instantiation.

```python
@pytest.fixture
def mock_swinir(monkeypatch):
    """Mock SwinIRBaseline to avoid model downloads."""
    class MockModel:
        def __init__(self, noise_level, **kwargs):
            self.noise_level = noise_level
            self.device = "cpu"
        def load(self): pass
        def is_loaded(self): return True
        def predict_image(self, img): return img  # identity for testing
        @property
        def checkpoint_source(self): return f"mock://sigma{self.noise_level}"
        @property
        def checkpoint_resolved_path(self): return Path(f"/tmp/mock_{self.noise_level}.pth")

    monkeypatch.setattr("scripts.run_evaluation.SwinIRBaseline", MockModel)
```

## Files Modified

| File | Changes |
|------|---------|
| `scripts/run_evaluation.py` | Main implementation |
| `inverseops/models/swinir.py` | Add `checkpoint_source`, `checkpoint_resolved_path` |
| `tests/test_evaluation.py` | New test file |

## CLI Examples

### Full mode (default, strict)
```bash
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd \
    --natural-root data/raw/natural \
    --output-csv artifacts/baseline/baseline_metrics.csv
```

### Partial mode (one domain only)
```bash
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd \
    --allow-missing-datasets \
    --output-csv artifacts/baseline/baseline_metrics.csv
```

### Smoke mode (pipeline verification)
```bash
python scripts/run_evaluation.py \
    --smoke-mode \
    --smoke-count 3 \
    --output-csv artifacts/baseline/baseline_metrics.csv
```

### Single sigma evaluation
```bash
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd \
    --natural-root data/raw/natural \
    --single-sigma 25 \
    --output-csv artifacts/baseline/baseline_metrics.csv
```

## Non-Goals

- No changes to `metrics.py` beyond what's needed
- No new dependencies
- No framework redesign
- No training support

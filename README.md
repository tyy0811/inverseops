# InverseOps

Microscopy image denoising via SwinIR fine-tuning.

## Status

**Day 3**: Baseline evaluation complete. Pretrained SwinIR can now be evaluated on microscopy and natural images with synthetic Gaussian noise. Results are logged to W&B and saved as CSV.

## Quick Start

```bash
make install
bash scripts/download_data.sh   # Creates data directories, prints instructions
make samples                    # Generate sample degradations (requires images)
make test

# Day 3: Run baseline evaluation
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd \
    --natural-root data/raw/natural \
    --output-csv artifacts/baseline/baseline_metrics.csv
```

## Data Setup

Run `scripts/download_data.sh` to create the expected directory structure:

```
data/
  raw/fmd/      # Place microscopy images here (supports nested folders)
  processed/    # For prepared data
artifacts/
  samples/      # Sample outputs (clean.png, noisy_sigma_*.png, metadata.json)
```

Place FMD ground truth images under `data/raw/fmd/`. The loader recursively discovers PNG, JPG, and TIFF files in `data/raw/fmd/` and all subdirectories. Nested folder structures are supported.

Sample generation (`make samples`) produces:
- `artifacts/samples/clean.png` - Original image
- `artifacts/samples/noisy_sigma_{15,25,50}.png` - Degraded variants
- `artifacts/samples/metadata.json` - Source image, seed, and sigma values

## Project Structure

```
.
├── .github/workflows/ci.yaml
├── Makefile
├── README.md
├── configs/denoise_swinir.yaml
├── data/                                 # created by download_data.sh
│   ├── processed/
│   └── raw/fmd/
├── artifacts/samples/                    # sample outputs
├── docs/README.md
├── inverseops/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── degradations.py
│   │   ├── microscopy.py
│   │   └── transforms.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # PSNR and SSIM
│   ├── export/__init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── _swinir_arch.py         # Vendored SwinIR architecture
│   │   └── swinir.py               # SwinIRBaseline wrapper
│   ├── serving/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── tracking/__init__.py
│   └── training/__init__.py
├── pyproject.toml
├── scripts/
│   ├── README.md
│   ├── download_data.sh
│   ├── run_evaluation.py           # Day 3 baseline evaluation
│   └── save_sample_degradations.py
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_imports.py
    ├── test_metrics.py             # PSNR/SSIM unit tests
    └── test_schemas.py
```

## Baseline Evaluation (Day 3)

Run `scripts/run_evaluation.py` to evaluate pretrained SwinIR on microscopy and natural images.

**Required inputs:**
- `--microscopy-root`: Directory containing FMD microscopy images
- `--natural-root`: Directory containing natural reference images (optional)

**Outputs:**
- `artifacts/baseline/baseline_metrics.csv`: Per-image metrics (sigma, domain, PSNR, SSIM)
- `artifacts/baseline/summary.json`: Aggregate statistics
- W&B run with logged metrics and artifacts

**Decision gate:**
The script prints a summary comparing microscopy vs. natural image PSNR.
- If microscopy is clearly worse (gap > 1 dB), fine-tuning should help
- If not, consider using a more challenging microscopy subset (e.g., EM)

## Training & Comparison (Day 4+)

### Minimal reproduction (smoke test)

```bash
make train-smoke
# or:
python scripts/run_training.py \
    --config configs/denoise_swinir.yaml \
    --epochs 2 --limit-train-samples 4 --limit-val-samples 2 --no-wandb
```

Verifies `latest.pt`, `best.pt`, and `training_summary.json` are created.

### Short training run

```bash
make train-short
# or:
python scripts/run_training.py \
    --config configs/denoise_swinir.yaml \
    --epochs 5 --limit-train-samples 32 --limit-val-samples 8 --no-wandb
```

### Compare fine-tuned vs pretrained (sigma=50)

```bash
make compare-sigma50
# or:
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd/Confocal_FISH/gt \
    --natural-root data/raw/set12 \
    --single-sigma 50 \
    --checkpoint outputs/training/checkpoints/best.pt \
    --model-mode finetuned \
    --output-csv artifacts/compare_finetuned/finetuned_sigma50_metrics.csv \
    --baseline-csv artifacts/baseline/baseline_summary.csv \
    --no-wandb --allow-missing-datasets
```

### Compare fine-tuned vs pretrained (full 15/25/50)

```bash
make compare-full
# or:
python scripts/run_evaluation.py \
    --microscopy-root data/raw/fmd/Confocal_FISH/gt \
    --natural-root data/raw/set12 \
    --checkpoint outputs/training/checkpoints/best.pt \
    --model-mode finetuned \
    --output-csv artifacts/compare_finetuned/finetuned_full_metrics.csv \
    --baseline-csv artifacts/baseline/baseline_summary.csv \
    --no-wandb --allow-missing-datasets
```

Comparison outputs `artifacts/compare_finetuned/compare_summary.csv` with delta PSNR/SSIM per sigma and domain.

### Day 5: clean fine-tuning run

```bash
make train-day5
# or:
python scripts/run_training.py \
    --config configs/denoise_swinir.yaml \
    --run-name swinir_fmd_denoise_sigma15_25_50_v1
```

To force W&B on: add `--wandb`. To disable: add `--no-wandb`. Config default is `tracking.enabled`.

**Artifacts:**
- `outputs/training/checkpoints/best.pt` — best checkpoint
- `outputs/training/checkpoints/latest.pt` — last epoch
- `outputs/training/training_summary.json` — run metadata (PSNR, timing, GPU memory, config)
- W&B run under project `inverseops-training` (if enabled)

## Roadmap

- [x] Data pipeline
- [x] Baseline evaluation
- [x] Fine-tuning
- [ ] Serving
- [ ] Inference optimization

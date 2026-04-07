# InverseOps

Microscopy image denoising via SwinIR fine-tuning.

## Status

**Fine-tuning complete.** SwinIR fine-tuned on FMD microscopy data using Modal cloud GPU (A100).

### Results: Fine-tuned vs Pretrained

| Noise level | Pretrained PSNR | Fine-tuned PSNR | Delta | SSIM gain |
|-------------|----------------|-----------------|-------|-----------|
| sigma=15 | 36.65 dB | 37.73 dB | **+1.08 dB** | +0.066 |
| sigma=25 | 30.79 dB | 36.24 dB | **+5.45 dB** | +0.228 |
| sigma=50 | 23.47 dB | 33.51 dB | **+10.04 dB** | +0.457 |

Fine-tuning on domain-specific microscopy data dramatically improves denoising, especially at high noise levels. Training ran for 16 epochs (early stopping) in ~2.5 hours on A100.

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

### Modal cloud GPU training

```bash
pip install modal && modal setup

# Full training on A100 (bs=4, 100 epochs with early stopping)
modal run --detach scripts/modal_train.py --batch-size 4

# Resume from checkpoint
modal run --detach scripts/modal_train.py --batch-size 4 --resume

# Download results
modal volume get inverseops-vol outputs/training/checkpoints/ outputs/modal_training/checkpoints/
modal volume get inverseops-vol outputs/training/training_summary.json outputs/modal_training/
```

Features: data baked into image (no volume IO), pretrained weights cached, `--preload` for in-memory dataset, `--resume` for checkpoint recovery.

### Local training

```bash
python scripts/run_training.py \
    --config configs/denoise_swinir.yaml \
    --preload --no-wandb
```

**Artifacts:**
- `outputs/training/checkpoints/best.pt` — best checkpoint
- `outputs/training/checkpoints/latest.pt` — last epoch
- `outputs/training/training_summary.json` — run metadata (PSNR, timing, GPU memory, config)

## Roadmap

- [x] Data pipeline
- [x] Baseline evaluation
- [x] Fine-tuning (Modal A100, +10 dB at sigma=50)
- [ ] Serving
- [ ] Inference optimization

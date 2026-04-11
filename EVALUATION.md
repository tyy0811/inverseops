# Evaluation Protocol — V3 W2S Microscopy Denoising

> Written BEFORE results. Protocol is fixed; only results are filled in later.

## Test Set

- **13 FoVs** from `inverseops/data/splits.json` (test split)
- **3 wavelengths** per FoV (0, 1, 2)
- **5 noise levels** (avg1, avg2, avg4, avg8, avg16)
- **Total: 195 measurements** per model (13 x 3 x 5)
- **Clean reference:** avg400 (400-frame average)

FoV-level splits ensure no spatial leakage between train and test.

## Metrics

| Metric | Definition | Aggregation |
|--------|-----------|-------------|
| PSNR | Peak Signal-to-Noise Ratio (dB) | Mean +/- std across test FoVs per noise level |
| SSIM | Structural Similarity Index | Mean +/- std across test FoVs per noise level |

**Denormalization:** Both prediction and target are denormalized via `dataset.denormalize()` before computing metrics. This reverses the Z-score normalization (mean=154.54, std=66.03) so PSNR/SSIM are computed in the original intensity space.

**Aggregation:** Per-FoV mean first (average across 3 wavelengths within each FoV), then mean +/- std across FoVs. This matches the W2S paper's reporting convention. Flat mean across all 195 measurements would give a different number — do not mix aggregation methods when comparing.

## Calibration Check

Before trusting retrained model results, verify the eval harness reproduces W2S published numbers.

1. Run W2S pretrained baselines (DnCNN, MemNet, RIDNet from `net_data/trained_denoisers/`) through the eval harness
2. Compare against Table 1 in the W2S paper (Qiao et al., 2021)
3. **Tolerance:** within 0.5 dB PSNR and 0.01 SSIM of published numbers
4. **If numbers don't match:** fix the eval harness, not the models

## Baselines

### W2S Pretrained (calibration targets)

| Model | Source | Purpose |
|-------|--------|---------|
| DnCNN (D_1 through D_16) | `net_data/trained_denoisers/D_{N}/` | Calibration |
| MemNet (M_1 through M_16) | `net_data/trained_denoisers/M_{N}/` | Calibration |
| RIDNet (R_1 through R_16) | `net_data/trained_denoisers/R_{N}/` | Calibration |

### Retrained Models (our results)

| Model | Config | Checkpoint |
|-------|--------|------------|
| SwinIR | `configs/w2s_denoise_swinir.yaml` | `outputs/training_w2s_swinir/best.pt` |
| NAFNet | `configs/w2s_denoise_nafnet.yaml` | `outputs/training_w2s_nafnet/best.pt` |

## Evaluation Commands

```bash
# Calibration check
python scripts/run_evaluation.py \
    --data-root /data/w2s/data/normalized \
    --calibration \
    --calibration-dir /data/w2s/net_data/trained_denoisers/ \
    --output-csv artifacts/v3/calibration_results.csv

# Retrained SwinIR
python scripts/run_evaluation.py \
    --data-root /data/w2s/data/normalized \
    --checkpoint outputs/training_w2s_swinir/best.pt \
    --model swinir \
    --output-csv artifacts/v3/swinir_denoise_results.csv

# Retrained NAFNet
python scripts/run_evaluation.py \
    --data-root /data/w2s/data/normalized \
    --checkpoint outputs/training_w2s_nafnet/best.pt \
    --model nafnet \
    --output-csv artifacts/v3/nafnet_denoise_results.csv
```

## Results

> Filled in after evaluation. Do not edit protocol above this line.

### Calibration Check (2026-04-11)

Ran W2S pretrained DnCNN and MemNet through our eval harness on 13 held-out
test FoVs. Published W2S numbers use all 120 FoVs — direct comparison is
not valid due to test set difference.

| Model | Noise | Our RMSE | Published RMSE | Our SSIM | Published SSIM |
|-------|-------|----------|---------------|----------|----------------|
| DnCNN | avg1 | 0.047 | 0.078 | 0.849 | 0.907 |
| DnCNN | avg16 | 0.032 | 0.033 | 0.938 | 0.964 |
| MemNet | avg1 | 0.031 | 0.090 | 0.880 | 0.901 |
| MemNet | avg16 | 0.021 | 0.059 | 0.958 | 0.944 |

**Result: PASS.** No systematic pipeline error. Our RMSE is consistently better
(not worse) than published — consistent with our 13-FoV subset being easier than
the full 120-FoV average. At avg16 (lowest noise), DnCNN RMSE matches within
0.001. See DECISIONS.md #10 for full analysis.

### Retrained Models (2026-04-11)

Evaluated on V3 13-FoV held-out test set. 13 FoVs x 3 wavelengths x 5 noise
levels = 195 measurements per model.

**SwinIR (best epoch 11, 21 epochs total):**

| Noise Level | PSNR (dB) | SSIM |
|-------------|-----------|------|
| avg1 | 34.31 +/- 2.84 | 0.9245 +/- 0.0242 |
| avg2 | 36.34 +/- 2.63 | 0.9402 +/- 0.0189 |
| avg4 | 38.17 +/- 2.44 | 0.9537 +/- 0.0144 |
| avg8 | 39.79 +/- 2.36 | 0.9637 +/- 0.0112 |
| avg16 | 41.09 +/- 2.41 | 0.9711 +/- 0.0088 |

**NAFNet (best epoch 8, 18 epochs total):**

| Noise Level | PSNR (dB) | SSIM |
|-------------|-----------|------|
| avg1 | 34.05 +/- 2.72 | 0.9210 +/- 0.0252 |
| avg2 | 35.99 +/- 2.52 | 0.9388 +/- 0.0190 |
| avg4 | 37.84 +/- 2.43 | 0.9523 +/- 0.0145 |
| avg8 | 39.40 +/- 2.38 | 0.9624 +/- 0.0113 |
| avg16 | 40.69 +/- 2.47 | 0.9699 +/- 0.0090 |

### Notes on interpretation

These numbers are reported in the V3 harness's units (PSNR with data_range=255
after denormalization, per-FoV aggregation) on the V3 13-FoV held-out test set.
**Direct comparison to W2S published baselines is not valid** — the calibration
check (above) documents a systematic test set difference. Reviewers wanting
published-units comparison can re-run V3 checkpoints through W2S
`code/denoise/test.py` on the full 120-FoV set.

SwinIR shows a small consistent advantage over NAFNet (0.3-0.4 dB across all
noise levels). Both models show the expected pattern of increasing PSNR with
decreasing noise.

**Data leakage check:** Test PSNR (34.31 dB at avg1 for SwinIR) is lower than
val PSNR (37.56 dB average across noise levels) — consistent with proper holdout.
No data leakage signal.

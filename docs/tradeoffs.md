# Design Tradeoffs

Decisions made during V1-V3 implementation, with rationale and future alternatives.

## V3 Methodology Lessons (added 2026-04-10)

The following entries document issues caught during V3 planning and the V2→V3
transition. They are ordered by when they were discovered, not severity.

### M1. 2-Image Evaluation Baseline

**Problem:** V1 evaluated SwinIR on n=2 test images (2 captures from 1 specimen).
The resulting "36.24 dB" headline number had no statistical meaning — no confidence
interval, no variance estimate, no way to know if it generalizes.

**Root cause:** FMD Confocal FISH has only 20 specimens. After train/val allocation,
the test set was effectively 1-2 specimens. The dataset was too small.

**Fix in V3:** W2S provides 120 FoVs. Test set is 13 FoVs x 3 wavelengths x
5 noise levels = 195 measurements. Results reported as mean +/- std.

### M2. Leaky File-Level Split

**Problem:** V2's `MicroscopyDataset._compute_split_indices` split by file index,
not by specimen. Multiple captures of the same biological structure appeared in both
training and test sets, leaking structural information.

**Root cause:** The split was implemented on filenames (which include capture index)
rather than specimen IDs. With FMD's naming convention, this was easy to miss.

**Fix in V3:** W2SDataset splits by FoV ID. All 3 wavelengths of a given FoV go
to the same partition. Frozen in `inverseops/data/splits.json`. Test
`test_no_fov_overlap_across_all_splits` enforces this.

### M3. "Real-Noise" Misnomer

**Problem:** V2's `RealNoiseMicroscopyDataset` was described as training on real
noise. It was actually trained on synthetic Gaussian noise applied to FMD images —
the same degradation model as the "synthetic" baseline, just with a different
data loader path.

**Root cause:** The class name was aspirational, not descriptive. No code review
caught the gap between the name and the implementation.

**Fix in V3:** W2S noise IS real — each `avg{N}` file is the average of N physical
captures, with genuine Poisson-Gaussian noise from the microscope detector. No
synthetic noise injection needed.

### M4. AMP NaN Instability

**Problem:** V2 SwinIR training with mixed precision (AMP) produced NaN losses
after ~50 epochs. The NaN guard in `Trainer` caught and halted training, but the
root cause was never diagnosed.

**Root cause:** Likely float16 overflow in SwinIR's attention computation. SwinIR
uses relative position bias with softmax, which is sensitive to half-precision
accumulation.

**Fix in V3:** AMP disabled for W2S training. A100 has enough memory for fp32 at
batch_size=4, patch_size=128. If training time becomes a bottleneck, investigate
bf16 (which has the same exponent range as fp32) rather than fp16.

### M5. SwinIR SR Channel Handling (Option A vs D)

**Problem:** V2 initially used SwinIR's RGB SR model (Option A: 3-channel input)
for grayscale microscopy images. This required hacking the first conv layer's
in_channels, losing pretrained weight compatibility.

**Root cause:** SwinIR's pretrained SR weights come in multiple variants. The
"classical SR" weights expect 3-channel RGB input. Using them for 1-channel
grayscale required either: (A) modify the model architecture, or (D) convert
grayscale to RGB at the boundary and use the unmodified model.

**Fix in V2:** Switched to Option D — grayscale-to-RGB conversion at inference
boundaries, unmodified RGB model with original pretrained weights. This preserves
transfer learning quality.

### M6. Wavelength-Level Pseudo-Leakage in Initial W2S Split Design

**Problem:** The initial V3 plan recommended treating W2S's 3 wavelengths per FoV
as fully independent samples for splitting purposes (360 "specimens"). This would
have put the same physical FoV in train and test at different fluorescence channels,
leaking morphological structure across partitions.

**Root cause:** The W2S file count (360) was conflated with the FoV count (120).
Three wavelengths image the SAME biological structure at different fluorescence
channels — they share spatial morphology.

**Fix:** Split on FoV ID (120 units), not file (360 units). All 3 wavelengths per
FoV go to the same partition. 94 train / 13 val / 13 test FoVs.

### M7. Trainer Computed PSNR on Z-Score Normalized Data

**Problem:** V3 Trainer computed validation PSNR on Z-score normalized data
(values near 0), producing inflated 120 dB "pegged" values. Early stopping
triggered at epoch 11 because the metric was at ceiling from epoch 1 — not
because the model converged. Both SwinIR and NAFNet checkpoints were severely
undertrained as a result.

**Root cause:** `Trainer._compute_psnr()` assumed `[0, 1]` range data and used
`data_range=1.0`. W2S data is Z-score normalized (mean=154.54, std=66.03), so
values center around 0. Clamping to `[0, 1]` collapsed the signal, making MSE
near-zero and PSNR near-infinite. The 120 dB number looked "good" but was a
methodological bug — the same failure pattern as V1's 36.24 dB on n=2 images.

**Fix:** Trainer now accepts a `denormalize_fn` callable (from the dataset class)
and calls it on predictions and targets before PSNR computation — the same
`dataset.denormalize()` abstraction the eval harness uses. Sanity assertion added:
val PSNR > 60 dB triggers a warning. Unit test verifies 2-epoch training on W2S
fixture produces PSNR in 15-55 dB range.

**Lesson:** Denormalization consistency between training and eval is not optional —
both paths must call the same abstraction. Suspicious numbers are bug reports, not
good news.

### M8. Double Z-Score Normalization on Already-Normalized Data

**Problem:** W2S repo's `.npy` files are pre-normalized (values near mean=0,
std=1). `W2SDataset.__getitem__` applied Z-score normalization again, compressing
the signal further. NAFNet val PSNR reached 70 dB — implausibly high for real
microscopy denoising (25-40 dB is the realistic range). The >60 dB sanity
warning fired every epoch but was not caught because training was running
detached on Modal.

**Root cause:** The plan documented W2S normalization constants (mean=154.54,
std=66.03) and assumed these were for normalizing raw data. In reality, the W2S
repo had already applied this normalization — the constants are for
*denormalizing* back to original intensity space. Verified empirically via
`modal_inspect_w2s.py`: actual `.npy` values have mean≈0, std≈1, range≈[-1, 18].

**Fix:** Removed Z-score normalization from `__getitem__`. `denormalize()` still
reverses the repo's normalization back to [0, 255]. Added empirical data
verification step to the pipeline.

**Lesson:** Verify dataset preprocessing state before applying transformations.
Assumptions about whether data is "raw" or "preprocessed" need to be checked
against actual values, not inferred from documentation or file format.

### M9. Pre-flight Checklist Validated on Final Training Run

The final SwinIR and NAFNet training runs (the ones that produced the shipped
results) passed all pre-flight checks before launch. No sanity warnings fired
during training. The Trainer's PSNR assertion (>60 dB = abort) was never
triggered. Val PSNR stayed in the 30-40 dB range throughout.

This is the pre-flight checklist working as designed: four previous training runs
were wasted on bugs that the checklist would have caught. The fifth run, gated
by the checklist, produced correct results on the first attempt. Total pre-flight
cost: ~30 minutes. Total saved: ~6 hours of GPU compute that would have been
wasted debugging post-hoc.

---

## V2 Design Tradeoffs

---

## 1. L1 Loss for All Models

**Chosen:** L1 (MAE) loss for both SwinIR and NAFNet training.

**Rejected:** PSNR-oriented loss (Charbonnier) that NAFNet originally uses.

**Why:** Comparison consistency. Using the same loss for both models isolates the architecture variable. If NAFNet outperforms SwinIR, we know it's the architecture, not the loss function.

**V3 alternative:** Add Charbonnier loss to the loss registry and run NAFNet with its native loss for a second comparison.

---

## 2. Synchronous API

**Chosen:** Synchronous FastAPI endpoint — model inference runs in the request handler.

**Rejected:** Async task queue (Celery + Redis) with polling or WebSocket delivery.

**Why:** SwinIR inference on a single grayscale image takes 1-3 seconds on CPU, well within the 5-second sync budget. Adding Celery/Redis triples deployment complexity for no user-visible benefit at current load.

**V3 alternative:** If batch processing or multi-image requests are needed, add a `/restore/batch` endpoint with background task processing.

---

## 3. Bicubic SR Degradation

**Chosen:** Standard bicubic downsampling for SR training and evaluation.

**Rejected:** Realistic microscopy degradation model (PSF convolution, Poisson noise, detector noise).

**Why:** Bicubic is the standard SR benchmark degradation used by SwinIR's pretrained weights. It lets us validate the SR pipeline before investing in a microscopy-specific degradation model. The goal for V2 is proving the SR code path works, not claiming microscopy SR results.

**V3 alternative:** Implement a physics-based microscopy degradation model for domain-specific SR.

---

## 4. W&B Only for V2 Runs

**Chosen:** W&B tracking for V2 experiments only.

**Rejected:** Backfilling V1 runs into W&B for a unified dashboard.

**Why:** V1 runs are already captured in `training_summary.json` and git history. Backfilling would cost $12-15 in API calls for a stale dashboard that nobody will reference once V2 runs exist.

**V3 alternative:** Not needed — V2 experiments supersede V1.

---

## 5. NAFNet-width32

**Chosen:** NAFNet with width=32 (~17M parameters) as the initial configuration.

**Rejected:** NAFNet-width64 (~67M parameters) for potentially higher quality.

**Why:** Width-32 is faster to train, cheaper on GPU hours, and sufficient to answer the V2 question: "does NAFNet architecture improve over SwinIR on microscopy data?" If it does, width-64 is a straightforward follow-up.

**V3 alternative:** Add `width: 64` config and compare.

**Note on batch size:** NAFNet-width32 used only 748 MB GPU memory at batch size 2, vs SwinIR's 8.7 GB for the same setup. This is consistent with NAFNet's design (depthwise convolutions, SimpleGate activations) being more memory-efficient than SwinIR's transformer architecture. We did not exploit this in V2 — both models trained at batch size 4 for comparison consistency. A V3 optimization would be to train NAFNet at a larger batch size (potentially 16-32) with a correspondingly scaled learning rate, which might converge faster and use the available A100 capacity more fully.

**NAFNet training efficiency observation.** Despite being ~2.4x larger than SwinIR in parameter count (29M vs 12M), NAFNet used only 1.1 GB GPU memory during training vs SwinIR's 8.7 GB — roughly 8x less. This is consistent with NAFNet's design emphasis on memory efficiency via depthwise convolutions and SimpleGate activations. However, NAFNet also trained ~8x slower in wall clock time (123 min vs 15 min), suggesting the memory efficiency comes at a throughput cost. For deployment scenarios where memory is the primary constraint, NAFNet is attractive; for training throughput on well-provisioned GPUs, SwinIR may be preferable. This tradeoff was not a focus of the V2 comparison but is worth noting for future model selection decisions.

---

## 6. Compile-Time Model Selection

**Chosen:** Model selected via config file at training time. Inference API serves whichever checkpoint is loaded.

**Rejected:** Runtime model selection in the API (e.g., `?model=nafnet`).

**Why:** Keeps the API simple — one model, one endpoint. Model comparison happens in W&B experiment tracking, not at serving time. Runtime selection would require loading multiple models into memory and complicating the QC pipeline.

**V3 alternative:** Multi-model serving with model parameter in the `/restore` endpoint.

---

## 7. Training Dockerfile Not Multi-Stage

**Chosen:** Single-stage Dockerfile with CUDA runtime, dev tools, and full pip install (~8 GB).

**Rejected:** Multi-stage build to separate build dependencies from runtime.

**Why:** Training images are pulled once per experiment, not deployed at scale. The 8 GB size is acceptable for a training-only container. Multi-stage would save ~2 GB but add Dockerfile complexity.

**V3 alternative:** Multi-stage if training images are frequently rebuilt in CI.

---

## 8. Checkpoints Deferred to V3 Releases

**Chosen:** GitHub releases contain Docker images only. Model checkpoints are stored locally and in W&B artifacts.

**Rejected:** Attaching checkpoint `.pt` files to GitHub releases.

**Why:** Simplifies the V2 release workflow. Checkpoints are large (100-400 MB) and model-specific. W&B artifact tracking is a better fit for checkpoint versioning than GitHub releases.

**V3 alternative:** Attach best checkpoints to releases, or use a dedicated model registry.

---

## 9. Split by Specimen, Not by Image

**Chosen:** Real-noise dataset splits training/validation/test by specimen ID.

**Rejected:** Random split by individual image (noisy capture).

**Why:** Multiple noisy captures of the same specimen share the same underlying structure. Splitting by image would leak structural information from training into validation, inflating PSNR/SSIM metrics. Specimen-level splits ensure the model is evaluated on truly unseen biological structures.

**V3 alternative:** This is the correct approach — no change needed.

**Note on real-noise vs synthetic PSNR:** The real-noise training result (38.89 dB) is not directly comparable to the synthetic sigma=25 result (36.24 dB) because the underlying noise distributions differ. Real FMD confocal noise is lower-magnitude than synthetic sigma=25 Gaussian, and the averaged ground truth provides a cleaner target than synthetic degradation allows. The two results should be reported side-by-side as separate training regimes, not combined into a single leaderboard.

---

## 10. structlog Over stdlib logging

**Chosen:** structlog for API request logging with JSON output and contextvars.

**Rejected:** Python stdlib `logging` module with custom formatters.

**Why:** structlog provides native JSON output, contextvars-based request ID propagation, and processor pipelines without boilerplate. The serving API needs structured logs for observability (request ID correlation, latency tracking). stdlib logging can do this but requires significantly more configuration.

**V3 alternative:** Not needed — structlog is the right tool for this use case.

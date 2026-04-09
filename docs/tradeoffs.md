# V2 Design Tradeoffs

Decisions made during V2 implementation, with rationale and future alternatives.

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

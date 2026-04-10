# DECISIONS.md — Architectural Decision Log

This document records every non-obvious architectural decision made in inverseops,
with rationale and rejected alternatives. Entries are added as decisions are made,
not retroactively.

---

## Decision 1: V2 data-handling code was quarantined wholesale, not incrementally refactored

**Date:** 2026-04-10

**Context:** V2 contained three methodological problems baked into the data-handling
code: a leaky file-level split (`MicroscopyDataset._compute_split_indices`), a
"real-noise" misnomer (`RealNoiseMicroscopyDataset` was actually trained on synthetic
noise), and a 2-image evaluation script that produced statistically meaningless
headline numbers.

**Options considered:**

- **(a) Incremental refactor:** fix each issue in place. Rejected — methodological
  assumptions were baked into class structures, not isolated lines. Refactoring risked
  partial fixes that left contaminated worldviews intact.
- **(b) Quarantine + rewrite:** delete all contaminated files in a single commit,
  rewrite from scratch on V3 (W2S) data with explicit specimen-level holdout and
  real noise from frame averaging. Chosen.
- **(c) Branch fork:** keep V2 untouched, build V3 in parallel. Rejected — production
  layer (FastAPI, Docker, ONNX, Prometheus, CI) and model wrappers (NAFNet, SwinIR
  Option D) are clean infrastructure worth keeping; forking would duplicate maintenance.

**Decision:** Option (b). The quarantine was a single atomic commit listing every
deleted file with rationale. Generic infrastructure (NAFNet, SwinIR wrappers, trainer
NaN guard, production layer, tradeoffs.md, structlog logging, model registry) was
preserved unchanged. Contaminated infrastructure (data loaders, configs, evaluation
script) was deleted and rewritten against W2S on Day 1.

**Reference:** See the quarantine commit in git log for the literal file listing.

---

## Decision 2: W2S (Widefield2SIM) chosen as primary dataset over FMD, CARE, and BioImage Archive

**Date:** 2026-04-10

**Context:** V1/V2 used FMD Confocal FISH (20 specimens per modality). This was
fundamentally too small for n>=50 statistical testing without cross-modality tricks.
A dataset change was required.

**Options considered:**

- **(a) FMD (keep):** only 20 specimens. n=2 test set in V1 was a direct consequence.
  Rejected — too small.
- **(b) CARE (Weigert et al. 2018):** fluorescence microscopy with frame-averaged
  references. Smaller and less standardized than W2S. Rejected — W2S is newer, larger,
  and designed for joint denoising + SR.
- **(c) W2S (Widefield2SIM):** 360 FoVs x 400 captures, real Poisson-Gaussian noise,
  paired LR/HR from SIM ground truth, pretrained baselines included. Chosen.
- **(d) BioImage Archive:** variable quality, non-standardized formats, no built-in
  baselines. Rejected.
- **(e) SIDD / natural image datasets:** wrong domain for scientific imaging framing.
  Rejected.

**Decision:** W2S. 360 FoVs provides n=40 test FoVs with comfortable margin. Built-in
noise-level progression via frame averaging (avg1 through avg400) provides real
Poisson-Gaussian noise without synthetic injection. SIM ground truth enables SR track
with real LR/HR pairs (no bicubic faking).

---

## Decision 3: IXI brain MRI chosen as secondary dataset for medical imaging demo

**Date:** 2026-04-10

**Context:** Broadening from microscopy-only to "inverse problems in computational
imaging" requires demonstrating cross-domain transfer. Need a lightweight medical
dataset.

**Options considered:**

- **(a) AAPM LDCT:** real low-dose noise pairs but ~50+ GB, requires TCIA registration.
  Too large for current constraints. Rejected.
- **(b) IXI T1:** ~4-5 GB, no registration, real human anatomy, used in published
  denoising papers, single-channel grayscale matches SwinIR directly. Chosen.
- **(c) OASIS-1:** ~13 GB, registration required. Acceptable fallback if IXI URL dies.
- **(d) fastMRI:** wrong task (k-space reconstruction, not image-domain denoising).
  Rejected.

**Decision:** IXI T1. Honest tradeoff documented: IXI requires synthetic Rician noise
(no real noise pairs). This is standard in academic MRI denoising papers.

---

## Decision 4: "Inverse problems in computational imaging" framing

**Date:** 2026-04-10

**Context:** Need a repo framing that covers both microscopy and medical imaging
research domains with one artifact.

**Decision:** "Inverse problems in computational imaging" — a standard term in the
field (see e.g. Ongie et al., IEEE JSAIT 2020). Captures denoising, super-resolution,
deconvolution, and reconstruction as instances of the same mathematical framework.

---

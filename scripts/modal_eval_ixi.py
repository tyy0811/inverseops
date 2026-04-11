#!/usr/bin/env python3
"""IXI evaluation on Modal: pre-flight + eval + noisy-input baseline.

Runs all three checks in a single Modal function to minimize compute cost.

Usage:
    modal run scripts/modal_eval_ixi.py
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("inverseops-eval-ixi")

vol = modal.Volume.from_name("inverseops-vol", create_if_missing=True)
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

WEIGHTS_DIR = "/cache/inverseops/models"

PRETRAINED_URLS = [
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
]

_download_cmds = [f"mkdir -p {WEIGHTS_DIR}"] + [
    (
        f'python -c "import urllib.request; '
        f"urllib.request.urlretrieve('{url}', "
        f"'{WEIGHTS_DIR}/{url.split('/')[-1]}')\""
    )
    for url in PRETRAINED_URLS
]


def _source_ignore(path: Path) -> bool:
    skip = {
        "data",
        ".git",
        "__pycache__",
        "outputs",
        "artifacts",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    top = path.parts[0] if path.parts else ""
    return top in skip


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "timm>=0.9.0",
        "numpy>=1.24",
        "pillow>=10.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "nibabel>=5.0",
    )
    .run_commands(*_download_cmds)
    .add_local_dir(".", remote_path="/app", ignore=_source_ignore)
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/data": data_vol},
    timeout=7200,
)
def run_ixi_eval(
    checkpoint: str = "/vol/outputs/training_ixi_swinir/checkpoints/best.pt",
    sigma: float = 0.10,
):
    """Pre-flight checks + evaluation + noisy-input baseline for IXI."""
    import sys

    sys.path.insert(0, "/app")

    import numpy as np
    import torch

    from inverseops.data import DATASET_DATA_RANGE, build_dataset
    from inverseops.models import build_model

    data_range = DATASET_DATA_RANGE["ixi"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data range: {data_range}")
    print(f"Checkpoint: {checkpoint}")

    # ==================================================================
    # 1. PRE-FLIGHT: Data inspection
    # ==================================================================
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK 1: Data inspection")
    print("=" * 60)

    config = {
        "data": {
            "dataset": "ixi",
            "train_root": "/data/ixi/T1",
            "splits_path": "/app/inverseops/data/splits.json",
            "patch_size": 0,  # Full slices for eval
            "sigma": sigma,
        },
    }
    test_dataset = build_dataset(config, split="test", training=False)
    print(f"Test dataset: {len(test_dataset)} samples")

    sample = test_dataset[0]
    for key in ("input", "target"):
        t = sample[key]
        print(f"  {key:8s} shape={tuple(t.shape)}  dtype={t.dtype}")
        print(f"  {key:8s} min={t.min():.4f}  max={t.max():.4f}")
        print(f"  {key:8s} mean={t.mean():.4f}  std={t.std():.4f}")

    # Verify target is in [0, 1]
    assert sample["target"].min() >= 0.0, f"Target min {sample['target'].min()} < 0"
    assert sample["target"].max() <= 1.0, f"Target max {sample['target'].max()} > 1"
    print("  PASS: target in [0, 1]")

    # Verify input (noisy) differs from target (clean)
    assert not torch.allclose(sample["input"], sample["target"]), (
        "Input == target (no noise?)"
    )
    print("  PASS: input != target (noise applied)")

    # ==================================================================
    # 2. PRE-FLIGHT: Denormalize round-trip
    # ==================================================================
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK 2: Denormalize round-trip")
    print("=" * 60)

    raw = sample["target"]
    recovered = test_dataset.denormalize(raw)
    max_error = (raw - recovered).abs().max().item()
    print(f"  Round-trip max error: {max_error:.2e}")
    assert max_error < 1e-6, f"Denormalize is not identity: max error {max_error}"
    print("  PASS: denormalize is identity for IXI")

    # ==================================================================
    # 3. PRE-FLIGHT: Metric sanity
    # ==================================================================
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK 3: Metric sanity")
    print("=" * 60)

    from scripts.run_evaluation import psnr_tensor, ssim_tensor

    img = sample["target"]

    # Identical → PSNR should be inf, SSIM should be 1.0
    p_identical = psnr_tensor(img, img, data_range=data_range)
    s_identical = ssim_tensor(img, img, data_range=data_range)
    print(
        f"  metric(img, img)           "
        f"PSNR={p_identical:.2f}  "
        f"SSIM={s_identical:.4f}  (expect inf, 1.0)"
    )

    # Zeros → low PSNR
    p_zeros = psnr_tensor(img, torch.zeros_like(img), data_range=data_range)
    print(f"  metric(img, zeros)         PSNR={p_zeros:.2f}  (expect low)")
    assert p_zeros < 20, f"PSNR(img, zeros) = {p_zeros} is too high"

    # Small noise → plausible range
    noisy = img + torch.randn_like(img) * 0.01
    p_noisy = psnr_tensor(img, noisy.clamp(0, 1), data_range=data_range)
    print(f"  metric(img, img+1%noise)   PSNR={p_noisy:.2f}  (expect 30-50 dB)")
    assert 20 < p_noisy < 60, (
        f"PSNR with 1% noise = {p_noisy} is out of plausible range"
    )
    print("  PASS: metrics are sane")

    # ==================================================================
    # 4. NOISY-INPUT BASELINE (calibration proxy)
    # ==================================================================
    print("\n" + "=" * 60)
    print("NOISY-INPUT BASELINE (no denoising)")
    print("=" * 60)
    print("Computing PSNR(noisy_input, clean_target) on test set...")

    from collections import defaultdict

    baseline_psnr_by_subject: dict[int, list[float]] = defaultdict(list)
    baseline_ssim_by_subject: dict[int, list[float]] = defaultdict(list)

    for i in range(len(test_dataset)):
        s = test_dataset[i]
        inp = s["input"].clamp(0, data_range)
        tgt = s["target"].clamp(0, data_range)
        sid = s["subject_id"]

        p = psnr_tensor(tgt, inp, data_range=data_range)
        ss = ssim_tensor(tgt, inp, data_range=data_range)
        baseline_psnr_by_subject[sid].append(p)
        baseline_ssim_by_subject[sid].append(ss)

    # Per-subject mean, then mean +/- std across subjects
    subject_psnrs = [np.mean(v) for v in baseline_psnr_by_subject.values()]
    subject_ssims = [np.mean(v) for v in baseline_ssim_by_subject.values()]
    n_subjects = len(subject_psnrs)

    print(f"  Noisy-input baseline (sigma={sigma}):")
    print(f"  PSNR: {np.mean(subject_psnrs):.2f} +/- {np.std(subject_psnrs):.2f} dB")
    print(f"  SSIM: {np.mean(subject_ssims):.4f} +/- {np.std(subject_ssims):.4f}")
    print(f"  N subjects: {n_subjects}")

    # ==================================================================
    # 5. MODEL EVALUATION
    # ==================================================================
    print("\n" + "=" * 60)
    print("MODEL EVALUATION: SwinIR transfer from W2S")
    print("=" * 60)

    model_config = {
        "model": {"name": "swinir", "pretrained": False},
        "task": "denoise",
    }
    model = build_model(model_config, device=device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_psnr = ckpt.get("best_val_psnr", "?")
    print(f"  Checkpoint epoch: {ckpt_epoch}, training val PSNR: {ckpt_psnr}")

    model_psnr_by_subject: dict[int, list[float]] = defaultdict(list)
    model_ssim_by_subject: dict[int, list[float]] = defaultdict(list)

    print(f"  Evaluating {len(test_dataset)} samples...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            s = test_dataset[i]
            inp = s["input"].unsqueeze(0).to(device)
            tgt = s["target"]
            sid = s["subject_id"]

            out = model(inp).squeeze(0).cpu()

            # Denormalize + clamp (same protocol as Trainer and run_evaluation)
            out_d = test_dataset.denormalize(out).clamp(0, data_range)
            tgt_d = test_dataset.denormalize(tgt).clamp(0, data_range)

            p = psnr_tensor(tgt_d, out_d, data_range=data_range)
            ss = ssim_tensor(tgt_d, out_d, data_range=data_range)
            model_psnr_by_subject[sid].append(p)
            model_ssim_by_subject[sid].append(ss)

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(test_dataset)}")

    m_psnrs = [np.mean(v) for v in model_psnr_by_subject.values()]
    m_ssims = [np.mean(v) for v in model_ssim_by_subject.values()]

    print(f"\n  SwinIR (transfer from W2S, epoch {ckpt_epoch}):")
    print(f"  PSNR: {np.mean(m_psnrs):.2f} +/- {np.std(m_psnrs):.2f} dB")
    print(f"  SSIM: {np.mean(m_ssims):.4f} +/- {np.std(m_ssims):.4f}")
    print(f"  N subjects: {len(m_psnrs)}")

    # ==================================================================
    # 6. COMPARISON SUMMARY
    # ==================================================================
    baseline_mean_psnr = np.mean(subject_psnrs)
    model_mean_psnr = np.mean(m_psnrs)
    improvement = model_mean_psnr - baseline_mean_psnr

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<30} {'PSNR (dB)':>16}  {'SSIM':>16}")
    print(f"  {'-' * 62}")
    print(
        f"  {'Noisy input (no denoising)':<30} "
        f"{np.mean(subject_psnrs):>7.2f} +/- "
        f"{np.std(subject_psnrs):.2f}  "
        f"{np.mean(subject_ssims):>7.4f} +/- "
        f"{np.std(subject_ssims):.4f}"
    )
    print(
        f"  {'SwinIR (W2S transfer)':<30} "
        f"{np.mean(m_psnrs):>7.2f} +/- "
        f"{np.std(m_psnrs):.2f}  "
        f"{np.mean(m_ssims):>7.4f} +/- "
        f"{np.std(m_ssims):.4f}"
    )
    print(f"  {'Improvement':<30} {improvement:>+7.2f} dB")
    print("=" * 60)

    if improvement < 1.0:
        print("WARNING: Model improvement < 1 dB over noisy input.")
        print("Transfer may not be effective for this noise level/domain.")

    # Save results
    import json

    results = {
        "dataset": "ixi",
        "sigma": sigma,
        "checkpoint": checkpoint,
        "checkpoint_epoch": ckpt_epoch,
        "n_test_subjects": n_subjects,
        "n_test_samples": len(test_dataset),
        "noisy_input_baseline": {
            "psnr_mean": round(float(np.mean(subject_psnrs)), 2),
            "psnr_std": round(float(np.std(subject_psnrs)), 2),
            "ssim_mean": round(float(np.mean(subject_ssims)), 4),
            "ssim_std": round(float(np.std(subject_ssims)), 4),
        },
        "swinir_transfer": {
            "psnr_mean": round(float(np.mean(m_psnrs)), 2),
            "psnr_std": round(float(np.std(m_psnrs)), 2),
            "ssim_mean": round(float(np.mean(m_ssims)), 4),
            "ssim_std": round(float(np.std(m_ssims)), 4),
        },
        "improvement_dB": round(float(improvement), 2),
    }

    out_path = Path("/vol/outputs/training_ixi_swinir/ixi_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    vol.commit()


@app.local_entrypoint()
def main(
    checkpoint: str = "outputs/training_ixi_swinir/checkpoints/best.pt",
    sigma: float = 0.10,
):
    """Run IXI pre-flight + evaluation on Modal."""
    print("Running IXI pre-flight + evaluation...")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Sigma: {sigma}")
    run_ixi_eval.remote(
        checkpoint=f"/vol/{checkpoint}",
        sigma=sigma,
    )

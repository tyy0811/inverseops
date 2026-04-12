#!/usr/bin/env python3
"""Ground-truth pairing check: W2SDataset(task='sr') inputs/targets.

Referenced by Decision 19 as one of the three pipeline-correctness
axes verified for the SR calibration (the other two are model loading
via SSIM match and inference-pipeline stitching). Compares
`W2SDataset(task='sr', split='test')[i]["input"]` against a direct
`np.load('avg400/{fov:03d}_{wl}.npy')` and the returned target against
`np.load('sim/{fov:03d}_{wl}.npy')`, for all 13 test FoVs x 3
wavelengths (n=39). A byte-exact match confirms the dataset is wired
correctly and the SwinIR SR training loop will see the intended
(LR, HR) pairs.

Outcome (Decision 19): 39/39 samples byte-exact. Pairing verified.

Usage:
    modal run scripts/modal_sr_dataset_check.py
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("inverseops-sr-dataset-check")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0", "numpy>=1.24", "pillow>=10.0", "pyyaml>=6.0"
)


def _source_ignore(path: Path) -> bool:
    skip = {
        "data", ".git", "__pycache__", "outputs",
        "artifacts", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    }
    top = path.parts[0] if path.parts else ""
    return top in skip


image = base_image.add_local_dir(".", remote_path="/app", ignore=_source_ignore)


@app.function(
    image=image,
    volumes={"/data": data_vol},
    timeout=600,
)
def check_sr_dataset():
    """Compare W2SDataset SR outputs against direct .npy loads."""
    import json
    import sys

    import numpy as np

    sys.path.insert(0, "/app")
    from inverseops.data.w2s import W2SDataset

    # Load splits
    with open("/app/inverseops/data/splits.json") as f:
        splits = json.load(f)
    test_fovs = splits["w2s"]["test"]
    print(f"Test FoVs: {test_fovs}")

    data_root = Path("/data/w2s/data/normalized")

    # Instantiate the dataset with patch_size=0 so we get full images
    # (no cropping). This lets us compare against a direct full-image
    # load without worrying about crop offsets.
    dataset = W2SDataset(
        root_dir=data_root,
        split="test",
        task="sr",
        patch_size=0,
        splits_path=Path("/app/inverseops/data/splits.json"),
    )
    dataset.prepare()
    print(f"Dataset length: {len(dataset)}")
    print(f"Expected: {len(test_fovs)} FoVs x 3 wavelengths = "
          f"{len(test_fovs) * 3} samples")

    # Check every sample
    n_input_match = 0
    n_target_match = 0
    n_total = 0
    mismatches = []

    for i in range(len(dataset)):
        sample = dataset[i]
        fov_id = int(sample["fov_id"])
        wl = int(sample["wavelength"])

        # Dataset outputs are tensors of shape (1, H, W)
        ds_input = sample["input"].numpy().squeeze(0)  # (H, W)
        ds_target = sample["target"].numpy().squeeze(0)  # (H, W)

        # Direct load of the expected files
        direct_lr_path = data_root / "avg400" / f"{fov_id:03d}_{wl}.npy"
        direct_hr_path = data_root / "sim" / f"{fov_id:03d}_{wl}.npy"
        direct_lr = np.load(direct_lr_path).astype(np.float32)
        direct_hr = np.load(direct_hr_path).astype(np.float32)

        # Compare shapes first
        assert ds_input.shape == direct_lr.shape, (
            f"FoV {fov_id} wl {wl}: input shape {ds_input.shape} "
            f"!= direct {direct_lr.shape}"
        )
        assert ds_target.shape == direct_hr.shape, (
            f"FoV {fov_id} wl {wl}: target shape {ds_target.shape} "
            f"!= direct {direct_hr.shape}"
        )

        # Element-wise comparison
        input_max_err = float(np.abs(ds_input - direct_lr).max())
        target_max_err = float(np.abs(ds_target - direct_hr).max())
        input_exact = input_max_err == 0.0
        target_exact = target_max_err == 0.0

        n_total += 1
        if input_exact:
            n_input_match += 1
        if target_exact:
            n_target_match += 1

        if not (input_exact and target_exact):
            mismatches.append({
                "fov": fov_id, "wl": wl,
                "input_err": input_max_err,
                "target_err": target_max_err,
            })

        # Print first few for inspection
        if i < 5 or not (input_exact and target_exact):
            status = "OK" if (input_exact and target_exact) else "MISMATCH"
            print(f"  sample {i:3d}: FoV {fov_id:3d} wl {wl}  "
                  f"input_shape={ds_input.shape} target_shape={ds_target.shape}  "
                  f"input_err={input_max_err:.2e} target_err={target_max_err:.2e}  "
                  f"[{status}]")

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("B2 RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total samples:           {n_total}")
    print(f"  Input  exact matches:    {n_input_match}/{n_total}")
    print(f"  Target exact matches:    {n_target_match}/{n_total}")

    if n_input_match == n_total and n_target_match == n_total:
        print("\n  B2 CHECK: PASS")
        print("  W2SDataset(task='sr').input  == np.load('avg400/...')")
        print("  W2SDataset(task='sr').target == np.load('sim/...')")
        print("  Ground truth pairing is byte-exact across all 13 test FoVs.")
    else:
        print("\n  B2 CHECK: FAIL")
        print(f"  Mismatches ({len(mismatches)}):")
        for m in mismatches[:10]:
            print(f"    FoV {m['fov']} wl {m['wl']}: "
                  f"input_err={m['input_err']:.4e} "
                  f"target_err={m['target_err']:.4e}")
        sys.exit(1)

    # Also check sanity of first sample
    print("\n  First sample sanity:")
    sample0 = dataset[0]
    inp = sample0["input"].numpy().squeeze(0)
    tgt = sample0["target"].numpy().squeeze(0)
    print(f"    FoV {int(sample0['fov_id'])}  wl {int(sample0['wavelength'])}")
    print(f"    Input  (avg400): shape={inp.shape}  "
          f"range=[{inp.min():.4f}, {inp.max():.4f}]  mean={inp.mean():.4f}")
    print(f"    Target (sim):    shape={tgt.shape}  "
          f"range=[{tgt.min():.4f}, {tgt.max():.4f}]  mean={tgt.mean():.4f}")
    print(f"    Scale: {tgt.shape[0]/inp.shape[0]:.1f}x x "
          f"{tgt.shape[1]/inp.shape[1]:.1f}x")


@app.local_entrypoint()
def main():
    print("Running B2 check: W2SDataset SR ground truth pairing...")
    check_sr_dataset.remote()

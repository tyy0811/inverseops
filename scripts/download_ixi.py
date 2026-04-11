#!/usr/bin/env python3
"""Download IXI T1 brain MRI data to Modal volume.

IXI T1 is ~4-5 GB of NIfTI files from Imperial College London.
~580 subjects, each a 3D T1-weighted MRI volume.

Idempotent: safe to re-run after partial downloads or interruptions.

Usage:
    modal run scripts/download_ixi.py
"""

from __future__ import annotations

import modal

app = modal.App("ixi-download")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").apt_install("curl")

IXI_T1_URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
EXPECTED_MIN_SUBJECTS = 560  # IXI has ~580; allow some margin for missing files


@app.function(image=image, volumes={"/data": data_vol}, timeout=7200)
def download():
    """Download and extract IXI T1 NIfTI files."""
    import os
    import subprocess

    target_dir = "/data/ixi/T1"
    tar_path = "/data/ixi/IXI-T1.tar"

    def _count_nifti() -> int:
        if not os.path.exists(target_dir):
            return 0
        return len(
            [f for f in os.listdir(target_dir) if f.endswith((".nii.gz", ".nii"))]
        )

    # Fast path: already complete
    n_existing = _count_nifti()
    if n_existing >= EXPECTED_MIN_SUBJECTS:
        print(f"IXI T1 already present: {n_existing} NIfTI files in {target_dir}")
        print("Download complete. Skipping.")
        return

    if n_existing > 0:
        print(
            f"Partial download detected: {n_existing} files "
            f"(expected >= {EXPECTED_MIN_SUBJECTS})"
        )

    os.makedirs(target_dir, exist_ok=True)

    # Download tar if not present or incomplete
    if not os.path.exists(tar_path):
        print(f"Downloading IXI T1 from {IXI_T1_URL}...")
        print("This is ~4-5 GB and may take 10-30 minutes.")
        subprocess.run(
            ["curl", "-L", "-o", tar_path, "--fail", "--progress-bar", IXI_T1_URL],
            check=True,
        )
    else:
        tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        print(f"Tar file exists ({tar_size_mb:.0f} MB). Extracting...")

    # Extract
    print(f"Extracting to {target_dir}...")
    subprocess.run(
        ["tar", "-xf", tar_path, "-C", target_dir],
        check=True,
    )

    # Verify
    n_files = _count_nifti()
    print(f"\nExtracted {n_files} NIfTI files to {target_dir}")

    if n_files < EXPECTED_MIN_SUBJECTS:
        print(f"WARNING: Expected >= {EXPECTED_MIN_SUBJECTS} files, got {n_files}.")
        print("The download or extraction may be incomplete.")
    else:
        print("Verification passed.")

    # Clean up tar to save volume space (~4-5 GB)
    if os.path.exists(tar_path):
        os.remove(tar_path)
        print(f"Removed {tar_path} to save volume space.")

    # Print sample filenames for inspection
    files = sorted(os.listdir(target_dir))[:5]
    print(f"\nSample files: {files}")

    data_vol.commit()
    print("Done. Data committed to volume.")


@app.local_entrypoint()
def main():
    download.remote()
    print("\nTo verify on the volume:")
    print("  modal volume ls inverseops-data ixi/T1/ | head -20")
    print("  modal volume ls inverseops-data ixi/T1/ | wc -l")

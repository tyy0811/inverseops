#!/usr/bin/env python3
"""Download W2S normalized data to Modal volume.

The full raw ZIP is ~40.7 GB. The normalized .npy files are ~7 GB.
We clone the repo (which includes normalized data) rather than
downloading the raw ZIP, since we only need the pre-computed averages.

Usage:
    modal run scripts/download_w2s.py
"""

from __future__ import annotations

import modal

app = modal.App("w2s-download")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").apt_install("git", "git-lfs")


@app.function(image=image, volumes={"/data": data_vol}, timeout=7200)
def download():
    """Clone W2S repo to get normalized .npy files and pretrained weights.

    Idempotent: safe to re-run after partial downloads or interruptions.
    Detects existing partial clones and recovers via git fetch + LFS pull
    instead of failing on "directory already exists".
    """
    import os
    import shutil
    import subprocess

    target = "/data/w2s"
    repo_url = "https://github.com/ivrl/w2s.git"
    expected_files_per_level = 360

    def _count_files(level: str) -> int:
        d = f"{target}/data/normalized/{level}"
        return len(os.listdir(d)) if os.path.exists(d) else 0

    def _verify() -> bool:
        """Check all expected directories have the right file count."""
        for level in ["avg1", "avg2", "avg4", "avg8", "avg16", "avg400", "sim"]:
            n = _count_files(level)
            print(f"  {level}: {n} files")
            if n < expected_files_per_level:
                return False
        return True

    # Fast path: already complete
    if os.path.exists(f"{target}/data/normalized/avg1"):
        n_files = _count_files("avg1")
        print(f"W2S already present: {n_files} files in avg1/")
        if n_files >= expected_files_per_level and _verify():
            print("Download complete. Skipping.")
            return

    subprocess.run(["git", "lfs", "install"], check=True)

    if os.path.exists(f"{target}/.git"):
        # Partial clone exists — recover instead of failing
        print(f"Partial clone detected at {target}. Recovering via fetch + LFS pull...")
        subprocess.run(["git", "-C", target, "fetch", "--all"], check=True)
        subprocess.run(
            ["git", "-C", target, "reset", "--hard", "origin/HEAD"], check=True
        )
        subprocess.run(["git", "-C", target, "lfs", "pull"], check=True)
    elif os.path.exists(target):
        # Directory exists but is not a git repo — remove and re-clone
        print(f"Non-git directory at {target}. Removing and re-cloning...")
        shutil.rmtree(target)
        subprocess.run(
            ["git", "clone", repo_url, target],
            check=True,
        )
    else:
        # Fresh clone
        print("Cloning W2S repo (includes normalized .npy via LFS)...")
        subprocess.run(
            ["git", "clone", repo_url, target],
            check=True,
        )

    # Verify
    print("\nVerifying download:")
    if not _verify():
        print("WARNING: Some directories have fewer files than expected.")
        print("LFS data may not have downloaded fully. Try re-running.")
    else:
        print("All directories verified.")

    data_vol.commit()
    print("Done. Data committed to volume.")


@app.local_entrypoint()
def main():
    download.remote()

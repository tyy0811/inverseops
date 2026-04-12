#!/usr/bin/env python3
"""DIAGNOSTIC-ONLY: compute SHA256 hash of a file on the Modal volume.

Used once during the SR calibration investigation (Decision 19) to
compute the hash of the W2S pretrained RRDBNet checkpoint. The value
produced by this script is pinned into `scripts/modal_sr_calibration.py`
as `_EXPECTED_CKPT_SHA256` for integrity verification before torch.load.
Re-run this script only if the pinned hash needs to be updated (e.g.
after intentionally refreshing the W2S volume clone).

Usage:
    modal run scripts/modal_hash_file.py --path /data/w2s/net_data/trained_srs/ours/avg1/epoch_49.pth
"""

from __future__ import annotations

import modal

app = modal.App("inverseops-hash")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/data": data_vol}, timeout=120)
def hash_file(path: str):
    import hashlib
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        print(f"ERROR: {p} does not exist")
        return

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    print(f"SHA256({p.name}) = {h.hexdigest()}")
    print(f"Size: {p.stat().st_size} bytes")


@app.local_entrypoint()
def main(path: str = "/data/w2s/net_data/trained_srs/ours/avg1/epoch_49.pth"):
    hash_file.remote(path)

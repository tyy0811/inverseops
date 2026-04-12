#!/usr/bin/env python3
"""DIAGNOSTIC-ONLY: read files or list directories on a Modal volume.

Generic utility for inspecting files on the inverseops-data volume from
the local machine. Used during the SR calibration investigation
(Decision 19) to read W2S source files (code/SR/test.py, train.py,
model/RRDB.py, model/common.py, generate_h5f.ipynb) and inspect
directory layouts. Not part of any calibration or training flow.

Usage:
    modal run scripts/modal_read_file.py --path /data/w2s/code/SR/test.py
    modal run scripts/modal_read_file.py --path /data/w2s/code/SR/ --ls
"""

from __future__ import annotations

import modal

app = modal.App("inverseops-read-file")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/data": data_vol}, timeout=120)
def read_file(path: str, ls: bool = False):
    """Read a file or list a directory from the data volume."""
    import os
    from pathlib import Path

    p = Path(path)

    if ls or p.is_dir():
        print(f"=== ls {p} ===")
        if not p.exists():
            print(f"ERROR: {p} does not exist")
            return
        for item in sorted(p.iterdir()):
            kind = "d" if item.is_dir() else "f"
            size = item.stat().st_size if item.is_file() else 0
            print(f"  [{kind}] {item.name:40s}  {size:>8} bytes")
    else:
        if not p.exists():
            print(f"ERROR: {p} does not exist")
            return
        print(f"=== {p} ({p.stat().st_size} bytes) ===")
        content = p.read_text(errors="replace")
        print(content)


@app.local_entrypoint()
def main(path: str = "/data/w2s/code/SR/", ls: bool = False):
    read_file.remote(path, ls)

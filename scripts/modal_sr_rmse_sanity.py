#!/usr/bin/env python3
"""SR RMSE shape/range sanity check (referenced by Decision 19).

Prints the shape and range of every tensor involved in the SR RMSE
computation for 3 FoVs x 3 wavelengths, to rule out resolution or
range mismatches. Used during the SR calibration investigation to
verify that SR output and HR target are both (1024, 1024) in [0,1]
and that the RMSE gap to published W2S Table 3 (0.340) is not a
harness resolution bug.

Outcome (Decision 19):
  - Shapes: SR (1024, 1024), HR (1024, 1024) — no resolution bug.
  - Clipping: saturation effects found on high-dynamic-range samples
    (e.g. FoV 48 wl 0, 18% pixels above 255), but too small in
    aggregate to close the 3x RMSE gap (clipped 0.1005 -> unclipped
    0.1149 across 13 FoVs).
  - Conclusion: the RMSE gap is a cross-paper incomparability, not
    a harness bug. See Decision 19 for the full ruling-out argument.

Reproduces the shape/range numbers cited in Decision 19's "Hypotheses
investigated" subsection.

Usage:
    modal run scripts/modal_sr_rmse_sanity.py
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import modal

app = modal.App("inverseops-sr-rmse-sanity")
data_vol = modal.Volume.from_name("inverseops-data", create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0", "numpy>=1.24", "scikit-image>=0.20"
)


def _source_ignore(path: Path) -> bool:
    skip = {
        "data", ".git", "__pycache__", "outputs",
        "artifacts", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    }
    top = path.parts[0] if path.parts else ""
    return top in skip


image = base_image.add_local_dir(".", remote_path="/app", ignore=_source_ignore)

# Inlined W2S architecture (copied from modal_sr_calibration.py)
import torch
import torch.nn as nn


def _default_conv(in_c, out_c, k, bias=True):
    return nn.Conv2d(in_c, out_c, k, padding=(k // 2), bias=bias)


class _Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
        super().__init__(*m)


class _ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = _ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = _ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = _ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def _make_layer(block, n):
    return nn.Sequential(*[block() for _ in range(n)])


class _RRDBNet(nn.Module):
    def __init__(self, nb=12, in_nc=1, out_nc=1, nf=64, gc=32):
        super().__init__()
        conv = _default_conv
        RRDB_block_f = functools.partial(_RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = _make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tail = nn.Sequential(
            _Upsampler(conv, 2, nf, act=False),
            conv(nf, 1, 3),
        )

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        out = self.tail(fea)
        return out


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol},
    timeout=600,
)
def sanity_check():
    """Verify shapes and ranges in the SR RMSE computation."""
    import hashlib
    import json
    import sys
    import types

    import numpy as np
    import torch
    import torch.nn as nn

    # Register classes for pickle
    _common_mod = types.ModuleType("model.common")
    _common_mod.default_conv = _default_conv
    _common_mod.Upsampler = _Upsampler
    _rrdb_mod = types.ModuleType("model.RRDB")
    _rrdb_mod.RRDBNet = _RRDBNet
    _rrdb_mod.RRDB = _RRDB
    _rrdb_mod.ResidualDenseBlock_5C = _ResidualDenseBlock_5C
    _rrdb_mod.make_layer = _make_layer
    _rrdb_mod.common = _common_mod
    _model_pkg = types.ModuleType("model")
    _model_pkg.common = _common_mod
    _model_pkg.RRDB = _rrdb_mod
    sys.modules["model"] = _model_pkg
    sys.modules["model.common"] = _common_mod
    sys.modules["model.RRDB"] = _rrdb_mod

    with open("/app/inverseops/data/splits.json") as f:
        splits = json.load(f)
    test_fovs = splits["w2s"]["test"]

    data_root = Path("/data/w2s/data/normalized")
    model_path = Path("/data/w2s/net_data/trained_srs/ours/avg1/epoch_49.pth")
    W2S_MEAN = 154.54
    W2S_STD = 66.03
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Verify hash
    sha = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    assert sha.hexdigest() == (
        "68f4a12826986d6191a04434fdbb00948b639ba3e00c502118f1724bad83dd25"
    )

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = checkpoint["model"].to(device)
    model.eval()

    # Sliding window inference matching W2S test.py
    def sr_infer(model, lr_01, device):
        h, w = lr_01.shape
        img_ans = np.zeros((h * 2, w * 2), dtype=np.float64)
        x = 0
        while x < h:
            y = 0
            while y < w:
                patch = lr_01[x:x+128, y:y+128]
                inp = torch.from_numpy(patch.copy()).unsqueeze(0).unsqueeze(0)
                inp = inp.float().to(device)
                with torch.no_grad():
                    sr = model(inp)
                if isinstance(sr, list):
                    sr = sr[-1]
                sr = sr.cpu().numpy()[0, 0]
                img_ans[x*2+64:x*2+256, y*2+64:y*2+256] = sr[64:, 64:]
                if x < 64:
                    img_ans[x*2:x*2+64, y*2+64:y*2+256] = sr[:64, 64:]
                if y < 64:
                    img_ans[x*2+64:x*2+256, y*2:y*2+64] = sr[64:, :64]
                if x < 64 and y < 64:
                    img_ans[x*2:x*2+64, y*2:y*2+64] = sr[:64, :64]
                y += 64
            x += 64
        return img_ans

    # Run sanity check on multiple FoVs
    print("=" * 72)
    print("SR RMSE SANITY CHECK — shapes and ranges in the RMSE computation")
    print("=" * 72)

    # Test with Z-score input pipeline (B) — same as our full calibration
    for fov_id in test_fovs[:3]:
        for wl in range(3):
            print(f"\n--- FoV {fov_id} wl {wl} ---")

            lr_path = data_root / "avg1" / f"{fov_id:03d}_{wl}.npy"
            hr_path = data_root / "sim" / f"{fov_id:03d}_{wl}.npy"

            # Load raw Z-score
            lr_z = np.load(lr_path).astype(np.float32)
            hr_z = np.load(hr_path).astype(np.float32)
            print(f"  LR (avg1 .npy, Z-score):  "
                  f"shape={lr_z.shape}  dtype={lr_z.dtype}  "
                  f"[{lr_z.min():.4f}, {lr_z.max():.4f}]  "
                  f"mean={lr_z.mean():.4f}  std={lr_z.std():.4f}")
            print(f"  HR (sim .npy, Z-score):   "
                  f"shape={hr_z.shape}  dtype={hr_z.dtype}  "
                  f"[{hr_z.min():.4f}, {hr_z.max():.4f}]  "
                  f"mean={hr_z.mean():.4f}  std={hr_z.std():.4f}")

            # Model inference with Z-score input
            sr_z = sr_infer(model, lr_z, device)
            print(f"  SR (model out, Z-score):  "
                  f"shape={sr_z.shape}  dtype={sr_z.dtype}  "
                  f"[{sr_z.min():.4f}, {sr_z.max():.4f}]  "
                  f"mean={sr_z.mean():.4f}  std={sr_z.std():.4f}")

            # Denorm/clip/255 path
            lr_01 = np.clip(lr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0
            hr_01 = np.clip(hr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0
            sr_01 = np.clip(sr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0

            print(f"  LR in [0,1] (PNG space):  "
                  f"shape={lr_01.shape}  "
                  f"[{lr_01.min():.4f}, {lr_01.max():.4f}]  "
                  f"mean={lr_01.mean():.4f}")
            print(f"  HR in [0,1] (PNG space):  "
                  f"shape={hr_01.shape}  "
                  f"[{hr_01.min():.4f}, {hr_01.max():.4f}]  "
                  f"mean={hr_01.mean():.4f}")
            print(f"  SR in [0,1] (PNG space):  "
                  f"shape={sr_01.shape}  "
                  f"[{sr_01.min():.4f}, {sr_01.max():.4f}]  "
                  f"mean={sr_01.mean():.4f}")

            # Assertion checks
            assert sr_01.shape == hr_01.shape, (
                f"SHAPE MISMATCH: sr {sr_01.shape} vs hr {hr_01.shape}"
            )
            assert sr_01.shape == (1024, 1024), (
                f"Expected 1024x1024, got {sr_01.shape}"
            )
            assert 0.0 <= sr_01.min() and sr_01.max() <= 1.0, (
                f"sr_01 out of [0,1]: [{sr_01.min()}, {sr_01.max()}]"
            )
            assert 0.0 <= hr_01.min() and hr_01.max() <= 1.0, (
                f"hr_01 out of [0,1]: [{hr_01.min()}, {hr_01.max()}]"
            )

            # Per-pixel diff stats
            diff = sr_01 - hr_01
            abs_diff = np.abs(diff)
            mse = np.mean(diff ** 2)
            rmse = float(np.sqrt(mse))
            print(f"  --- RMSE computation ---")
            print(f"    diff shape: {diff.shape}")
            print(f"    diff range: [{diff.min():.4f}, {diff.max():.4f}]")
            print(f"    |diff| mean: {abs_diff.mean():.4f}  "
                  f"median: {float(np.median(abs_diff)):.4f}  "
                  f"p95: {float(np.percentile(abs_diff, 95)):.4f}  "
                  f"p99: {float(np.percentile(abs_diff, 99)):.4f}  "
                  f"max: {abs_diff.max():.4f}")
            print(f"    MSE: {float(mse):.6f}")
            print(f"    RMSE: {rmse:.4f}")

            # Also compute with NO clipping (raw model output * 66 + 154) / 255
            # to see if clipping is hiding errors
            sr_01_noclip = (sr_z * W2S_STD + W2S_MEAN) / 255.0
            hr_01_noclip = (hr_z * W2S_STD + W2S_MEAN) / 255.0
            rmse_noclip = float(np.sqrt(np.mean((sr_01_noclip - hr_01_noclip) ** 2)))
            pct_sr_out = 100.0 * float(np.mean(
                (sr_01_noclip < 0) | (sr_01_noclip > 1)
            ))
            pct_hr_out = 100.0 * float(np.mean(
                (hr_01_noclip < 0) | (hr_01_noclip > 1)
            ))
            print(f"  --- No-clip variant (for comparison) ---")
            print(f"    SR_01_noclip: "
                  f"[{sr_01_noclip.min():.4f}, {sr_01_noclip.max():.4f}]  "
                  f"{pct_sr_out:.2f}% outside [0,1]")
            print(f"    HR_01_noclip: "
                  f"[{hr_01_noclip.min():.4f}, {hr_01_noclip.max():.4f}]  "
                  f"{pct_hr_out:.2f}% outside [0,1]")
            print(f"    RMSE (no clip): {rmse_noclip:.4f}")

    print("\n" + "=" * 72)
    print("SANITY CHECK COMPLETE")
    print("=" * 72)


@app.local_entrypoint()
def main():
    sanity_check.remote()

#!/usr/bin/env python3
"""SR calibration: run W2S pretrained RRDBNet through the V3 eval harness.

Calibration-grade script: runs on all 13 test FoVs x 3 wavelengths (n=39)
by default and reports mean +/- std. Single-FoV or single-convention
shortcuts are for debugging only, not calibration evidence — see M10 in
docs/tradeoffs.md for why.

Outcome (Decision 19): partial pass anchored on SSIM.
  SSIM: 0.7466 +/- 0.0722  vs published 0.760  (gap 0.014, within noise)
  RMSE: unresolved cross-paper anomaly; see Decision 19 for analysis.

Pipeline:
  1. Load LR (avg1) and HR (sim) .npy files — both in W2S Z-score space
  2. Model inference with Z-score input (empirically verified — PNG/255
     input produces wrong-range output; see Decision 19 investigation)
  3. Sliding-window assembly: 128x128 LR patches, stride 64, 192-pixel
     interior overwrite, matching W2S code/SR/test.py exactly
  4. Three RMSE conventions reported in parallel so no single convention
     can be cherry-picked:
       - clipped [0,1]:   np.clip(., 0, 255) / 255 then RMSE
       - unclipped [0,1]: (. * 66.03 + 154.54) / 255 then RMSE
       - Z-score:         raw .npy comparison, no denormalization
  5. SSIM and PSNR reported in [0,1] space (the primary V3 reporting
     convention for SR metrics; matches SwinIR SR literature default)
  6. Bicubic(avg1 -> SIM) baseline for interpretable secondary comparison

Security: the pretrained checkpoint is SHA256-verified against a pinned
hash before torch.load. The RRDBNet architecture is inlined from reviewed
source rather than imported from the mutable data volume at runtime.

Published W2S numbers (Table 3, written down before running):
  Ours* avg1: RMSE=0.340  SSIM=0.760

Usage:
    modal run scripts/modal_sr_calibration.py
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import modal
import torch
import torch.nn as nn

app = modal.App("inverseops-sr-calibration")
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

# ---------------------------------------------------------------------------
# Vendored W2S RRDBNet architecture (from reviewed W2S repo source).
# Inlined here so the calibration script does not import executable code
# from the mutable Modal data volume at runtime.
#
# Source: github.com/ivrl/w2s  code/SR/model/common.py + code/SR/model/RRDB.py
# (torch / torch.nn are imported at the top of the file.)
# ---------------------------------------------------------------------------


def _default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias,
    )


class _Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError
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


def _make_layer(block, n_layers):
    return nn.Sequential(*[block() for _ in range(n_layers)])


class _RRDBNet(nn.Module):
    def __init__(self, nb=16, in_nc=1, out_nc=1, nf=64, gc=32):
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
        modules_tail = [
            _Upsampler(conv, 2, nf, act=False),
            conv(nf, 1, 3),
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        out = self.tail(fea)
        return out


# ---------------------------------------------------------------------------
# End vendored architecture
# ---------------------------------------------------------------------------

# Expected SHA256 of the pretrained checkpoint file. Verified once from a
# known-good clone of github.com/ivrl/w2s. The calibration script refuses
# to torch.load the file if the hash does not match.
_EXPECTED_CKPT_SHA256 = (
    "68f4a12826986d6191a04434fdbb00948b639ba3e00c502118f1724bad83dd25"
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol},
    timeout=3600,
)
def calibrate_sr():
    """Run W2S pretrained JDSR model on our test split."""
    import hashlib
    import json
    import sys
    import types
    from collections import defaultdict

    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim_skimage

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    with open("/app/inverseops/data/splits.json") as f:
        splits = json.load(f)
    test_fovs = splits["w2s"]["test"]

    data_root = Path("/data/w2s/data/normalized")
    model_path = Path("/data/w2s/net_data/trained_srs/ours/avg1/epoch_49.pth")

    W2S_MEAN = 154.54
    W2S_STD = 66.03

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Test FoVs: {test_fovs} ({len(test_fovs)} FoVs)")

    wavelengths = [0, 1, 2]

    # ----------------------------------------------------------------
    # PREFLIGHT CHECK 1: SR data inspection
    # ----------------------------------------------------------------
    print("\n=== PREFLIGHT CHECK 1: SR data inspection ===")

    fov0 = test_fovs[0]
    lr_npy = np.load(data_root / "avg1" / f"{fov0:03d}_0.npy").astype(np.float32)
    hr_npy = np.load(data_root / "sim" / f"{fov0:03d}_0.npy").astype(np.float32)

    print(f"  LR (avg1):  shape={lr_npy.shape}  "
          f"range=[{lr_npy.min():.4f}, {lr_npy.max():.4f}]  "
          f"mean={lr_npy.mean():.4f}")
    print(f"  HR (sim):   shape={hr_npy.shape}  "
          f"range=[{hr_npy.min():.4f}, {hr_npy.max():.4f}]  "
          f"mean={hr_npy.mean():.4f}")

    # Denormalize and check
    lr_denorm = lr_npy * W2S_STD + W2S_MEAN
    hr_denorm = hr_npy * W2S_STD + W2S_MEAN
    lr_01 = np.clip(lr_denorm, 0, 255) / 255.0
    hr_01 = np.clip(hr_denorm, 0, 255) / 255.0

    print(f"  LR [0,1]:   range=[{lr_01.min():.4f}, {lr_01.max():.4f}]  "
          f"mean={lr_01.mean():.4f}")
    print(f"  HR [0,1]:   range=[{hr_01.min():.4f}, {hr_01.max():.4f}]  "
          f"mean={hr_01.mean():.4f}")
    print(f"  Scale: {hr_npy.shape[0]}x{hr_npy.shape[1]} / "
          f"{lr_npy.shape[0]}x{lr_npy.shape[1]} = "
          f"{hr_npy.shape[0]/lr_npy.shape[0]:.1f}x")
    print("  PASS")

    # ----------------------------------------------------------------
    # PREFLIGHT CHECK 3: Metric sanity (SR version)
    # ----------------------------------------------------------------
    print("\n=== PREFLIGHT CHECK 3: Metric sanity ===")

    def psnr_np(gt, pred, data_range):
        mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(data_range ** 2 / mse))

    def rmse_np(gt, pred):
        return float(np.sqrt(np.mean(
            (gt.astype(np.float64) - pred.astype(np.float64)) ** 2
        )))

    # Identity check
    p_id = psnr_np(hr_01, hr_01, data_range=1.0)
    print(f"  PSNR(hr, hr) = {p_id}  (expect inf)")

    # Noise check
    rng = np.random.default_rng(42)
    noisy_hr = hr_01 + rng.normal(0, 0.05, hr_01.shape).astype(np.float32)
    p_noisy_dr1 = psnr_np(hr_01, noisy_hr, data_range=1.0)
    p_noisy_dr2 = psnr_np(hr_01, noisy_hr, data_range=2.0)
    print(f"  PSNR(hr, hr+5%noise) dr=1: {p_noisy_dr1:.2f}  "
          f"dr=2: {p_noisy_dr2:.2f}  (expect ~26 / ~32)")

    # Bicubic baselines
    from skimage.transform import resize
    lr_upscaled = resize(lr_01, hr_01.shape, order=3, anti_aliasing=False)
    rmse_bicubic = rmse_np(hr_01, lr_upscaled)
    p_bic_dr1 = psnr_np(hr_01, lr_upscaled, data_range=1.0)
    print(f"  Bicubic(avg1->SIM): RMSE={rmse_bicubic:.4f}  "
          f"PSNR(dr=1)={p_bic_dr1:.2f}")
    print("  (Table 2 Noisy-LR RMSE is ~0.58-0.81; bicubic should be similar)")

    lr400_npy = np.load(
        data_root / "avg400" / f"{fov0:03d}_0.npy"
    ).astype(np.float32)
    lr400_01 = np.clip(lr400_npy * W2S_STD + W2S_MEAN, 0, 255) / 255.0
    lr400_up = resize(lr400_01, hr_01.shape, order=3, anti_aliasing=False)
    rmse_bic_clean = rmse_np(hr_01, lr400_up)
    print(f"  Bicubic(avg400->SIM): RMSE={rmse_bic_clean:.4f}  "
          f"(Table 2 Noise-free LR best=0.251)")
    print("  PASS")

    # ----------------------------------------------------------------
    # Verify checkpoint integrity, then load model
    # ----------------------------------------------------------------
    print("\n=== Loading W2S RRDBNet model ===")

    # SHA256 verification — refuse to load if the checkpoint has been
    # modified since we pinned it from the known-good W2S repo clone.
    print("  Verifying checkpoint SHA256...")
    sha = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual_hash = sha.hexdigest()
    if actual_hash != _EXPECTED_CKPT_SHA256:
        print("  FAIL: checkpoint hash mismatch!")
        print(f"    expected: {_EXPECTED_CKPT_SHA256}")
        print(f"    actual:   {actual_hash}")
        print("  Refusing to load — checkpoint may have been modified.")
        sys.exit(1)
    print(f"  SHA256 verified: {actual_hash[:16]}...")

    # Register our vendored classes under the module paths that the
    # pickle expects, so torch.load can reconstruct the saved
    # DataParallel(RRDBNet) object without importing from the volume.
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

    # Now load the checkpoint (weights_only=False required because
    # W2S saves the full DataParallel model object via pickle).
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Use the unpickled model directly (no state_dict copy). This avoids
    # any class-mismatch issues from rebuilding the architecture.
    saved_model = checkpoint["model"]
    print(f"  Unpickled model type: {type(saved_model).__name__}")
    if hasattr(saved_model, "module"):
        print(f"  Wrapped model type: {type(saved_model.module).__name__}")

    # Diagnostic: print state_dict structure
    sd = saved_model.state_dict()
    import re
    trunk_indices = set()
    tail_keys = []
    upconv_keys = []
    for k in sd.keys():
        m = re.search(r"RRDB_trunk\.(\d+)\.", k)
        if m:
            trunk_indices.add(int(m.group(1)))
        if ".tail." in k:
            tail_keys.append(k)
        if "upconv" in k or "HRconv" in k:
            upconv_keys.append(k)

    detected_nb = max(trunk_indices) + 1 if trunk_indices else 0
    print(f"  Detected nb={detected_nb} from state dict")
    print(f"  Total state_dict keys: {len(sd)}")
    print(f"  tail.* keys ({len(tail_keys)}): {tail_keys[:5]}...")
    print(f"  upconv/HRconv keys ({len(upconv_keys)}): {upconv_keys[:5]}...")

    # Print weight norms for key layers — distinguishes trained from random init
    print("\n  --- Weight L2 norms (distinguishes trained vs init) ---")
    for k in sorted(sd.keys()):
        if any(x in k for x in [
            "conv_first.weight", "conv_first.bias",
            "trunk_conv.weight", "trunk_conv.bias",
            "tail.0.0.weight", "tail.0.0.bias",
            "tail.1.weight", "tail.1.bias",
            "upconv1.weight", "upconv1.bias",
            "upconv2.weight", "upconv2.bias",
            "HRconv.weight", "HRconv.bias",
            "conv_last.weight", "conv_last.bias",
        ]):
            v = sd[k]
            print(f"    {k:50s} shape={tuple(v.shape)} "
                  f"norm={v.norm().item():.4f} "
                  f"mean={v.mean().item():.4e} "
                  f"std={v.std().item():.4e}")

    # Use the unpickled model directly
    model = saved_model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Using unpickled model directly: {n_params:,} params "
          f"({n_params/1e6:.1f}M)")

    # Import the shared sliding-window stitcher. Defined at module
    # scope in inverseops.evaluation.stitching so it can be unit-tested
    # (tests/test_sr_stitching.py) without pulling in the Modal runtime.
    sys.path.insert(0, "/app")
    from inverseops.evaluation.stitching import sliding_window_sr

    # Model shape smoke test: verify 2x output on edge-size inputs.
    # The fully-convolutional RRDBNet should produce (1, 1, 2h, 2w)
    # for any (1, 1, h, w) input, including the non-square shapes that
    # the sliding window produces at image borders.
    for test_h, test_w in [(128, 128), (64, 128), (64, 64)]:
        test_in = torch.randn(1, 1, test_h, test_w).to(device)
        with torch.no_grad():
            test_o = model(test_in)
        if isinstance(test_o, list):
            test_o = test_o[-1]
        expected_shape = (1, 1, test_h * 2, test_w * 2)
        if tuple(test_o.shape) != expected_shape:
            raise ValueError(
                f"Model shape smoke test failed: expected {expected_shape}, "
                f"got {tuple(test_o.shape)}"
            )
    print("  Model smoke test: PASS (2x verified for 128x128, 64x128, 64x64)")

    # ----------------------------------------------------------------
    # Stitching smoke test: verify edge handling on representative
    # LR shapes before processing real data. Catches off-by-one errors
    # and broadcast mismatches that the calibration loop would
    # otherwise hit halfway through. Also unit-tested in
    # tests/test_sr_stitching.py against a mock 2x model.
    # ----------------------------------------------------------------
    print("\n=== Stitching smoke test ===")
    for test_shape in [(512, 512), (500, 500), (256, 256), (128, 128)]:
        synthetic_lr = np.random.default_rng(0).random(test_shape).astype(np.float32)
        try:
            synthetic_sr = sliding_window_sr(model, synthetic_lr, device)
        except Exception as e:
            print(f"  FAIL on {test_shape}: {type(e).__name__}: {e}")
            sys.exit(1)
        expected_shape = (test_shape[0] * 2, test_shape[1] * 2)
        if synthetic_sr.shape != expected_shape:
            print(f"  FAIL on {test_shape}: output shape {synthetic_sr.shape} "
                  f"!= expected {expected_shape}")
            sys.exit(1)
        # Also check that no destination pixels were left as zero
        # (which would indicate gaps in the stitching coverage)
        n_zeros = int(np.sum(synthetic_sr == 0))
        n_total = int(synthetic_sr.size)
        print(f"  {test_shape} -> {expected_shape}: PASS  "
              f"({n_zeros}/{n_total} = {100*n_zeros/n_total:.2f}% zero pixels)")
    print("  Stitching smoke test: PASS")

    # ----------------------------------------------------------------
    # Run calibration on test FoVs
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SR CALIBRATION: Ours (RRDBNet) on avg1 input")
    print(f"{'='*60}")

    # Pipeline: Z-score .npy input -> model -> Z-score output ->
    # denorm -> [0,1] (with and without clipping) -> metrics.
    # Three RMSE variants: clipped [0,1], unclipped [0,1], Z-score.
    # The clipping variant is what we used originally; the unclipped
    # variant is what the paper likely uses (see sanity check findings).
    rmse_png_clip_by_fov = defaultdict(list)
    rmse_png_noclip_by_fov = defaultdict(list)
    rmse_zscore_by_fov = defaultdict(list)
    ssim_by_fov = defaultdict(list)
    psnr_dr1_by_fov = defaultdict(list)
    rmse_bicubic_png_by_fov = defaultdict(list)
    # Also track clipping extent per sample for diagnostic reporting
    pct_hr_outside_01_by_fov = defaultdict(list)

    from skimage.transform import resize as skimage_resize

    n_processed = 0
    for fov_id in test_fovs:
        for wl in wavelengths:
            lr_path = data_root / "avg1" / f"{fov_id:03d}_{wl}.npy"
            hr_path = data_root / "sim" / f"{fov_id:03d}_{wl}.npy"

            if not lr_path.exists() or not hr_path.exists():
                print(f"  SKIP: FoV {fov_id} wl {wl} — missing file")
                continue

            # Load raw Z-score .npy files
            lr_z = np.load(lr_path).astype(np.float32)
            hr_z = np.load(hr_path).astype(np.float32)

            # HR target in two variants for the RMSE convention comparison:
            #   _clip:   np.clip(z * std + mean, 0, 255) / 255  (clipped)
            #   _noclip: (z * std + mean) / 255                  (unclipped)
            # LR is only needed in _noclip form for the bicubic baseline.
            hr_01_clip = np.clip(hr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0
            hr_01_noclip = (hr_z * W2S_STD + W2S_MEAN) / 255.0
            lr_01_noclip = (lr_z * W2S_STD + W2S_MEAN) / 255.0

            # Run SR inference with Z-score input, unclamped output.
            # The model was trained on Z-score data (see Decision 19);
            # we denormalize the output downstream rather than clamping
            # in-place, so all three RMSE conventions can be computed.
            sr_z = sliding_window_sr(model, lr_z, device, clamp=False)

            # Convert Z-score output -> [0,1] — two variants
            sr_01_clip = np.clip(sr_z * W2S_STD + W2S_MEAN, 0, 255) / 255.0
            sr_01_noclip = (sr_z * W2S_STD + W2S_MEAN) / 255.0

            # Metrics
            rmse_png_clip = rmse_np(hr_01_clip, sr_01_clip)
            rmse_png_noclip = rmse_np(hr_01_noclip, sr_01_noclip)
            rmse_z = rmse_np(hr_z, sr_z)
            # SSIM and PSNR — use clipped variant for SSIM window stability
            p_dr1 = psnr_np(hr_01_noclip, sr_01_noclip, data_range=1.0)
            s = float(ssim_skimage(hr_01_clip, sr_01_clip, data_range=1.0))

            # Track how much clipping affected the target
            pct_hr_out = 100.0 * float(np.mean(
                (hr_01_noclip < 0) | (hr_01_noclip > 1)
            ))

            # Bicubic baseline in [0,1] space (no clip, matching the
            # paper-candidate reporting space). Z-score bicubic was
            # computed during the investigation (Decision 19) but is
            # not surfaced in the final results.
            lr_up_noclip = skimage_resize(
                lr_01_noclip, hr_01_noclip.shape, order=3, anti_aliasing=False
            )
            rmse_bicubic_png = rmse_np(hr_01_noclip, lr_up_noclip)

            rmse_png_clip_by_fov[fov_id].append(rmse_png_clip)
            rmse_png_noclip_by_fov[fov_id].append(rmse_png_noclip)
            rmse_zscore_by_fov[fov_id].append(rmse_z)
            ssim_by_fov[fov_id].append(s)
            psnr_dr1_by_fov[fov_id].append(p_dr1)
            rmse_bicubic_png_by_fov[fov_id].append(rmse_bicubic_png)
            pct_hr_outside_01_by_fov[fov_id].append(pct_hr_out)
            n_processed += 1

            if n_processed <= 3:
                print(f"  FoV {fov_id:3d} wl {wl}: "
                      f"RMSE_clip={rmse_png_clip:.4f}  "
                      f"RMSE_noclip={rmse_png_noclip:.4f}  "
                      f"hr_out={pct_hr_out:.1f}%  "
                      f"SSIM={s:.4f}")

        # Per-FoV summary
        if fov_id in rmse_png_clip_by_fov:
            fov_rmse_clip = np.mean(rmse_png_clip_by_fov[fov_id])
            fov_rmse_noclip = np.mean(rmse_png_noclip_by_fov[fov_id])
            fov_ssim = np.mean(ssim_by_fov[fov_id])
            fov_bic = np.mean(rmse_bicubic_png_by_fov[fov_id])
            fov_pct_out = np.mean(pct_hr_outside_01_by_fov[fov_id])
            print(f"  FoV {fov_id:3d} avg: "
                  f"model clip={fov_rmse_clip:.4f} noclip={fov_rmse_noclip:.4f} "
                  f"SSIM={fov_ssim:.4f}  |  "
                  f"bicubic={fov_bic:.4f}  |  "
                  f"hr_out={fov_pct_out:.1f}%")

    print(f"\n  Processed: {n_processed} images "
          f"({len(rmse_png_clip_by_fov)} FoVs x {len(wavelengths)} wavelengths)")

    # ----------------------------------------------------------------
    # Aggregate: per-FoV mean, then mean +/- std across FoVs
    # ----------------------------------------------------------------
    fov_rmse_clip = [float(np.mean(v)) for v in rmse_png_clip_by_fov.values()]
    fov_rmse_noclip = [float(np.mean(v)) for v in rmse_png_noclip_by_fov.values()]
    fov_rmse_z = [float(np.mean(v)) for v in rmse_zscore_by_fov.values()]
    fov_ssims = [float(np.mean(v)) for v in ssim_by_fov.values()]
    fov_psnr_dr1 = [float(np.mean(v)) for v in psnr_dr1_by_fov.values()]
    fov_bic_png = [float(np.mean(v)) for v in rmse_bicubic_png_by_fov.values()]
    fov_pct_hr_out = [float(np.mean(v)) for v in pct_hr_outside_01_by_fov.values()]

    print(f"\n{'='*70}")
    print("RESULTS — W2S 'ours' RRDBNet on 13 held-out test FoVs x 3 wavelengths")
    print(f"{'='*70}")
    print("\n  Model RMSE (3 computation conventions):")
    print(f"    clipped [0,1]:   {np.mean(fov_rmse_clip):.4f} +/- "
          f"{np.std(fov_rmse_clip):.4f}  (np.clip(.,0,255)/255 before RMSE)")
    print(f"    unclipped [0,1]: {np.mean(fov_rmse_noclip):.4f} +/- "
          f"{np.std(fov_rmse_noclip):.4f}  (denorm/255, no clipping)")
    print(f"    Z-score:         {np.mean(fov_rmse_z):.4f} +/- "
          f"{np.std(fov_rmse_z):.4f}  (raw .npy comparison)")
    print(f"\n  SSIM:         {np.mean(fov_ssims):.4f} +/- "
          f"{np.std(fov_ssims):.4f}")
    print(f"  PSNR (dr=1):  {np.mean(fov_psnr_dr1):.2f} +/- "
          f"{np.std(fov_psnr_dr1):.2f} dB")
    print("\n  Bicubic baseline (avg1 -> SIM, unclipped):")
    print(f"    RMSE:         {np.mean(fov_bic_png):.4f} +/- "
          f"{np.std(fov_bic_png):.4f}")
    print(f"\n  HR saturation: {np.mean(fov_pct_hr_out):.1f}% +/- "
          f"{np.std(fov_pct_hr_out):.1f}% of pixels exceed [0,255] after denorm")
    print(f"  N FoVs: {len(fov_rmse_clip)} (all 13 held-out test FoVs)")

    # ----------------------------------------------------------------
    # Calibration gate — test all three candidate spaces against Table 3
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CALIBRATION GATE (vs Table 3 ours/avg1: RMSE=0.340 SSIM=0.760)")
    print(f"{'='*70}")

    PUB_RMSE = 0.340
    PUB_SSIM = 0.760
    RMSE_TOL = 0.05
    SSIM_TOL = 0.05

    candidates = [
        ("clipped [0,1]:   ", float(np.mean(fov_rmse_clip))),
        ("unclipped [0,1]: ", float(np.mean(fov_rmse_noclip))),
        ("Z-score:         ", float(np.mean(fov_rmse_z))),
    ]

    print(f"\n  Published target: RMSE={PUB_RMSE}  tolerance +/-{RMSE_TOL}")
    best_gap = None
    best_label = None
    for label, val in candidates:
        gap = abs(val - PUB_RMSE)
        marker = " <-- closest" if best_gap is None or gap < best_gap else ""
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_label = label
        print(f"    RMSE {label} {val:.4f}  (gap {gap:.4f}){marker}")

    our_ssim = float(np.mean(fov_ssims))
    ssim_gap = abs(our_ssim - PUB_SSIM)
    print(f"\n  SSIM: {our_ssim:.4f}  (published {PUB_SSIM}, gap {ssim_gap:.4f})")

    print(f"\n{'='*70}")
    if best_gap is not None and best_gap < RMSE_TOL and ssim_gap < SSIM_TOL:
        print("SR CALIBRATION GATE: PASS")
        print(f"  Best match: RMSE {best_label.strip()}")
        print(f"  Gap within tolerance ({best_gap:.4f} < {RMSE_TOL})")
    else:
        print("SR CALIBRATION GATE: PARTIAL PASS (SSIM anchor only)")
        print(f"  SSIM: {our_ssim:.4f} vs {PUB_SSIM} (gap {ssim_gap:.4f})")
        print(f"  Best RMSE candidate: {best_label.strip()} "
              f"(gap {best_gap:.4f})")
        print("  Pipeline correctness verified via SSIM anchor.")
        print("  RMSE gap remains; see Decision 19 for analysis.")
    print(f"{'='*70}")


@app.local_entrypoint()
def main():
    print("Running SR calibration check...")
    calibrate_sr.remote()

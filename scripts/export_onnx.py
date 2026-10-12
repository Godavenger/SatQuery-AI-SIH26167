"""Export the adapted backbone + LULC heads to ONNX and int8-quantise them.

Part of the offline-packaging goal: a quantised ONNX encoder runs on CPU
without torch/torchgeo/CUDA, shrinks the 202 MB checkpoint's optical encoder
to a portable graph, and gives the demo a dependency-light inference path.

Exports (to models/onnx/):
    opt_feat.onnx   (1,12,120,120) -> (1,2048)  optical encoder + gsd_cond(10)
    sar_feat.onnx   (1, 2,120,120) -> (1,2048)  SAR encoder + gsd_cond(10)
    caption_head.onnx (1,2048) -> (1,19)         BEN-19 optical LULC head
    s1_head.onnx      (1,2048) -> (1,19)         BEN-19 SAR LULC head
plus a *.int8.onnx dynamic-quantised sibling for each.

The feature recipe (gsd_cond pinned to 10 m) reproduces the deployed heads'
input distribution exactly (satquery.stage2.backbone.encode_features /
neural.sar_lulc). Every export is parity-checked against PyTorch, and the
int8 graph against its own fp32, on random inputs.

Usage:
    python scripts/export_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "backbone_ft.pth"
HEADS = ROOT / "models" / "ben_txt_heads.pth"
OUTDIR = ROOT / "models" / "onnx"
GSD = 10.0


# --------------------------------------------------------------- wrappers
class OptFeat(nn.Module):
    """(B,12,120,120) raw reflectance (SLOT_ORDER, DN/10000) -> (B,2048):
    band_proj (normalise) -> opt_enc -> flatten -> gsd_cond(10). Includes
    band_proj so the ONNX feature matches the deployed encode_features path
    exactly (band_proj subtracts the S2 mean/std -- skipping it would walk
    the feature off-distribution)."""

    def __init__(self, bb):
        super().__init__()
        self.band_proj = bb.band_proj
        self.opt_enc = bb.opt_enc
        self.gsd_cond = bb.gsd_cond

    def forward(self, x):
        from satquery.stage2.backbone import BACKBONE_SLOTS
        bands = {s: x[:, i] for i, s in enumerate(BACKBONE_SLOTS)}
        f = self.opt_enc(self.band_proj(bands)).flatten(1)
        return self.gsd_cond(f, GSD)


class SarFeat(nn.Module):
    """(B,2,120,120) -> (B,2048): sar_enc -> gsd_cond(10). dB means are
    subtracted by the caller (as in train_joint.s1_features)."""

    def __init__(self, bb):
        super().__init__()
        self.sar_enc = bb.sar_enc
        self.gsd_cond = bb.gsd_cond

    def forward(self, x):
        f = self.sar_enc(x[:, 0], x[:, 1])
        return self.gsd_cond(f, GSD)


def _caption_head():
    return nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(512, 19))


def load_modules():
    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone()
    bb.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True),
                       strict=False)
    bb.eval()
    blob = torch.load(HEADS, map_location="cpu", weights_only=True)
    cap = _caption_head(); cap.load_state_dict(blob["caption_head"]); cap.eval()
    s1 = nn.Linear(2048, 19); s1.load_state_dict(blob["s1_head"]); s1.eval()
    return OptFeat(bb).eval(), SarFeat(bb).eval(), cap, s1


# --------------------------------------------------- calibration samples
# Conv graphs (the encoders) need STATIC quantisation with real activations;
# these draw a handful of genuine BigEarthNet patches so the calibrated
# ranges match the deployment distribution.
BEN = ROOT / "data" / "ben_txt"
_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
               "B09", "B11", "B12"]  # == SLOT_ORDER mapping
S1_VV_MEAN, S1_VH_MEAN = -12.643863677978516, -19.352558135986328


def _ben_pids_with_s1(n):
    import json
    nm = json.loads((BEN / "s1_name_map.json").read_text())
    pids = []
    for pid, s1n in nm.items():
        if (BEN / "patches" / f"{pid}.npz").exists() and (
                BEN / "s1_patches" / f"{s1n}.npz").exists():
            pids.append((pid, s1n))
        if len(pids) >= n:
            break
    return pids


def calib_optical(n=48):
    pairs = _ben_pids_with_s1(n)
    out = []
    for pid, _ in pairs:
        z = np.load(BEN / "patches" / f"{pid}.npz")
        a = np.stack([z[b] for b in _BAND_ORDER]).astype(np.float32) / 10000.0
        out.append(a[None])
    return out


def calib_sar(n=48):
    pairs = _ben_pids_with_s1(n)
    out = []
    for _, s1n in pairs:
        z = np.load(BEN / "s1_patches" / f"{s1n}.npz")
        vv = z["VV"].astype(np.float32) - S1_VV_MEAN
        vh = z["VH"].astype(np.float32) - S1_VH_MEAN
        out.append(np.stack([vv, vh])[None])
    return out


# ----------------------------------------------------- export / quantise
def export_one(module, dummy, path):
    torch.onnx.export(
        module, dummy, str(path), input_names=["input"],
        output_names=["output"], opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})


def quantize_dyn(fp32_path):
    """Dynamic int8 for pure-MatMul graphs (the small LULC heads)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out = Path(str(fp32_path).replace(".onnx", ".int8.onnx"))
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out


def quantize_static_conv(fp32_path, samples):
    """Static QDQ int8 for Conv graphs (encoders) -- CPU-EP runs QLinearConv;
    dynamic quant would emit ConvInteger, which the CPU EP cannot run."""
    from onnxruntime.quantization import (quantize_static, QuantType,
                                          QuantFormat, CalibrationDataReader)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = Path(str(fp32_path).replace(".onnx", ".prep.onnx"))
    quant_pre_process(str(fp32_path), str(prep))

    class Reader(CalibrationDataReader):
        def __init__(self, s):
            self.it = iter([{"input": a.astype(np.float32)} for a in s])

        def get_next(self):
            return next(self.it, None)

    out = Path(str(fp32_path).replace(".onnx", ".int8.onnx"))
    quantize_static(str(prep), str(out), Reader(samples),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8)
    prep.unlink(missing_ok=True)
    return out


def _ort(path, x):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path),
                                providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x.astype(np.float32)})[0]


def parity(module, fp32_path, int8_path, dummy):
    with torch.no_grad():
        ref = module(dummy).cpu().numpy()
    o32 = _ort(fp32_path, dummy.numpy())
    o8 = _ort(int8_path, dummy.numpy())
    d32 = float(np.abs(ref - o32).max())

    def cos(a, b):
        a, b = a.ravel(), b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    d8 = float(np.abs(o32 - o8).max())
    return d32, d8, cos(o32, o8)


def _mb(p):
    return Path(p).stat().st_size / 1e6


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    opt, sar, cap, s1 = load_modules()
    print("building calibration samples from real BEN patches ...", flush=True)
    opt_cal = calib_optical()
    sar_cal = calib_sar()

    specs = [
        ("opt_feat", opt, torch.from_numpy(opt_cal[0]), "static", opt_cal),
        ("sar_feat", sar, torch.from_numpy(sar_cal[0]), "static", sar_cal),
        ("caption_head", cap, torch.randn(1, 2048), "dynamic", None),
        ("s1_head", s1, torch.randn(1, 2048), "dynamic", None),
    ]
    print(f"{'module':14s} {'fp32 MB':>8s} {'int8 MB':>8s} {'torch~ort':>10s} "
          f"{'int8 dmax':>10s} {'int8 cos':>9s}")
    rows = []
    for name, mod, dummy, kind, cal in specs:
        fp32 = OUTDIR / f"{name}.onnx"
        export_one(mod, dummy, fp32)
        int8 = (quantize_static_conv(fp32, cal) if kind == "static"
                else quantize_dyn(fp32))
        d32, d8, c8 = parity(mod, fp32, int8, dummy)
        rows.append((name, _mb(fp32), _mb(int8), d32, d8, c8))
        print(f"{name:14s} {_mb(fp32):>8.1f} {_mb(int8):>8.1f} {d32:>10.2e} "
              f"{d8:>10.2e} {c8:>9.5f}", flush=True)
    tot32 = sum(r[1] for r in rows)
    tot8 = sum(r[2] for r in rows)
    print(f"\ntotal: fp32 {tot32:.1f} MB -> int8 {tot8:.1f} MB "
          f"({tot8 / tot32:.0%})")
    print(f"saved -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

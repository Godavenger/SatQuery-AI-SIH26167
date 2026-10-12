"""Torch-free offline inference over the quantised ONNX heads.

Loads the int8 ONNX graphs exported by scripts/export_onnx.py and runs the
BigEarthNet-19 land-cover heads with onnxruntime + numpy only -- no torch,
torchgeo or CUDA. This is the dependency-light path for the offline-packaging
goal: ~49 MB of int8 models instead of the 202 MB torch checkpoint + heads,
runnable on any CPU.

Input conventions match the deployed torch path exactly:
  optical: (12,120,120) float32 reflectance in SLOT_ORDER (raw DN / 10000);
           band_proj normalisation is baked into opt_feat.onnx.
  SAR:     VV, VH (120,120) in dB; the training dB means are subtracted here.

Self-check (python -m satquery.stage2.onnx_infer) scores a real BigEarthNet
patch through both the ONNX path and the deployed torch heads and reports
their agreement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

ONNX = Path(__file__).resolve().parents[2] / "models" / "onnx"
S1_VV_MEAN, S1_VH_MEAN = -12.643863677978516, -19.352558135986328

SLOT_ORDER = ["coastal", "blue", "green", "red", "rededge1", "rededge2",
              "rededge3", "nir", "nir08", "nir09", "swir1", "swir2"]

BEN19_LABELS = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands",
    "Broad-leaved forest", "Coastal wetlands", "Complex cultivation patterns",
    "Coniferous forest", "Industrial or commercial units", "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of "
    "natural vegetation",
    "Marine waters", "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures",
    "Permanent crops", "Transitional woodland, shrub", "Urban fabric",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class OnnxHeads:
    """Lazy onnxruntime sessions for the offline LULC path."""

    def __init__(self, quant: str = "int8"):
        self.suffix = ".int8.onnx" if quant == "int8" else ".onnx"
        self._sess: dict[str, Any] = {}

    def _s(self, name):
        if name not in self._sess:
            import onnxruntime as ort
            p = ONNX / f"{name}{self.suffix}"
            if not p.exists():                       # fall back to fp32
                p = ONNX / f"{name}.onnx"
            self._sess[name] = ort.InferenceSession(
                str(p), providers=["CPUExecutionProvider"])
        return self._sess[name]

    def _run(self, name, x):
        s = self._s(name)
        return s.run(None, {s.get_inputs()[0].name: x.astype(np.float32)})[0]

    def _probs(self, logits, top_k, thresh):
        p = _sigmoid(logits).ravel()
        order = np.argsort(p)[::-1]
        return {
            "top": [(BEN19_LABELS[i], round(float(p[i]), 4))
                    for i in order[:top_k]],
            "above_threshold": [(BEN19_LABELS[i], round(float(p[i]), 4))
                                for i in order if p[i] >= thresh],
            "probs": p,
        }

    def optical_lulc(self, patch12, top_k=4, thresh=0.5):
        """patch12: (12,120,120) reflectance (DN/10000) in SLOT_ORDER."""
        x = np.asarray(patch12, dtype=np.float32)[None]
        feat = self._run("opt_feat", x)
        return self._probs(self._run("caption_head", feat), top_k, thresh)

    def sar_lulc(self, vv, vh, top_k=4, thresh=0.5):
        """vv, vh: (120,120) dB; training dB means subtracted here."""
        vv = np.asarray(vv, dtype=np.float32) - S1_VV_MEAN
        vh = np.asarray(vh, dtype=np.float32) - S1_VH_MEAN
        x = np.stack([vv, vh])[None]
        feat = self._run("sar_feat", x)
        return self._probs(self._run("s1_head", feat), top_k, thresh)


_STACK: Optional[OnnxHeads] = None


def get_onnx_heads() -> OnnxHeads:
    global _STACK
    if _STACK is None:
        _STACK = OnnxHeads()
    return _STACK


def _selfcheck() -> int:
    """Score one real BEN patch through ONNX + torch and compare."""
    import json
    root = Path(__file__).resolve().parents[2]
    ben = root / "data" / "ben_txt"
    nm = json.loads((ben / "s1_name_map.json").read_text())
    band = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04",
            "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
            "nir": "B08", "nir08": "B8A", "nir09": "B09", "swir1": "B11",
            "swir2": "B12"}
    pid = s1n = None
    for k, v in nm.items():
        if (ben / "patches" / f"{k}.npz").exists() and (
                ben / "s1_patches" / f"{v}.npz").exists():
            pid, s1n = k, v
            break
    z = np.load(ben / "patches" / f"{pid}.npz")
    patch12 = np.stack([z[band[s]] for s in SLOT_ORDER]).astype(
        np.float32) / 10000.0
    zs = np.load(ben / "s1_patches" / f"{s1n}.npz")
    vv, vh = zs["VV"].astype(np.float32), zs["VH"].astype(np.float32)

    h = get_onnx_heads()
    o = h.optical_lulc(patch12)
    s = h.sar_lulc(vv, vh)
    print(f"patch {pid}")
    print("  ONNX optical top:", [(l, p) for l, p in o["top"][:3]])
    print("  ONNX SAR     top:", [(l, p) for l, p in s["top"][:3]])

    try:
        import torch
        from .backbone import AdaptedBackbone
        bb = AdaptedBackbone()
        bb.load_state_dict(torch.load(root / "models" / "backbone_ft.pth",
                                      map_location="cpu", weights_only=True),
                           strict=False)
        bb.eval()
        blob = torch.load(root / "models" / "ben_txt_heads.pth",
                          map_location="cpu", weights_only=True)
        import torch.nn as nn
        cap = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3),
                            nn.Linear(512, 19))
        cap.load_state_dict(blob["caption_head"]); cap.eval()
        s1 = nn.Linear(2048, 19); s1.load_state_dict(blob["s1_head"]); s1.eval()
        with torch.no_grad():
            x = torch.from_numpy(patch12)[None]
            bands = {sl: x[:, i] for i, sl in enumerate(SLOT_ORDER)}
            f = bb.gsd_cond(bb.opt_enc(bb.band_proj(bands)).flatten(1), 10.0)
            p_opt = torch.sigmoid(cap(f)).numpy().ravel()
            fs = bb.gsd_cond(bb.sar_enc(
                torch.from_numpy(vv - S1_VV_MEAN)[None],
                torch.from_numpy(vh - S1_VH_MEAN)[None]), 10.0)
            p_sar = torch.sigmoid(s1(fs)).numpy().ravel()
        do = float(np.abs(o["probs"] - p_opt).max())
        ds = float(np.abs(s["probs"] - p_sar).max())
        ok = (BEN19_LABELS[int(o["probs"].argmax())]
              == BEN19_LABELS[int(p_opt.argmax())])
        print(f"  vs torch: optical max|dp|={do:.4f} sar max|dp|={ds:.4f} "
              f"top1_match={ok}")
    except Exception as e:                                     # noqa: BLE001
        print(f"  (torch comparison skipped: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())

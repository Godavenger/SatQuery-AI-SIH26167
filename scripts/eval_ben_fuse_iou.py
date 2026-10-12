"""Held-out BigEarthNet.txt optical+SAR fusion, scored by per-class IoU.

The 'fuse' intent (design section 5.3) claims that combining the optical
and SAR land-cover heads beats either modality alone. This measures that
on the manually verified BigEarthNet.txt *bench* split -- the held-out set
the encoder and heads were never fitted on (train_ben_txt / train_joint
train only on split=="train") -- restricted to the patches that carry BOTH
an S2 optical patch and an S1 (VV/VH) SAR patch on disk.

Prediction is multi-label over the 19 BigEarthNet-19 classes, reproducing
the deployed paths in satquery.stage2.neural exactly:

    optical : sigmoid(caption_head( gsd_cond( opt_enc(band_proj(bands)) ) ))
    sar     : sigmoid(s1_head(     gsd_cond( sar_enc(vv-mu, vh-mu)     ) ))
    fused   : late fusion of the two probability vectors (mean / max)

Metric: per-class IoU_c = TP_c / (TP_c + FP_c + FN_c) across the held-out
patches (multi-label Jaccard), plus macro-mIoU (mean over classes with
ground-truth support), micro-IoU, macro-F1 and example-based IoU. Reported
for optical-only, SAR-only, mean-fused and max-fused, at the deployed 0.5
threshold and at per-class thresholds tuned on the train split (a fair
"operated" setting; tuning never sees the held-out data).

Writes data/ben_fuse_iou_metrics.json (folds into the benchmark PDF via
scripts/gen_benchmark_report.py).

Usage:
    python scripts/eval_ben_fuse_iou.py            # full bench held-out set
    python scripts/eval_ben_fuse_iou.py --limit 200    # smoke subset
    python scripts/eval_ben_fuse_iou.py --no-tune       # 0.5 threshold only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
BEN = ROOT / "data" / "ben_txt"
PATCH_DIR = BEN / "patches"
S1_DIR = BEN / "s1_patches"
MANIFEST = BEN / "manifest.json"
NAME_MAP = BEN / "s1_name_map.json"
CKPT = ROOT / "models" / "backbone_ft.pth"
HEADS = ROOT / "models" / "ben_txt_heads.pth"
OUT = ROOT / "data" / "ben_fuse_iou_metrics.json"

# S2 band file -> encoder slot (train_ben_txt.py BAND_TO_SLOT / SLOT_ORDER)
BAND_TO_SLOT = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08", "B09": "nir09",
    "B11": "swir1", "B12": "swir2",
}
SLOT_ORDER = ["coastal", "blue", "green", "red", "rededge1", "rededge2",
              "rededge3", "nir", "nir08", "nir09", "swir1", "swir2"]
SLOT_TO_BAND = {v: k for k, v in BAND_TO_SLOT.items()}

# dB means the S1 head was trained against (train_joint.s1_features).
S1_VV_MEAN, S1_VH_MEAN = -12.643863677978516, -19.352558135986328

_dev = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- loading
def _cache_tag() -> str:
    """First 8 hex of the checkpoint sha -- cache key per encoder version."""
    h = hashlib.sha256()
    with open(CKPT, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:8]


def _caption_head() -> nn.Sequential:
    """Optical BEN-19 head (satquery.stage2.neural._caption_head)."""
    return nn.Sequential(
        nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 19))


def load_backbone():
    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone().to(_dev)
    sd = torch.load(CKPT, map_location=_dev, weights_only=True)
    bb.load_state_dict(sd, strict=False)
    return bb.eval()


def load_heads():
    blob = torch.load(HEADS, map_location=_dev, weights_only=True)
    if blob.get("arch") != "v2":
        raise RuntimeError(f"{HEADS} is not the joint v2 artifact")
    cap = _caption_head()
    cap.load_state_dict(blob["caption_head"])
    s1 = nn.Linear(2048, 19)
    s1.load_state_dict(blob["s1_head"])
    return cap.to(_dev).eval(), s1.to(_dev).eval()


def held_out_rows(manifest, name_map, split="bench"):
    """Manifest rows in `split` that have BOTH an S2 and an S1 patch."""
    rows = []
    for r in manifest["patches"]:
        pid = r["patch_id"]
        if r.get("split") != split:
            continue
        if not (PATCH_DIR / f"{pid}.npz").exists():
            continue
        s1n = name_map.get(pid)
        if not s1n or not (S1_DIR / f"{s1n}.npz").exists():
            continue
        rows.append(r)
    return rows


# --------------------------------------------------------------- encoding
@torch.no_grad()
def _encode_optical(bb, pids, batch=64):
    """(N, 2048) optical features -- band_proj -> opt_enc -> gsd_cond(10)."""
    out = []
    for i in range(0, len(pids), batch):
        chunk = pids[i:i + batch]
        arrs = []
        for pid in chunk:
            z = np.load(PATCH_DIR / f"{pid}.npz")
            arrs.append(np.stack(
                [z[SLOT_TO_BAND[s]] for s in SLOT_ORDER]).astype(
                    np.float32) / 10000.0)
        x = torch.from_numpy(np.stack(arrs)).to(_dev)     # (B,12,120,120)
        bands = {s: x[:, j] for j, s in enumerate(SLOT_ORDER)}
        with torch.autocast("cuda", enabled=_dev == "cuda"):
            f = bb.opt_enc(bb.band_proj(bands)).flatten(1)
            f = bb.gsd_cond(f, gsd=10.0)
        out.append(f.float().cpu().numpy())
        print(f"  opt {min(i + batch, len(pids))}/{len(pids)}", flush=True)
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def _encode_sar(bb, pids, name_map, batch=64):
    """(N, 2048) SAR features -- (vv-mu, vh-mu) -> sar_enc -> gsd_cond(10)."""
    out = []
    for i in range(0, len(pids), batch):
        chunk = pids[i:i + batch]
        arrs = []
        for pid in chunk:
            z = np.load(S1_DIR / f"{name_map[pid]}.npz")
            arrs.append(np.stack([z["VV"], z["VH"]]).astype(np.float32))
        x = torch.from_numpy(np.stack(arrs)).to(_dev)     # (B,2,120,120)
        vv = x[:, 0] - S1_VV_MEAN
        vh = x[:, 1] - S1_VH_MEAN
        with torch.autocast("cuda", enabled=_dev == "cuda"):
            f = bb.sar_enc(vv, vh)
            f = bb.gsd_cond(f, gsd=10.0)
        out.append(f.float().cpu().numpy())
        print(f"  sar {min(i + batch, len(pids))}/{len(pids)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def features(bb, rows, name_map, tag):
    """Optical + SAR features for `rows`, cached per encoder sha + tag."""
    cache = BEN / f"fuse_feat_{tag}.{_cache_tag()}.npz"
    pids = [r["patch_id"] for r in rows]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have_o = dict(zip([str(i) for i in z["id"]], z["opt"]))
        have_s = dict(zip([str(i) for i in z["id"]], z["sar"]))
        if all(p in have_o for p in pids):
            print(f"  [{tag}] features cached ({len(pids)})", flush=True)
            return (np.stack([have_o[p] for p in pids]),
                    np.stack([have_s[p] for p in pids]))
    opt = _encode_optical(bb, pids)
    sar = _encode_sar(bb, pids, name_map)
    np.savez(cache, id=np.array(pids), opt=opt, sar=sar)
    return opt, sar


# ---------------------------------------------------------------- metrics
def label_matrix(rows, labels):
    idx = {l: i for i, l in enumerate(labels)}
    Y = np.zeros((len(rows), len(labels)), dtype=bool)
    for i, r in enumerate(rows):
        for l in r["labels"]:
            if l in idx:
                Y[i, idx[l]] = True
    return Y


def metrics(P, Y, thr):
    """Multi-label metrics at threshold `thr` (scalar or per-class array)."""
    pred = P >= thr
    tp = (pred & Y).sum(0).astype(np.int64)
    fp = (pred & ~Y).sum(0).astype(np.int64)
    fn = (~pred & Y).sum(0).astype(np.int64)
    support = Y.sum(0).astype(np.int64)
    iou_c = tp / np.maximum(tp + fp + fn, 1)
    f1_c = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    has = support > 0
    inter = (pred & Y).sum(1)
    union = (pred | Y).sum(1)
    ex_iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0).mean()
    tps, fps, fns = int(tp.sum()), int(fp.sum()), int(fn.sum())
    return {
        "macro_miou": round(float(iou_c[has].mean()), 4),
        "micro_iou": round(float(tps / max(tps + fps + fns, 1)), 4),
        "macro_f1": round(float(f1_c[has].mean()), 4),
        "micro_f1": round(float(2 * tps / max(2 * tps + fps + fns, 1)), 4),
        "example_iou": round(float(ex_iou), 4),
        "exact_match": round(float((pred == Y).all(1).mean()), 4),
        "per_class_iou": {int(i): round(float(iou_c[i]), 4)
                          for i in range(len(iou_c))},
        "per_class_f1": {int(i): round(float(f1_c[i]), 4)
                         for i in range(len(f1_c))},
        "support": {int(i): int(support[i]) for i in range(len(support))},
    }


def tune_thresholds(P, Y, grid=None):
    """Per-class threshold maximising F1 on (P, Y); classes with no support
    keep 0.5. Returns a (C,) array. Meant to be fit on TRAIN only."""
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.05)
    C = P.shape[1]
    thr = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        y = Y[:, c]
        if y.sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = P[:, c] >= t
            tp = int((pred & y).sum())
            fp = int((pred & ~y).sum())
            fn = int((~pred & y).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[c] = best_t
    return thr


# ------------------------------------------------------------------- main
@torch.no_grad()
def probs(cap, s1, opt_feat, sar_feat, batch=2048):
    """Return dict of (N,19) sigmoid-prob matrices for each setting."""
    po, ps = [], []
    for i in range(0, len(opt_feat), batch):
        xo = torch.from_numpy(opt_feat[i:i + batch]).to(_dev)
        xs = torch.from_numpy(sar_feat[i:i + batch]).to(_dev)
        po.append(torch.sigmoid(cap(xo)).cpu().numpy())
        ps.append(torch.sigmoid(s1(xs)).cpu().numpy())
    p_opt = np.concatenate(po).astype(np.float32)
    p_sar = np.concatenate(ps).astype(np.float32)
    return {"optical": p_opt, "sar": p_sar,
            "fused_mean": (p_opt + p_sar) / 2.0,
            "fused_max": np.maximum(p_opt, p_sar)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap held-out patches (smoke run)")
    ap.add_argument("--split", default="bench", choices=["bench", "train"])
    ap.add_argument("--no-tune", action="store_true",
                    help="report the deployed 0.5 threshold only")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    labels = manifest["labels"]
    name_map = json.loads(NAME_MAP.read_text())
    bb = load_backbone()
    cap, s1 = load_heads()

    ho_rows = held_out_rows(manifest, name_map, args.split)
    if args.limit:
        ho_rows = ho_rows[:args.limit]
    print(f"held-out ({args.split}) patches with both S2+S1: {len(ho_rows)}",
          flush=True)

    t0 = time.time()
    ho_opt, ho_sar = features(bb, ho_rows, name_map, args.split)
    Yh = label_matrix(ho_rows, labels)
    Ph = probs(cap, s1, ho_opt, ho_sar)

    settings = ["optical", "sar", "fused_mean", "fused_max"]
    result = {
        "benchmark": "BigEarthNet.txt bench split (held-out) "
                     "-- optical+SAR fusion, per-class IoU",
        "n_heldout": len(ho_rows),
        "classes": labels,
        "note": "multi-label per-class IoU (Jaccard) over held-out patches; "
                "fused = late fusion of the deployed optical (caption_head) "
                "and SAR (s1_head) BEN-19 probability vectors.",
        "threshold_0.5": {s: metrics(Ph[s], Yh, 0.5) for s in settings},
    }

    if not args.no_tune:
        tr_rows = held_out_rows(manifest, name_map, "train")
        print(f"train patches for threshold tuning: {len(tr_rows)}",
              flush=True)
        tr_opt, tr_sar = features(bb, tr_rows, name_map, "train")
        Yt = label_matrix(tr_rows, labels)
        Pt = probs(cap, s1, tr_opt, tr_sar)
        tuned = {}
        for s in settings:
            thr = tune_thresholds(Pt[s], Yt)
            tuned[s] = metrics(Ph[s], Yh, thr)
            tuned[s]["thresholds"] = [round(float(t), 2) for t in thr]
        result["n_train_tuning"] = len(tr_rows)
        result["threshold_tuned"] = tuned

    # fusion gain vs the better single modality (macro-mIoU)
    def gain(block):
        best_single = max(block["optical"]["macro_miou"],
                          block["sar"]["macro_miou"])
        return {
            "best_single_macro_miou": round(best_single, 4),
            "fused_mean_macro_miou": block["fused_mean"]["macro_miou"],
            "fused_max_macro_miou": block["fused_max"]["macro_miou"],
            "gain_mean_vs_best_single": round(
                block["fused_mean"]["macro_miou"] - best_single, 4),
            "gain_max_vs_best_single": round(
                block["fused_max"]["macro_miou"] - best_single, 4),
        }
    result["fusion_gain"] = {"threshold_0.5": gain(result["threshold_0.5"])}
    if not args.no_tune:
        result["fusion_gain"]["threshold_tuned"] = gain(
            result["threshold_tuned"])

    OUT.write_text(json.dumps(result, indent=2))
    _print_summary(result, labels)
    print(f"\nsaved -> {OUT}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


def _print_summary(result, labels):
    for block_name in ("threshold_0.5", "threshold_tuned"):
        block = result.get(block_name)
        if not block:
            continue
        print("\n" + "=" * 68)
        print(f"BigEarthNet fuse per-class IoU -- {block_name} "
              f"(n={result['n_heldout']})")
        print("=" * 68)
        print(f"{'setting':12s} {'macro-mIoU':>10s} {'micro-IoU':>10s} "
              f"{'macro-F1':>9s} {'ex-IoU':>8s} {'exact':>7s}")
        for s in ("optical", "sar", "fused_mean", "fused_max"):
            m = block[s]
            print(f"{s:12s} {m['macro_miou']:>10.4f} {m['micro_iou']:>10.4f} "
                  f"{m['macro_f1']:>9.4f} {m['example_iou']:>8.4f} "
                  f"{m['exact_match']:>7.4f}")
        g = result["fusion_gain"][block_name]
        print(f"  fusion gain (mean vs best single, macro-mIoU): "
              f"{g['gain_mean_vs_best_single']:+.4f}")


if __name__ == "__main__":
    sys.exit(main())

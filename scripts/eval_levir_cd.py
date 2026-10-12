"""LEVIR-CD change-mask evaluation for the deployed ChangeHead (F1 + IoU).

The CDVQA numbers (benchmark PDF) score change *questions*; this scores the
change *mask* the deployed ChangeHead actually draws, against pixel-level
ground truth. Dataset: LEVIR-CD test split, 256x256 cropped
(HF ericyu/LEVIRCD_Cropped_256, ~70 MB) -- 2,048 co-registered bi-temporal
RGB tile pairs with binary building-change masks.

The ChangeHead is UNSUPERVISED change-vector analysis (CVA): change =
per-pixel L2 spectral difference exceeding a fraction of the scene's 90th
percentile magnitude (satquery.stage2.heads.ChangeHead._cva). LEVIR-CD masks
mark BUILDING change only, so generic CVA -- which also fires on seasonal /
vegetation / lighting change -- is being asked a narrower question than it
answers. The result is therefore an honest bound on unsupervised change
LOCALISATION, reported at the deployed threshold (cva_thresh=1.5) and at the
best-F1 threshold found by a sweep (so the ceiling is visible too), not a
claim that the system is a building-change detector.

Writes data/levir_cd_metrics.json (folds into the benchmark PDF).

Usage:
    python scripts/eval_levir_cd.py            # full test split
    python scripts/eval_levir_cd.py --limit 200    # smoke subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "levir_cd_metrics.json"
HF_REPO = "ericyu/LEVIRCD_Cropped_256"

# ChangeHead._cva maps cva_thresh -> a norm threshold of cva_thresh/3.0;
# the deployed default cva_thresh=1.5 => norm threshold 0.5.
DEPLOYED_CVA_THRESH = 1.5


# ------------------------------------------------------------ change map
def cva_norm(rgb_a, rgb_b):
    """Reproduce ChangeHead._cva's normalised magnitude for RGB tiles.

    _cva takes the L2 norm of the per-band difference, divides by its 90th
    percentile, and thresholds. Returning the pre-threshold ``norm`` lets us
    sweep the threshold once per tile instead of re-encoding. Validated
    against the real ChangeHead._cva in ``_selfcheck``.
    """
    a = rgb_a.astype(np.float32)
    b = rgb_b.astype(np.float32)
    vecs = [(b[..., c] - a[..., c]) ** 2 for c in range(a.shape[-1])]
    mag = np.sqrt(sum(vecs) / len(vecs))
    p90 = float(np.percentile(mag, 90))
    if p90 < 1e-8:
        return np.zeros(mag.shape, dtype=np.float32)
    return (mag / p90).astype(np.float32)


def _tile(rgb, tid):
    from satquery.contracts import Tile
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return Tile(id=tid, sensor_id="generic_rgb",
                bands={"red": r.astype(np.float32),
                       "green": g.astype(np.float32),
                       "blue": b.astype(np.float32)}, gsd=0.5)


def _selfcheck(a, b):
    """Assert the inline norm+threshold equals ChangeHead._cva byte for byte."""
    from satquery.stage2.heads import ChangeHead
    ref = ChangeHead._cva(_tile(a, "a"), _tile(b, "b"), DEPLOYED_CVA_THRESH)
    mine = (cva_norm(a, b) > DEPLOYED_CVA_THRESH / 3.0).astype(np.float32)
    ok = np.array_equal(ref, mine)
    print(f"  self-check vs ChangeHead._cva: "
          f"{'OK' if ok else 'MISMATCH'} ({int(mine.sum())} changed px)",
          flush=True)
    return ok


# --------------------------------------------------------------- metrics
def prf_iou(tp, fp, fn):
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    return (round(float(prec), 4), round(float(rec), 4),
            round(float(f1), 4), round(float(iou), 4))


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap test tiles (smoke run)")
    args = ap.parse_args()

    from datasets import load_dataset
    print(f"loading {HF_REPO} test split ...", flush=True)
    ds = load_dataset(HF_REPO, split="test", verification_mode="no_checks")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    print(f"test tiles: {n}", flush=True)

    grid = [round(x, 2) for x in np.arange(0.1, 3.01, 0.1)]
    deployed_t = DEPLOYED_CVA_THRESH / 3.0            # 0.5
    if deployed_t not in grid:
        grid.append(deployed_t)
        grid = sorted(grid)
    acc = {t: [0, 0, 0] for t in grid}                # tp, fp, fn (change)
    per_tile_iou = []
    gt_pos = tot_px = 0
    checked = False
    t0 = time.time()

    for i in range(n):
        ex = ds[i]
        a = np.asarray(ex["imageA"])
        b = np.asarray(ex["imageB"])
        gt = np.asarray(ex["label"].convert("L")) > 127
        if not checked:
            _selfcheck(a, b)
            checked = True
        norm = cva_norm(a, b)
        for t in grid:
            pred = norm > t
            acc[t][0] += int((pred & gt).sum())
            acc[t][1] += int((pred & ~gt).sum())
            acc[t][2] += int((~pred & gt).sum())
        pred_d = norm > deployed_t
        inter = int((pred_d & gt).sum())
        union = int((pred_d | gt).sum())
        per_tile_iou.append(inter / union if union > 0 else 1.0)
        gt_pos += int(gt.sum())
        tot_px += gt.size
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    # deployed + swept-best (by global change-class F1)
    dep = prf_iou(*acc[deployed_t])
    f1_by_t = {t: prf_iou(*acc[t])[2] for t in grid}
    best_t = max(f1_by_t, key=f1_by_t.get)
    best = prf_iou(*acc[best_t])
    _write(n, gt_pos, tot_px, deployed_t, dep, best_t, best,
           per_tile_iou, grid, acc, t0)
    return 0


def _write(n, gt_pos, tot_px, deployed_t, dep, best_t, best,
           per_tile_iou, grid, acc, t0):
    def block(t, prf):
        return {"cva_thresh": round(t * 3.0, 3), "norm_threshold": round(t, 3),
                "precision": prf[0], "recall": prf[1], "f1": prf[2],
                "iou": prf[3]}
    result = {
        "benchmark": "LEVIR-CD test (256px crops) -- ChangeHead change-mask "
                     "F1 / IoU",
        "source": HF_REPO,
        "n_tiles": n,
        "change_pixel_fraction": round(gt_pos / max(tot_px, 1), 4),
        "method": "unsupervised change-vector analysis (ChangeHead._cva); "
                  "metrics are for the change (positive) class, computed "
                  "globally over all test pixels.",
        "note": "LEVIR-CD masks mark building change only; CVA detects all "
                "spectral change, so this bounds unsupervised change "
                "localisation rather than measuring a building-change "
                "specialist.",
        "deployed": block(deployed_t, dep),
        "best_f1": block(best_t, best),
        "mean_per_tile_iou_deployed": round(
            float(np.mean(per_tile_iou)), 4),
        "f1_sweep": {str(t): prf_iou(*acc[t])[2] for t in grid},
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 64)
    print(f"LEVIR-CD change-mask (n={n} tiles, "
          f"{result['change_pixel_fraction']:.1%} change pixels)")
    print("=" * 64)
    print(f"{'setting':22s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'IoU':>7s}")
    print(f"{'deployed (cva=1.5)':22s} {dep[0]:>7.4f} {dep[1]:>7.4f} "
          f"{dep[2]:>7.4f} {dep[3]:>7.4f}")
    print(f"{'best-F1 (cva=%.2f)' % (best_t * 3):22s} {best[0]:>7.4f} "
          f"{best[1]:>7.4f} {best[2]:>7.4f} {best[3]:>7.4f}")
    print("=" * 64)
    print(f"saved -> {OUT}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

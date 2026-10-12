"""Look at CDVQA pairs directly.

1. Save 8 side-by-side montages (before | after | 8x diff grid)
   for yes-change and no-change pairs each.
2. Compare top-cell diff statistics between yes vs no groups.
"""
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = ROOT / "data" / "bench"
OUT = BENCH / "cdvqa_look"
OUT.mkdir(exist_ok=True)

from eval_benchmarks import CDVQA_TRAIN_SHARDS, cdvqa_examples, _cdvqa_iter


def _img(raw):
    im = Image.open(BytesIO(raw)).convert("RGB").resize(
        (256, 256), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-3:
        return np.full_like(a, 128)
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255)


ex, _ = cdvqa_examples(CDVQA_TRAIN_SHARDS, 0)
qa = {e[0]: e for e in ex}
saved = {"yes": 0, "no": 0}
stats = {"yes": [], "no": []}
seen = set()
for t, k, j in _cdvqa_iter(CDVQA_TRAIN_SHARDS, 0):
    if k in seen or k not in qa:
        continue
    try:
        f0 = _img(t.extractfile(k + ".0.img").read())
        f1 = _img(t.extractfile(k + ".1.img").read())
    except (KeyError, tarfile.TarError):
        continue
    seen.add(k)
    e = qa[k]
    if e[4] != "change_or_not":
        continue
    lab = "yes" if e[3] == 0 else "no"
    d = np.abs(f0 - f1).mean(-1)                    # (256,256)
    cells = d.reshape(8, 32, 8, 32).mean((1, 3))     # 8x8
    stats[lab].append((cells.max(), np.sort(cells.ravel())[-5:].mean(),
                       cells.mean(), d.std()))
    if saved[lab] < 8:
        grid = (cells / max(cells.max(), 1e-3) * 255).astype(np.uint8)
        gm = Image.fromarray(grid).resize((256, 256), Image.NEAREST)
        gm3 = np.stack([np.asarray(gm)] * 3, -1)
        row = Image.fromarray(
            np.concatenate([f0.astype(np.uint8), f1.astype(np.uint8),
                            gm3], 1))
        row.save(OUT / f"{lab}_{saved[lab]}.png")
        saved[lab] += 1

for lab in ("yes", "no"):
    s = np.asarray(stats[lab]) if stats[lab] else np.zeros((0, 4))
    print(f"{lab}-change pairs n={len(s)}")
    for i, nm in enumerate(["max-cell diff", "top5-cell diff",
                            "mean diff", "pixel std"]):
        if len(s):
            print(f"   {nm:16s} median {np.median(s[:, i]):.4f}  "
                  f"p25 {np.percentile(s[:, i], 25):.4f}  "
                  f"p75 {np.percentile(s[:, i], 75):.4f}")

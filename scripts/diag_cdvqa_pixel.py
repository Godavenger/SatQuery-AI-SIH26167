"""Does ANY pixel-difference signal separate CDVQA labels?

Extracts cheap spatial statistics from the raw before/after images
(absolute difference pooled over a coarse spatial grid), then runs
logistic-regression CV. If this cannot beat the majority baseline,
the labels carry no image evidence at this scale and no encoder
architecture will rescue them.
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

from eval_benchmarks import CDVQA_TRAIN_SHARDS, cdvqa_examples, _cdvqa_iter


def _img(raw):
    im = Image.open(BytesIO(raw)).convert("RGB").resize(
        (256, 256), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-3:
        return np.full_like(a, 0.5)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def spatial_diff_stats(f0, f1):
    """Coarse per-cell mean |diff| + brightness change -> feature vector."""
    d = np.abs(f0 - f1)                      # (256,256,3)
    cells = []
    for gy in range(8):
        for gx in range(8):
            c = d[gy * 32:(gy + 1) * 32, gx * 32:(gx + 1) * 32]
            cells.append(c.mean())           # per-cell change magnitude
    cells = np.asarray(cells)
    b0 = f0.mean(axis=(0, 1))
    b1 = f1.mean(axis=(0, 1))
    return np.concatenate([cells, b0, b1, b1 - b0]).astype(np.float32)


# collect pair stats from all valid train shards
X, y, types = [], [], []
_ex, _g = cdvqa_examples(CDVQA_TRAIN_SHARDS, 0)
label_of = {e[0]: e[3] for e in _ex}
type_of = {e[0]: e[4] for e in _ex}
seen = set()
for t, k, j in _cdvqa_iter(CDVQA_TRAIN_SHARDS, 0):
    if k in seen or k not in label_of:
        continue
    try:
        f0 = _img(t.extractfile(k + ".0.img").read())
        f1 = _img(t.extractfile(k + ".1.img").read())
    except (KeyError, tarfile.TarError):
        continue
    seen.add(k)
    X.append(spatial_diff_stats(f0, f1))
    y.append(label_of[k])
    types.append(type_of[k])
X = np.stack(X)
y = np.array(y)
t_arr = np.array(types)
print(f"pairs {len(X)}  dims {X.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000))
acc = cross_val_score(pipe, X, y, cv=5)
print(f"ALL  5-fold CV {acc.mean():.4f} +- {acc.std():.4f} "
      f"(majority {max(np.mean(y==0), np.mean(y==1)):.4f})")
for t in sorted(set(t_arr)):
    m = t_arr == t
    if m.sum() < 200:
        continue
    a = cross_val_score(pipe, X[m], y[m], cv=5)
    print(f"  {t:18s} CV {a.mean():.4f} (n={m.sum()}, "
          f"majority {max(np.mean(y[m]==0), np.mean(y[m]==1)):.4f})")

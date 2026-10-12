"""Diagnose why the CDVQA supervised scorer won't train.

1. feature scale (image feats vs text embeddings)
2. does ANY signal exist? sklearn logistic regression, 5-fold CV
3. label balance per question type
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = ROOT / "data" / "bench"

from eval_benchmarks import (CDVQA_TRAIN_SHARDS, cdvqa_examples,
                             cdvqa_pair_features, _change_vec)

print("== collecting examples ==")
ex, _ = cdvqa_examples(CDVQA_TRAIN_SHARDS, 0)
tr_feats = cdvqa_pair_features(CDVQA_TRAIN_SHARDS, 0,
                                BENCH / "feat_cdvqa_train.npz")
cv = _change_vec(tr_feats)

keys = [e[0] for e in ex if e[0] in cv]
X = np.stack([cv[k] for k in keys])
y = np.array([e[3] for e in ex if e[0] in cv])  # 0=yes 1=no
types = [e[4] for e in ex if e[0] in cv]
print(f"pairs: {len(X)}  feats: {X.shape}  yes: {(y==0).mean():.3f}")
for t in sorted(set(types)):
    m = np.array([tt == t for tt in types])
    print(f"  {t:18s} n={m.sum():4d}  yes={np.nanmean(y[m]==0):.3f}")

h = X.shape[1] // 2
print("\n== feature scale ==")
print(f"|f0-f1| L2 mean {np.linalg.norm(X[:, :h], axis=1).mean():.2f}")
print(f"|f0+f1| L2 mean {np.linalg.norm(X[:, h:], axis=1).mean():.2f}")
print(f"per-dim std  diff {X[:, :h].std():.4f}  sum {X[:, h:].std():.4f}")

print("\n== logistic regression (standardized) ==")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000, C=1.0))
acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV: {acc.mean():.4f} +- {acc.std():.4f}  "
      f"(majority {max(np.mean(y==0), np.mean(y==1)):.4f})")

# per-type CV on the single biggest type to see if signal is type-specific
t_arr = np.array(types)
for t in sorted(set(t_arr)):
    m = t_arr == t
    if m.sum() < 200:
        continue
    a = cross_val_score(pipe, X[m], y[m], cv=5, scoring="accuracy")
    print(f"  {t:18s} CV {a.mean():.4f} (n={m.sum()}, "
          f"majority {max(np.mean(y[m]==0), np.mean(y[m]==1)):.4f})")

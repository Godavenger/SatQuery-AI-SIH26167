"""Dependency-free caption + grounding metrics (offline, pure Python/NumPy).

pycocoevalcap needs a JVM for METEOR/SPICE and is fragile on Windows; nltk /
rouge_score are not installed. The project targets offline packaging, so the
canonical caption metrics are implemented here directly and unit-checked in
__main__:

    bleu(cands, refs)        -> BLEU-1..4 (corpus, clipped, brevity penalty)
    rouge_l(cands, refs)     -> ROUGE-L F (LCS, beta=1.2, COCO convention)
    cider_d(cands, refs)     -> CIDEr-D (TF-IDF n-gram cosine, sigma=6)
    box_iou(a, b)            -> IoU of two [x0,y0,x1,y1] boxes
    grounding_scores(...)    -> mean IoU + Acc@0.5 over matched boxes

Each cand is one string; each ref is a list of reference strings (VRSBench
ships one reference caption per image, so the list has length 1).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_PUNC = re.compile(r"[^a-z0-9 ]+")


def tokenize(s: str) -> list[str]:
    return _PUNC.sub(" ", str(s).lower()).split()


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


# ------------------------------------------------------------------- BLEU
def bleu(cands: list[str], refs: list[list[str]], max_n: int = 4) -> dict:
    """Corpus BLEU-1..max_n (Papineni et al.), clipped, brevity penalty."""
    clipped = [0] * max_n
    total = [0] * max_n
    c_len = r_len = 0
    for cand, ref_list in zip(cands, refs):
        ct = tokenize(cand)
        c_len += len(ct)
        # closest reference length for the brevity penalty
        r_len += min((len(tokenize(r)) for r in ref_list),
                     key=lambda rl: (abs(rl - len(ct)), rl))
        for n in range(1, max_n + 1):
            cg = _ngrams(ct, n)
            if not cg:
                continue
            maxref: Counter = Counter()
            for r in ref_list:
                rg = _ngrams(tokenize(r), n)
                for g, cnt in rg.items():
                    if cnt > maxref[g]:
                        maxref[g] = cnt
            clipped[n - 1] += sum(min(cnt, maxref[g]) for g, cnt in cg.items())
            total[n - 1] += sum(cg.values())
    precisions = [(clipped[i] / total[i]) if total[i] else 0.0
                  for i in range(max_n)]
    bp = 1.0 if c_len > r_len else (math.exp(1 - r_len / c_len)
                                    if c_len > 0 else 0.0)
    out = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p = precisions[n - 1]
        log_sum += math.log(p) if p > 0 else -1e9
        out[f"bleu_{n}"] = round(bp * math.exp(log_sum / n), 4)
    return out


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(cands: list[str], refs: list[list[str]], beta: float = 1.2) -> dict:
    """Sentence-level ROUGE-L F averaged over the corpus (COCO beta=1.2)."""
    scores = []
    for cand, ref_list in zip(cands, refs):
        ct = tokenize(cand)
        best = 0.0
        for r in ref_list:
            rt = tokenize(r)
            l = _lcs(ct, rt)
            if l == 0 or not ct or not rt:
                continue
            p = l / len(ct)
            rec = l / len(rt)
            f = ((1 + beta ** 2) * p * rec) / (rec + beta ** 2 * p + 1e-12)
            best = max(best, f)
        scores.append(best)
    return {"rouge_l": round(float(np.mean(scores)) if scores else 0.0, 4)}


def cider_d(cands: list[str], refs: list[list[str]], max_n: int = 4,
            sigma: float = 6.0) -> dict:
    """CIDEr-D (Vedantam et al. 2015): TF-IDF n-gram cosine with a Gaussian
    length penalty. IDF is computed over the reference corpus (one document
    per image). Returns the mean CIDEr-D and per-n breakdown."""
    N = len(cands)
    df = [defaultdict(float) for _ in range(max_n)]
    ref_grams = []
    for ref_list in refs:
        per_n = []
        for n in range(max_n):
            g = Counter()
            for r in ref_list:
                g += _ngrams(tokenize(r), n + 1)
            per_n.append(g)
            for gram in set(g):
                df[n][gram] += 1
        ref_grams.append(per_n)
    log_N = math.log(max(N, 1))

    def vec(counts, n):
        v, norm = {}, 0.0
        for g, c in counts.items():
            w = c * (log_N - math.log(max(df[n][g], 1.0)))
            v[g] = w
            norm += w * w
        return v, math.sqrt(norm)

    per_n_scores = [[] for _ in range(max_n)]
    for cand, per_n_ref, ref_list in zip(cands, ref_grams, refs):
        ct = tokenize(cand)
        lr = float(np.mean([len(tokenize(r)) for r in ref_list]))
        for n in range(max_n):
            cv, cn = vec(_ngrams(ct, n + 1), n)
            rv, rn = vec(per_n_ref[n], n)
            if cn == 0 or rn == 0:
                per_n_scores[n].append(0.0)
                continue
            dot = sum(w * rv.get(g, 0.0) for g, w in cv.items())
            pen = math.exp(-((len(ct) - lr) ** 2) / (2 * sigma ** 2))
            per_n_scores[n].append(10.0 * (dot / (cn * rn)) * pen)
    by_n = {f"cider_d_{n+1}": round(float(np.mean(per_n_scores[n])), 4)
            for n in range(max_n)}
    overall = float(np.mean([np.mean(per_n_scores[n]) for n in range(max_n)]))
    return {"cider_d": round(overall, 4), **by_n}


def box_iou(a, b) -> float:
    """IoU of two axis-aligned boxes [x0, y0, x1, y1]."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = ua + ub - inter
    return float(inter / denom) if denom > 0 else 0.0


def grounding_scores(pred_boxes: list, gt_boxes: list) -> dict:
    """Greedy one-to-one match by IoU; mean IoU + Acc@0.5 / Acc@0.7."""
    ious, used = [], set()
    for gt in gt_boxes:
        best, bj = 0.0, -1
        for j, pb in enumerate(pred_boxes):
            if j in used:
                continue
            v = box_iou(pb, gt)
            if v > best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
        ious.append(best)
    if not ious:
        return {"mean_iou": 0.0, "acc@0.5": 0.0, "acc@0.7": 0.0, "n": 0}
    ious = np.asarray(ious)
    return {"mean_iou": round(float(ious.mean()), 4),
            "acc@0.5": round(float((ious >= 0.5).mean()), 4),
            "acc@0.7": round(float((ious >= 0.7).mean()), 4),
            "n": int(len(ious))}


if __name__ == "__main__":
    c = ["the cat sat on the mat"]
    r = [["the cat sat on the mat"]]
    b = bleu(c, r)
    assert b["bleu_4"] == 1.0, b
    assert rouge_l(c, r)["rouge_l"] == 1.0
    assert abs(box_iou([0, 0, 2, 2], [1, 1, 3, 3]) - (1 / 7)) < 1e-6
    part = bleu(["the cat"], [["the cat sat on the mat"]])
    assert 0.0 < part["bleu_1"] <= 1.0
    g = grounding_scores([[0, 0, 2, 2]], [[0, 0, 2, 2], [5, 5, 6, 6]])
    assert g["acc@0.5"] == 0.5, g
    print("vlm_metrics self-check OK")
    print("  BLEU", b)
    print("  ROUGE-L", rouge_l(c, r))
    print("  grounding", g)

"""Zero-shot + supervised evaluation on RSVQA-HR (high-resolution RSVQA).

Mirrors scripts/eval_benchmarks.py's RSVQA-LR path but for the high-res split
(Zenodo record 6344367, USGS 15 cm aerial, 512x512 tiles). Reuses that
module's encoder/scorer machinery wholesale -- the only differences are the
data layout and the answer vocabulary:

    presence, comp  -> yes / no
    count           -> the 5 RSVQA-HR count bins (0, between 1 and 10, ...)
    area            -> the RSVQA-HR area bins

Options per type are discovered from the split's own answers (as in the
VRSBench path), so the binned count/area classes are handled without
hard-coding. Zero-shot uses the BigEarthNet.txt joint scorer directly;
supervised adapts a fresh scorer on the HR train split.

Usage:
    python scripts/eval_rsvqa_hr.py --mode both
    python scripts/eval_rsvqa_hr.py --mode zero-shot --limit 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_benchmarks as EB

ROOT = Path(__file__).resolve().parents[1]
HR = ROOT / "data" / "bench" / "rsvqa_hr"
# image dir is resolved at runtime (the tar extracts to one of these)
IMG_CANDIDATES = [HR / "Data", HR / "Images", HR / "Data" / "Images",
                  HR / "images", HR]

Q = {s: HR / f"USGS_split_{s}_questions.json"
     for s in ("train", "val", "test", "test_phili")}
A = {s: HR / f"USGS_split_{s}_answers.json"
     for s in ("train", "val", "test", "test_phili")}
IMJ = {s: HR / f"USGS_split_{s}_images.json"
       for s in ("train", "val", "test", "test_phili")}


def _img_dir():
    for d in IMG_CANDIDATES:
        if d.exists() and any(d.glob("*.tif")):
            return d
    raise FileNotFoundError(
        "RSVQA-HR image tiles not found; extract Images.tar first "
        f"(looked in {[str(c) for c in IMG_CANDIDATES]})")


def _img_path(img_dir, img_id):
    for ext in (".tif", ".tiff", ".png", ".jpg"):
        p = img_dir / f"{img_id}{ext}"
        if p.exists():
            return p
    return img_dir / f"{img_id}.tif"


def _option_pools(split, per_type=8):
    """type -> ranked answer vocabulary from a split's answers (diagnostic)."""
    ans = {a["id"]: a for a in json.load(open(A[split]))["answers"]}
    qs = json.load(open(Q[split]))["questions"]
    pools = defaultdict(Counter)
    for q in qs:
        if not q.get("active", True):
            continue
        a = ans.get(q["answers_ids"][0])
        if a is None or not a.get("active", True):
            continue
        pools[q["type"]][EB._norm(a["answer"])] += 1
    return {t: [k for k, _ in c.most_common(per_type)] for t, c in pools.items()}


# The Zenodo release stores raw exact count/area answers; the RSVQA-HR paper's
# protocol bins them into 5 classes before scoring. Reproduce those bins so
# count/area become clean MCQ (and comparable to the literature).
COUNT_BINS = ["0", "between 1 and 10", "between 11 and 100",
              "between 101 and 1000", "more than 1000"]
AREA_BINS = ["0m2", "between 1m2 and 10m2", "between 10m2 and 100m2",
             "between 100m2 and 1000m2", "more than 1000m2"]


def _range_bin(v, bins):
    if v == 0:
        return bins[0]
    if v <= 10:
        return bins[1]
    if v <= 100:
        return bins[2]
    if v <= 1000:
        return bins[3]
    return bins[4]


def _binned(gold, t):
    """(options, gold_label) for count/area; None if the answer won't parse."""
    s = gold.strip()
    if t == "area":
        # answers are "<n>m2"; strip the unit FIRST or its '2' leaks into the
        # digits ("0m2" -> "02" -> 2). Handle m2 and m<superscript-2>.
        s = s.replace("m2", "").replace("m²", "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    v = int(digits)
    if t == "count":
        return COUNT_BINS, _range_bin(v, COUNT_BINS)
    return AREA_BINS, _range_bin(v, AREA_BINS)


def hr_examples(split, img_dir):
    """(img_id, question, options, gold, type); gold index into options."""
    ans = {a["id"]: a for a in json.load(open(A[split]))["answers"]}
    imgs = {im["id"]: im for im in json.load(open(IMJ[split]))["images"]}
    qs = json.load(open(Q[split]))["questions"]
    ex = []
    for q in qs:
        if not q.get("active", True):
            continue
        img_id = q["img_id"]
        if not imgs.get(img_id, {}).get("active", True):
            continue
        if not _img_path(img_dir, img_id).exists():
            continue
        a = ans.get(q["answers_ids"][0])
        if a is None:
            continue
        gold = EB._norm(a["answer"])
        t = q["type"]
        if t in ("presence", "comp"):
            if gold not in ("yes", "no"):
                continue
            opts, gold_lab = ["yes", "no"], gold
        elif t in ("count", "area"):
            binned = _binned(gold, t)
            if binned is None:
                continue
            opts, gold_lab = binned
        else:
            continue
        stem = EB._norm(q["question"].rstrip("?"))
        ex.append((img_id, stem, opts, opts.index(gold_lab), t))
    return ex


def _feats(ids, img_dir, cache_name):
    paths = [_img_path(img_dir, i) for i in ids]
    feats = EB.feature_cache(paths, HR / cache_name)
    return dict(zip(ids, feats))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="both",
                    choices=["zero-shot", "supervised", "both"])
    ap.add_argument("--split", default="test", choices=["test", "test_phili"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=40000,
                    help="cap supervised-adaptation train examples "
                         "(fits the 6 GB GPU; 0 = no cap)")
    ap.add_argument("--out", default=str(ROOT / "data" /
                                         "rsvqa_hr_metrics.json"))
    args = ap.parse_args()

    def _persist(results):
        out = Path(args.out)
        existing = json.loads(out.read_text()) if out.exists() else {}
        existing.update(results)
        out.write_text(json.dumps(existing, indent=2))
        return out

    img_dir = _img_dir()
    print(f"image tiles dir: {img_dir}", flush=True)
    test_ex = hr_examples(args.split, img_dir)
    if args.limit:
        test_ex = test_ex[:args.limit]
    print(f"rsvqa_hr {args.split}: {len(test_ex)} examples "
          f"({Counter(e[4] for e in test_ex)})", flush=True)

    results = {}
    test_ids = sorted({e[0] for e in test_ex})
    test_feats = _feats(test_ids, img_dir, f"feat_rsvqa_hr_{args.split}.npz")

    if args.mode in ("zero-shot", "both"):
        model, emb = EB.load_zero_shot()
        st, c, n, sk = EB.run_eval(model, emb, test_ex, test_feats)
        results["rsvqa_hr_zero_shot"] = {**EB._summary(
            f"RSVQA-HR {args.split} -- ZERO-SHOT (BEN.txt joint scorer)",
            st, c, n, sk, f"{len(test_ids)} images"),
            "benchmark": "RSVQA-HR", "split": args.split}
        _persist(results)   # save now so a supervised OOM can't lose this

    if args.mode in ("supervised", "both"):
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        train_ex = hr_examples("train", img_dir)
        print(f"rsvqa_hr train: {len(train_ex)} examples", flush=True)
        tr_ids = sorted({e[0] for e in train_ex})
        tr_feats = _feats(tr_ids, img_dir, "feat_rsvqa_hr_train.npz")
        tr = [(e[1], e[2], e[0], e[3]) for e in train_ex]
        if args.limit_train and len(tr) > args.limit_train:
            import random
            random.Random(42).shuffle(tr)
            tr = tr[:args.limit_train]
            print(f"  (subsampled train to {len(tr)} examples for GPU memory)",
                  flush=True)
        model, emb = EB.adapt_scorer(tr, tr_feats)
        st, c, n, sk = EB.run_eval(model, emb, test_ex, test_feats)
        results["rsvqa_hr_supervised"] = {**EB._summary(
            f"RSVQA-HR {args.split} -- SUPERVISED (train-split adapted)",
            st, c, n, sk, f"{len(test_ids)} images"),
            "benchmark": "RSVQA-HR", "split": args.split,
            "train_examples_used": len(tr)}
        EB._save(model, HR / "rsvqa_hr_adapted.pkl")

    out = _persist(results)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

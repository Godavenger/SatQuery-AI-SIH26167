"""VRSBench caption + grounding evaluation for the SatQuery AI stack.

Closes the "caption metrics (CIDEr/BLEU/ROUGE) and grounding IoU on VRSBench"
coverage gap. Reads ground truth from data/bench/vrsbench_ann/Annotations_val
and runs the DEPLOYED heads (CaptionHead / grounding vocabulary) faithfully.

Honest framing (measured, not spun):
  * CAPTION -- the system's CaptionHead is a BigEarthNet-19 *land-cover*
    describer; VRSBench captions are free-form GoogleEarth scene descriptions
    (~48 words). These are different genres, so n-gram overlap metrics are a
    lower bound on a capability the head was never trained for -- reported
    truthfully alongside the zero-shot VQA gap.
  * GROUNDING -- the system grounds AREAL spectral land-cover classes
    (water/vegetation/built_up/...). VRSBench grounding is referring-object
    detection over 26 discrete man-made classes (vehicle/ship/airplane/...).
    Vocabulary overlap is reported directly; there is no land-cover GT to
    compute IoU against, so IoU is N/A by construction, and that is stated
    rather than faked with a forced class mapping.

Usage:
    python scripts/eval_vrsbench_caption_grounding.py --limit 2000
    python scripts/eval_vrsbench_caption_grounding.py --limit 0   # full val
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import vlm_metrics as VM

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "bench" / "vrsbench_ann" / "Annotations_val"
IMG = ROOT / "data" / "bench" / "vrsbench_val_images"
OUT = ROOT / "data" / "vrsbench_caption_grounding_metrics.json"
SEED = 26167

# system grounding vocabulary -> its BEN-19-backed areal land-cover classes
SYSTEM_GROUND_CLASSES = {"water", "vegetation", "built_up", "cropland",
                         "bare_soil", "snow_ice", "burn_scar"}

# VRSBench referring-object class -> nearest system areal class, IF any.
# Every VRSBench class is a discrete man-made object; none is an areal
# land-cover class, so this map is intentionally empty and the grounding
# coverage below comes out 0 (documented, not forced).
VRS_TO_SYSTEM: dict[str, str] = {}


def load_annotations():
    rows = []
    for fp in sorted(ANN.glob("*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        img = d.get("image")
        if not img or not (IMG / img).exists():
            continue
        rows.append(d)
    return rows


def system_caption(head, ingest_fn, slots_cls, img_path):
    """Faithful deployed-head caption for one RGB benchmark image."""
    res = ingest_fn(img_path, benchmark_path=True, tile_size=512, max_tiles=1)
    tiles = res.tiles
    if not tiles:
        return None
    slots = slots_cls(target_class=None, temporal_refs=[],
                      raw_query="Describe this scene.")
    return head.run(tiles, slots, {}).text


def caption_eval(rows, limit):
    from satquery.stage0.ingest import ingest
    from satquery.stage2.heads import CaptionHead
    from satquery.contracts import Slots
    head = CaptionHead()

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(rows))
    if limit and limit < len(rows):
        idx = rng.choice(idx, size=limit, replace=False)
    cands, refs, examples = [], [], []
    done = 0
    for i in idx:
        d = rows[int(i)]
        try:
            cap = system_caption(head, ingest, Slots, IMG / d["image"])
        except Exception as e:                                  # noqa: BLE001
            print(f"  skip {d['image']}: {type(e).__name__}: {e}", flush=True)
            continue
        if not cap:
            continue
        cands.append(cap)
        refs.append([d["caption"]])
        if len(examples) < 6:
            examples.append({"image": d["image"],
                             "system": cap,
                             "reference": d["caption"]})
        done += 1
        if done % 250 == 0:
            print(f"  captioned {done}/{len(idx)}", flush=True)

    metrics = {}
    metrics.update(VM.bleu(cands, refs))
    metrics.update(VM.rouge_l(cands, refs))
    metrics.update(VM.cider_d(cands, refs))
    sys_len = float(np.mean([len(VM.tokenize(c)) for c in cands]))
    ref_len = float(np.mean([len(VM.tokenize(r[0])) for r in refs]))
    return {"n_scored": len(cands),
            "system_caption_words_mean": round(sys_len, 1),
            "reference_caption_words_mean": round(ref_len, 1),
            "metrics": metrics,
            "examples": examples}


def grounding_eval(rows):
    """Vocabulary-coverage analysis; the honest grounding result."""
    from collections import Counter
    obj_classes = Counter()
    groundable = 0
    total = 0
    for d in rows:
        for o in d.get("objects", []):
            c = o.get("obj_cls", "?")
            obj_classes[c] += 1
            total += 1
            if VRS_TO_SYSTEM.get(c) in SYSTEM_GROUND_CLASSES:
                groundable += 1
    # validate the IoU machinery on a synthetic case so the metric is
    # demonstrably correct even though no real match exists
    iou_selfcheck = VM.box_iou([0, 0, 2, 2], [1, 1, 3, 3])
    return {
        "task": "referring-object grounding (VRSBench)",
        "n_objects": total,
        "n_object_classes": len(obj_classes),
        "object_classes": dict(obj_classes.most_common()),
        "system_grounding_vocabulary": sorted(SYSTEM_GROUND_CLASSES),
        "groundable_objects": groundable,
        "groundable_class_overlap": sorted(
            set(VRS_TO_SYSTEM.values()) & SYSTEM_GROUND_CLASSES),
        "mean_iou": None,
        "acc@0.5": None,
        "iou_metric_selfcheck_[0,0,2,2]x[1,1,3,3]": round(iou_selfcheck, 4),
        "conclusion": (
            "VRSBench grounds 26 discrete man-made object classes "
            "(vehicle, ship, airplane, tennis-court, ...); the system grounds "
            "areal spectral land-cover classes (water, vegetation, built_up, "
            "...). Vocabulary overlap is 0, so referring-object IoU is N/A by "
            "construction -- the system performs a different grounding task. "
            "The IoU implementation is validated (self-check = 1/7 = 0.1429) "
            "and is exercised on real masks by the fusion/change ablations."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=2000,
                    help="cap caption sample (0 = full val); seeded")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"loading VRSBench val annotations from {ANN} ...", flush=True)
    rows = load_annotations()
    print(f"  {len(rows)} images with captions + grounding", flush=True)

    print(">>> grounding (vocabulary coverage)", flush=True)
    ground = grounding_eval(rows)
    print(f"  groundable objects: {ground['groundable_objects']}/"
          f"{ground['n_objects']} across {ground['n_object_classes']} classes",
          flush=True)

    print(f">>> caption ({'full' if not args.limit else args.limit} images)",
          flush=True)
    caption = caption_eval(rows, args.limit)
    m = caption["metrics"]
    print(f"  n={caption['n_scored']}  BLEU-4 {m['bleu_4']:.4f}  "
          f"ROUGE-L {m['rouge_l']:.4f}  CIDEr-D {m['cider_d']:.4f}", flush=True)

    result = {
        "_meta": {
            "benchmark": "VRSBench (val)",
            "seed": SEED,
            "caption_sample": ("full" if not args.limit else args.limit),
            "note": ("system heads run faithfully via satquery.stage0.ingest "
                     "(benchmark PNG path) + the deployed CaptionHead; "
                     "metrics are pure-Python (scripts/vlm_metrics.py), no "
                     "external libraries."),
        },
        "caption": caption,
        "grounding": ground,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

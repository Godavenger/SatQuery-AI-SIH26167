"""Stage 6 - evaluation harness (section 9).

Because metrics are normalised before combining, a zero anywhere is expensive.
The harness is built to run continuously so the weakest task is always known.
"It works in the demo" is never the strongest available claim.

Runs the five named representative queries from the problem statement plus
paraphrase variants, computes per-task accuracy and router confusion matrix,
and generates a scorecard for the PPT submission.

Usage:
    cd C:\\Users\\Sharanya\\Documents\\Hackathon\\SatQuery_AI_SIH26167
    python scripts/run_eval.py           # all tasks
    python scripts/run_eval.py --quick   # just the 5 PS queries
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "manifest.json"


# --------------------------------------------------------------------------
# representative queries from section 2.2 of the pipeline design
# --------------------------------------------------------------------------

PS_QUERIES = [
    # (query, expected_intent, tile_mode)
    ("Describe the land-cover and major objects visible in this image.",
     "describe", "single"),
    ("Highlight the water body referred to in the query.",
     "ground", "single"),
    ("What changed between these two dates, and where did the change occur?",
     "change", "bitemporal"),
    ("Has the built-up area increased, decreased, or remained unchanged?",
     "change-qa", "bitemporal"),
    ("Use the optical and SAR images together to identify built-up and water-covered regions.",
     "fuse", "fuse"),
]

PARAPHRASE_QUERIES = [
    ("Give me an overview of what this scene shows.", "describe", "single"),
    ("Find the flooded region on this image.", "ground", "single"),
    ("Show me the differences between these two satellite passes.", "change", "bitemporal"),
    ("Did the water extent grow or shrink?", "change-qa", "bitemporal"),
    ("Combine the radar and optical data to map water.", "fuse", "fuse"),
    ("What land cover is dominant here?", "vqa", "single"),
    ("Is there any snow or ice visible?", "vqa", "single"),
    ("How much vegetation is in this tile?", "vqa", "single"),
]


def _get_tiles(site: str, mode: str) -> tuple[list[str], list[str | None]]:
    if not MANIFEST.exists():
        return [], []
    m = json.loads(MANIFEST.read_text())
    e = m.get(site, {})
    s2 = [str(ROOT / x["path"]) for x in e.get("s2", [])
          if (ROOT / x["path"]).exists()]
    s1 = [str(ROOT / x["path"]) for x in e.get("s1", [])
          if (ROOT / x["path"]).exists()]
    ts_s2 = [x["date"] for x in e.get("s2", [])
             if (ROOT / x["path"]).exists()]
    ts_s1 = [x["date"] for x in e.get("s1", [])
             if (ROOT / x["path"]).exists()]

    if mode == "single":
        # prefer the most recent date: for flood sites this is the
        # representative scene (e.g. post-monsoon Assam in November).
        return s2[-1:], ts_s2[-1:]
    if mode == "bitemporal":
        if len(s2) >= 2:
            return s2[:2], ts_s2[:2]
        return s2[:1], ts_s2[:1]     # degrade; gate will abstain
    if mode == "fuse":
        paths, ts = [], []
        if s2:
            paths.append(s2[0]); ts.append(ts_s2[0])
        if s1:
            paths.append(s1[0]); ts.append(ts_s1[0])
        return paths, ts
    return s2[:1], ts_s2[:1]


def run_eval(queries: list, site: str = "assam", verbose: bool = True) -> dict:
    from satquery.pipeline import run_query
    from satquery.stage1.router import load_router

    router_v1 = load_router(None)                   # always rules
    router_v2 = load_router(ROOT / "models" / "router_v2.pkl")

    results = []
    for q, expected, mode in queries:
        paths, ts = _get_tiles(site, mode)
        if not paths:
            if verbose:
                print(f"  [skip] no data for mode={mode} at site={site}")
            continue

        # router accuracy
        pred_v1, conf_v1, _ = router_v1.route(q)
        pred_v2, conf_v2, _ = router_v2.route(q)

        # full pipeline
        try:
            tr = run_query(q, paths, timestamps=ts, use_cache=True)
            ans = tr.answer
            pipeline_intent = ans.intent if ans else "none"
            pipeline_conf = ans.confidence if ans else 0.0
            pipeline_abstained = ans.abstained if ans else True
            pipeline_text = (ans.text[:100] if ans else "")
        except Exception as ex:
            pipeline_intent = "error"
            pipeline_conf = 0.0
            pipeline_abstained = True
            pipeline_text = str(ex)[:80]

        row = {
            "query": q,
            "expected": expected,
            "v1_pred": pred_v1, "v1_conf": round(conf_v1, 3),
            "v2_pred": pred_v2, "v2_conf": round(conf_v2, 3),
            "v1_correct": pred_v1 == expected,
            "v2_correct": pred_v2 == expected,
            "pipeline_intent": pipeline_intent,
            "pipeline_conf": round(pipeline_conf, 3),
            "pipeline_abstained": pipeline_abstained,
            "pipeline_text": pipeline_text,
            "mode": mode,
        }
        results.append(row)
        if verbose:
            mark_v1 = "[OK]" if row["v1_correct"] else "[X]"
            mark_v2 = "[OK]" if row["v2_correct"] else "[X]"
            print(f"  {mark_v1}v1/{mark_v2}v2 expected={expected:10s} "
                  f"v1={pred_v1:10s} v2={pred_v2:10s}  "
                  f"pipe={pipeline_intent:10s} conf={pipeline_conf:.2f} "
                  f"{'ABSTAINED' if pipeline_abstained else 'answered'}")

    if not results:
        return {"error": "No mission imagery found. Install the runtime bundle or fetch data first.",
                "n_queries": 0, "site": site}

    v1_acc = sum(r["v1_correct"] for r in results) / len(results)
    v2_acc = sum(r["v2_correct"] for r in results) / len(results)
    answered = [r for r in results if not r["pipeline_abstained"]]
    pipe_acc = (sum(r["pipeline_intent"] == r["expected"] for r in answered)
                / len(answered)) if answered else 0.0

    # confusion matrix
    intents = ["vqa", "describe", "ground", "change", "change-qa", "fuse"]
    cm = {i: {j: 0 for j in intents} for i in intents}
    for r in results:
        if r["expected"] in cm and r["v2_pred"] in cm:
            cm[r["expected"]][r["v2_pred"]] += 1

    scorecard = {
        "n_queries": len(results),
        "router_v1_accuracy": round(v1_acc, 3),
        "router_v2_accuracy": round(v2_acc, 3),
        "router_improvement": round(v2_acc - v1_acc, 3),
        "pipeline_accuracy_on_answered": round(pipe_acc, 3),
        "metric_definition": "pipeline_accuracy_on_answered measures intent agreement, not answer correctness",
        "pipeline_errors": sum(r["pipeline_intent"] == "error" for r in results),
        "abstention_rate": round(1.0 - len(answered) / max(len(results), 1), 3),
        "site": site,
        "confusion_matrix_v2": cm,
        "rows": results,
    }
    return scorecard


def print_scorecard(sc: dict) -> None:
    print("\n" + "=" * 70)
    print("SATQUERY AI -- EVALUATION SCORECARD")
    print("=" * 70)
    print(f"Site:                    {sc.get('site', 'unknown')}")
    print(f"Queries:                 {sc.get('n_queries', 0)}")
    print(f"Router v1 accuracy:      {sc.get('router_v1_accuracy', 0):.1%}")
    print(f"Router v2 accuracy:      {sc.get('router_v2_accuracy', 0):.1%}  "
          f"(+{sc.get('router_improvement', 0):.1%} improvement)")
    print(f"Pipeline intent match:   {sc.get('pipeline_accuracy_on_answered', 0):.1%}  "
          f"(on answered queries; not answer accuracy)")
    print(f"Abstention rate:         {sc.get('abstention_rate', 0):.1%}")

    cm = sc.get("confusion_matrix_v2", {})
    if any(any(v for v in row.values()) for row in cm.values()):
        print("\nRouter v2 confusion matrix (rows=expected, cols=predicted):")
        ks = [k for k in cm if any(cm[k].values())]
        if ks:
            hdr = f"{'':12s}" + "".join(f"{k:11s}" for k in ks)
            print(hdr)
            for k in ks:
                row_vals = "".join(f"{cm[k].get(j, 0):11d}" for j in ks)
                print(f"{k:12s}{row_vals}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="Run only the five representative PS queries")
    ap.add_argument("--site", default="assam",
                    help="Which manifest site to use for pixel data")
    ap.add_argument("--out", default="data/runs/scorecard.json",
                    help="Where to write the JSON scorecard")
    args = ap.parse_args()

    queries = PS_QUERIES if args.quick else PS_QUERIES + PARAPHRASE_QUERIES

    print(f"Running {len(queries)} queries on site={args.site}...")
    sc = run_eval(queries, site=args.site, verbose=True)
    if sc.get("error"):
        print(sc["error"], file=sys.stderr)
        return 1
    print_scorecard(sc)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sc, indent=2, default=str))
    print(f"\nScorecard saved -> {out}")
    return 1 if sc.get("pipeline_errors") else 0


if __name__ == "__main__":
    sys.exit(main())

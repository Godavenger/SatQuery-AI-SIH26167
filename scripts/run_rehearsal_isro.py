"""Run the ISRO-profile rehearsal end-to-end and report what breaks.

Exercises every intent (vqa / describe / ground / change / fuse) against the
synthetic Cartosat-2S PAN, Cartosat-2S MX and RISAT scenes, and classifies each
failure as INGEST (Stage 0 profile/radiometry) or MODEL (Stage 2+).

Read-only with respect to existing artifacts: writes only
data/rehearsal/rehearsal_results.json.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from satquery.pipeline import run_query  # noqa: E402

REH = ROOT / "data" / "rehearsal"
PAN = str(REH / "cartosat2s_pan_2025-03-14.tif")
MX1 = str(REH / "cartosat2s_mx_2025-03-14.tif")
MX2 = str(REH / "cartosat2s_mx_2025-09-10.tif")
SAR = str(REH / "risat_sar_2025-03-15.tif")

CASES = [
    # label, query, [paths], [profile_ids], [timestamps]
    ("PAN vqa",        "Is there water in this image?",                 [PAN], ["cartosat2s_pan"], ["2025-03-14"]),
    ("PAN describe",   "Describe this scene.",                          [PAN], ["cartosat2s_pan"], ["2025-03-14"]),
    ("MX vqa",         "Is there water in this image?",                 [MX1], ["cartosat2s_mx"],  ["2025-03-14"]),
    ("MX describe",    "Describe this scene.",                          [MX1], ["cartosat2s_mx"],  ["2025-03-14"]),
    ("MX ground",      "Where is the water?",                           [MX1], ["cartosat2s_mx"],  ["2025-03-14"]),
    ("MX change",      "What changed between these two images?",        [MX1, MX2], ["cartosat2s_mx", "cartosat2s_mx"], ["2025-03-14", "2025-09-10"]),
    ("RISAT vqa",      "Is there water in this image?",                 [SAR], ["risat_sar"],      ["2025-03-15"]),
    ("RISAT describe", "Describe this scene.",                          [SAR], ["risat_sar"],      ["2025-03-15"]),
    ("MX+RISAT fuse",  "Fuse the optical and SAR data for this scene.",  [MX1, SAR], ["cartosat2s_mx", "risat_sar"], ["2025-03-14", "2025-03-15"]),

    # profile inference without an explicit hint -- what a bare upload does
    ("PAN inferred",   "Describe this scene.",                          [PAN], [None], [None]),
    ("MX inferred",    "Describe this scene.",                          [MX1], [None], [None]),
    ("RISAT inferred", "Describe this scene.",                          [SAR], [None], [None]),
]


def main() -> None:
    results = []
    for label, query, paths, profs, tss in CASES:
        entry: dict = {"case": label, "query": query, "paths": paths,
                       "profiles": profs}
        try:
            tr = run_query(query, paths, profile_ids=profs, timestamps=tss)
            ing = next((s for s in tr.steps if s.name == "ingest"), None)
            spec = next((s for s in tr.steps if s.name == "specialist"), None)
            gate = next((s for s in tr.steps if s.name == "check_compatibility"), None)
            ans = tr.answer
            entry.update({
                "ok": True,
                "intent": ans.intent if ans else None,
                "abstained": bool(ans and ans.abstained),
                "abstain_reason": ans.abstain_reason if ans else "",
                "confidence": round(float(ans.confidence), 4) if ans else None,
                "answer": (ans.text[:220] if ans else ""),
                "ingest_profile": (ing.detail or {}).get("profile") if ing else None,
                "ingest_warnings": (ing.detail or {}).get("warnings") if ing else None,
                "gate_action": (gate.detail or {}).get("gate_verdict", {}).get("action") if gate else None,
                "tool": spec.tool_name if spec else None,
            })
            print(f"[OK]   {label:16s} intent={entry['intent']} "
                  f"conf={entry['confidence']} prof={entry['ingest_profile']}")
            if entry["abstained"]:
                print(f"       ABSTAIN: {entry['abstain_reason'][:140]}")
            if entry["ingest_warnings"]:
                print(f"       ingest warn: {entry['ingest_warnings']}")
        except Exception as e:
            tb = traceback.format_exc()
            stage = "MODEL" if "stage2" in tb or "backbone" in tb or "heads" in tb else "INGEST"
            entry.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                          "stage_guess": stage,
                          "traceback_tail": tb.strip().splitlines()[-6:]})
            print(f"[FAIL] {label:16s} {stage}: {type(e).__name__}: {e}")
        results.append(entry)

    out = REH / "rehearsal_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    n_ok = sum(1 for r in results if r["ok"])
    print(f"{n_ok}/{len(results)} cases completed without raising")


if __name__ == "__main__":
    main()

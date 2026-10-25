"""Stage 1 - the compatibility gate (section 4.2).

The statement names five properties to verify - number, modality, format,
metadata, compatibility - so the gate is literally five fields with a logged
verdict each, passes included. Recording the passes matters as much as
recording the failures: the trace is scored on completeness, and a silent pass
is indistinguishable from a check that never ran.

Abstention is a feature, not an excuse. Annotations on the ISRO evaluation set
are undisclosed, so a confident wrong answer costs score with no way to detect
it beforehand, whereas a reasoned abstention is measured directly by abstention
precision in Stage 6.
"""

from __future__ import annotations

from typing import Optional

from ..contracts import GateVerdict, Tile
from ..stage0.ingest import PairCheck, check_pair
from .router import class_supported

#: The gate's tolerances. Exposed so the evaluation harness can sweep them.
MIN_FOOTPRINT_IOU = 0.90
MAX_RESIDUAL_SHIFT_PX = 2.0

SINGLE_TILE_INTENTS = frozenset({"vqa", "describe", "ground"})
BITEMPORAL_INTENTS = frozenset({"change", "change-qa"})


def run_gate(intent: str, tiles: list[Tile], query: str = "",
             pair_check: Optional[PairCheck] = None) -> tuple[GateVerdict, Optional[PairCheck]]:
    """Verify the five properties and decide proceed / abstain / reject / degrade."""
    v = GateVerdict()

    # ---- 1. number ------------------------------------------------------
    n = len(tiles)
    if intent in SINGLE_TILE_INTENTS:
        ok = n >= 1
        v.add("number", ok,
              f"{n} tile(s) supplied; {intent} requires at least 1"
              + ("" if ok else " - none supplied"),
              {"n_tiles": n, "required": ">=1"})
    elif intent in BITEMPORAL_INTENTS:
        ok = n == 2
        v.add("number", ok,
              f"{n} tile(s) supplied; {intent} requires exactly 2 "
              f"(a bi-temporal pair)",
              {"n_tiles": n, "required": "==2"})
    elif intent == "fuse":
        ok = n == 2
        v.add("number", ok,
              f"{n} tile(s) supplied; fuse requires exactly 2 "
              f"(one optical, one SAR)",
              {"n_tiles": n, "required": "==2"})
    else:
        v.add("number", False, f"unknown intent '{intent}'", {"n_tiles": n})
        v.action, v.reason = "reject", f"unknown intent '{intent}'"
        return v, pair_check

    if not v.fields[-1].passed:
        v.action = "reject" if intent in SINGLE_TILE_INTENTS else "abstain"
        v.reason = v.fields[-1].detail
        _fill_unrun(v, {"number"})
        return v, pair_check

    # ---- 2. modality ----------------------------------------------------
    mods = [t.modality for t in tiles]
    if intent == "fuse":
        has_opt = any("optical" in m for m in mods)
        has_sar = any("sar" in m for m in mods)
        ok = has_opt and has_sar
        v.add("modality", ok,
              f"modalities {mods}; fusion requires one optical and one SAR tile"
              + ("" if ok else " - inputs are not complementary"),
              {"modalities": mods, "optical": has_opt, "sar": has_sar})
        if not ok:
            # Degrade rather than refuse: a single-modality answer is still
            # useful and the trace says explicitly that fusion did not happen.
            v.action = "degrade"
            v.degraded_to = "describe" if mods[0] == "optical" else "describe"
            v.reason = (f"both inputs are {mods[0]}; degrading to "
                        "single-modality analysis with the loss of complementarity "
                        "flagged and confidence reduced")
    elif intent in BITEMPORAL_INTENTS:
        ok = len(set(mods)) == 1
        v.add("modality", ok, f"change comparison requires matching modalities; got {mods}",
              {"modalities": mods})
        if not ok:
            v.action, v.reason = "abstain", v.fields[-1].detail
            _fill_unrun(v, {"number", "modality"})
            return v, pair_check
    else:
        v.add("modality", True,
              f"modalities {mods} accepted for {intent}",
              {"modalities": mods})

    # ---- 3. format ------------------------------------------------------
    profiles = sorted({t.sensor_id for t in tiles})
    unreadable = [t.id for t in tiles if not t.bands]
    georef = [bool(t.crs) for t in tiles]
    fmt_ok = not unreadable
    v.add("format", fmt_ok,
          f"sensor profile(s) resolved: {profiles}; "
          f"{sum(georef)}/{len(tiles)} georeferenced"
          + (f"; unreadable: {unreadable}" if unreadable else ""),
          {"profiles": profiles, "georeferenced": georef,
           "bands": [sorted(t.bands) for t in tiles]})
    if not fmt_ok:
        v.action, v.reason = "reject", f"unreadable tile(s): {unreadable}"
        _fill_unrun(v, {"number", "modality", "format"})
        return v, pair_check

    # ---- 4. metadata ----------------------------------------------------
    gsds = [round(t.gsd, 3) for t in tiles]
    ts = [t.ts for t in tiles]
    meta_notes = [f"GSD {gsds} m", f"timestamps {ts}"]
    meta_ok = True

    usable = [t.clear_fraction > 0 for t in tiles]
    has_evidence = any(usable) if intent == "fuse" else all(usable)
    if not has_evidence:
        v.add("metadata", False, "no usable pixels remain after cloud/nodata masking",
              {"clear_fractions": [t.clear_fraction for t in tiles]})
        v.action, v.reason = "abstain", v.fields[-1].detail
        _fill_unrun(v, {"number", "modality", "format", "metadata"})
        return v, pair_check

    if intent in BITEMPORAL_INTENTS:
        distinct = bool(ts[0] and ts[1] and ts[0] != ts[1])
        meta_ok = distinct
        meta_notes.append(
            "timestamps distinct" if distinct
            else "timestamps identical or absent - a bi-temporal task needs two dates")

    gsd_spread = max(gsds) / max(min(gsds), 1e-6)
    if gsd_spread > 4.0:
        meta_notes.append(
            f"GSD spread {gsd_spread:.1f}x across inputs - resampled to the "
            "finer grid, recorded because scale mismatch biases area statistics")

    v.add("metadata", meta_ok, "; ".join(meta_notes),
          {"gsd_m": gsds, "timestamps": ts,
           "gsd_spread": round(gsd_spread, 3)})

    if not meta_ok:
        v.action = "abstain"
        v.reason = ("bi-temporal task requires two distinct acquisition dates; "
                    f"got {ts}")
        _fill_unrun(v, {"number", "modality", "format", "metadata"})
        return v, pair_check

    # ---- 5. compatibility -----------------------------------------------
    if intent in BITEMPORAL_INTENTS:
        pc = pair_check or check_pair(tiles[0], tiles[1], "bi_temporal",
                                      MAX_RESIDUAL_SHIFT_PX, MIN_FOOTPRINT_IOU)
        iou_ok = pc.footprint_iou >= MIN_FOOTPRINT_IOU
        shift_ok = pc.residual_magnitude_px <= MAX_RESIDUAL_SHIFT_PX
        ok = iou_ok and shift_ok
        v.add("compatibility", ok,
              (f"footprint IoU {pc.footprint_iou:.3f} "
               f"(threshold {MIN_FOOTPRINT_IOU}); residual alignment shift "
               f"{pc.residual_magnitude_px:.2f} px "
               f"(tolerance {MAX_RESIDUAL_SHIFT_PX} px) measured by phase "
               "correlation"),
              pc.to_dict())
        if not ok:
            v.action = "abstain"
            unmet = []
            if not iou_ok:
                unmet.append(f"footprint IoU {pc.footprint_iou:.3f} below "
                             f"{MIN_FOOTPRINT_IOU}")
            if not shift_ok:
                unmet.append(
                    f"residual misalignment {pc.residual_magnitude_px:.2f} px "
                    f"exceeds the {MAX_RESIDUAL_SHIFT_PX} px tolerance; at "
                    "sub-metre GSD this manufactures change that does not exist")
            v.reason = "; ".join(unmet)
        return v, pc

    if intent == "fuse":
        pc = pair_check or check_pair(tiles[0], tiles[1], "cross_modal",
                                      MAX_RESIDUAL_SHIFT_PX, MIN_FOOTPRINT_IOU)
        ok = pc.ok
        v.add("compatibility", ok, pc.detail, pc.to_dict())
        if not ok and v.action == "proceed":
            v.action = "degrade"
            v.degraded_to = "describe"
            v.reason = pc.detail
        return v, pc

    if intent in ("ground", "vqa"):
        supported, requested, nearest = class_supported(query)
        v.add("compatibility", supported,
              (f"target class '{requested}' is within the supported grounding "
               f"vocabulary" if supported and requested else
               f"'{requested}' is outside the supported grounding vocabulary; "
               f"nearest supported class is '{nearest}'" if requested else
               "no explicit target class named; grounding the dominant "
               "distinguishable class in the scene"),
              {"requested": requested, "supported": supported,
               "nearest_supported": nearest})
        if not supported:
            v.action = "abstain"
            v.reason = (
                f"'{requested}' requires an object detector this build does not "
                f"carry. The nearest supported class is '{nearest}'. Returning a "
                "box anyway would be a confident wrong answer on an evaluation "
                "set whose annotations are undisclosed.")
        return v, pair_check

    v.add("compatibility", True,
          f"single-tile {intent} has no cross-input compatibility constraint",
          {"n_tiles": len(tiles)})
    return v, pair_check


def _fill_unrun(v: GateVerdict, ran: set[str]) -> None:
    """Record the checks that did not run because an earlier one failed.

    Leaving them out would make a short-circuited gate look like a complete
    one; the reviewer should be able to see exactly how far the gate got.
    """
    for name in ("number", "modality", "format", "metadata", "compatibility"):
        if name not in ran:
            v.add(name, False, "not evaluated - an earlier check failed first",
                  {"skipped": True})

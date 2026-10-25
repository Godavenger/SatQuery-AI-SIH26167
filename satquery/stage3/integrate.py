"""Stage 3 - integration, confidence, abstention (section 6).

Three responsibilities:

    1. Calibration  -- temperature-scale raw confidence so 0.8 means 0.8
                       across all tasks. Each task has its own temperature T;
                       vqa and change-qa are fitted on held-out benchmark
                       splits (scripts/fit_calibration.py), the rest are
                       conservative priors -- see DEFAULT_TEMPERATURES.

    2. Cross-model  -- when grounding finds no water polygon but VQA answers
       conflict        "yes, water is present", raise a flag and lower the
                       merged confidence. This is the headline agentic
                       capability the statement claims: it cannot come from a
                       monolithic model.

    3. Abstention   -- below the calibrated threshold, return an explicit
                       insufficient-confidence result with supporting evidence
                       rather than a guess.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..contracts import Answer, Conflict, TaskResult, Tile, TraceRecord, TraceStep, Visual

# Per-task temperature. T > 1 softens an over-confident model; T < 1 sharpens
# a conservative one. Applied as p_cal = sigmoid(logit(p)/T) in ``calibrate``.
#
# FITTED (scripts/fit_calibration.py -> data/calibration_metrics.json,
# held-out 50/50 hold-out on RSVQA-LR + VRSBench + CDVQA):
#   vqa       = 2.11  -- pooled RSVQA-LR + VRSBench, task-adapted scorer
#                        (ECE 0.063 -> 0.020 on the eval half)
#   change-qa = 1.18  -- CDVQA, task-adapted scorer (ECE 0.051 -> 0.029)
# These come from the *task-adapted* scorer -- the mission-corpus deployment
# setting -- where confidence is informative (AUROC 0.83 / 0.77). The
# deployed *zero-shot* scorer's softmax confidence is near-uninformative on
# foreign benchmarks (AUROC ~0.5) and its temperature fit degenerates (pins
# at the search ceiling, i.e. a single scalar can only help by collapsing
# confidence to the base rate) -- which is exactly why the pipeline grounds
# answer confidence in auditable spectral fractions + cross-model
# reconciliation (stage2/heads.py) rather than raw softmax.
#
# PRIORS (not benchmark-fit -- these heads emit hand-designed, bounded
# confidences and no cached held-out (confidence, correct) set exists to fit
# them; kept as conservative softening and labelled honestly):
#   describe, ground, change, fuse
DEFAULT_TEMPERATURES: dict[str, float] = {
    "vqa":       2.11,   # fitted (adapted scorer, RSVQA-LR + VRSBench)
    "describe":  1.05,   # prior
    "ground":    1.20,   # prior
    "change":    1.10,   # prior
    "change-qa": 1.18,   # fitted (adapted scorer, CDVQA)
    "fuse":      1.12,   # prior
}

# Abstention threshold: below this calibrated confidence the answer is
# withheld. A zero anywhere in Stage 6 is expensive, so the default is
# conservative; it can be swept per-task in the evaluation harness.
DEFAULT_ABSTAIN_THRESH: dict[str, float] = {
    "vqa":       0.35,
    "describe":  0.30,
    "ground":    0.38,
    "change":    0.38,
    "change-qa": 0.40,
    "fuse":      0.35,
}


def calibrate(raw_conf: float, task: str,
              temperatures: Optional[dict[str, float]] = None) -> float:
    """Temperature-scale a raw confidence value (section 6.1).

    Applies platt-like softening via temperature T:

        calibrated = sigmoid(logit(raw) / T)

    where logit(p) = log(p / (1-p)).

    The effect is that a model that always says 0.95 gets dragged toward 0.80,
    while a model that under-reports confidence gets lifted.
    """
    T = (temperatures if temperatures is not None else DEFAULT_TEMPERATURES).get(task, 1.0)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("calibration temperature must be finite and positive")
    if not np.isfinite(raw_conf):
        return 0.0
    p = float(np.clip(raw_conf, 1e-6, 1.0 - 1e-6))
    logit = np.log(p / (1.0 - p))
    cal = float(1.0 / (1.0 + np.exp(-logit / T)))
    return round(cal, 4)


def detect_conflicts(results: dict[str, TaskResult],
                     tiles: list[Tile]) -> list[Conflict]:
    """Cross-model consistency check (section 6.2).

    Checks the one inconsistency the statement flags by name: grounding
    finding nothing while VQA asserts presence. Extends to the change / fuse
    pair when both are available.
    """
    conflicts: list[Conflict] = []

    vqa = results.get("vqa")
    ground = results.get("ground")
    if vqa and ground:
        vqa_says_present = ("yes" in vqa.text.lower() or
                            "present" in vqa.text.lower() or
                            "detected" in vqa.text.lower())
        ground_found = ground.visual.mask is not None and \
            (ground.visual.mask > 0).sum() > 0
        if vqa_says_present and not ground_found:
            conflicts.append(Conflict(
                between=("vqa", "ground"),
                detail=("VQA asserts target class is present "
                        "but grounding finds no region above the "
                        "minimum-area threshold"),
                penalty=0.12,
            ))
        elif not vqa_says_present and ground_found:
            conflicts.append(Conflict(
                between=("vqa", "ground"),
                detail=("Grounding found a region but VQA answered "
                        "that the class is absent"),
                penalty=0.08,
            ))

    change = results.get("change")
    fuse = results.get("fuse")
    if change and fuse:
        change_area = 0
        fuse_area = 0
        if change.visual.mask is not None:
            change_area = int((change.visual.mask > 0).sum())
        if fuse.visual.mask is not None:
            fuse_area = int((fuse.visual.mask > 0).sum())
        total = max(1, tiles[0].shape[0] * tiles[0].shape[1])
        if change_area > 0 and fuse_area == 0:
            conflicts.append(Conflict(
                between=("change", "fuse"),
                detail="Change model detected change but fusion mask is empty",
                penalty=0.06,
            ))
        elif change_area / total > 0.5 and fuse_area / total < 0.1:
            conflicts.append(Conflict(
                between=("change", "fuse"),
                detail=("Change mask covers >50% of tile but "
                        "fusion mask covers <10%; results may be inconsistent"),
                penalty=0.05,
            ))

    return conflicts


def merge(
    task_result: TaskResult,
    intent: str,
    plan_params: dict[str, Any],
    tiles: list[Tile],
    trace: TraceRecord,
    extra_results: Optional[dict[str, TaskResult]] = None,
    temperatures: Optional[dict[str, float]] = None,
    abstain_thresholds: Optional[dict[str, float]] = None,
    gate_degraded: bool = False,
) -> Answer:
    """Produce a calibrated, conflict-checked Answer from a TaskResult.

    Gate degradations lower confidence explicitly -- a fusion query answered
    from a single modality reports lower confidence and the trace says why.
    """
    import time
    t0 = time.perf_counter()

    cal = calibrate(task_result.confidence, intent, temperatures)

    # Penalise for gate degradation.
    if gate_degraded:
        cal = round(cal * 0.82, 4)

    # Penalise for GSD being outside the declared range.
    gsd_fit = float(plan_params.get("_gsd_fit", 1.0))
    if gsd_fit < 0.95:
        cal = round(cal * (0.85 + 0.15 * gsd_fit), 4)

    conflicts: list[Conflict] = []
    if extra_results:
        conflicts = detect_conflicts(
            {**extra_results, intent: task_result}, tiles)
        penalty = sum(c.penalty for c in conflicts)
        if penalty > 0:
            cal = round(max(0.1, cal - penalty), 4)

    thresh = (abstain_thresholds or DEFAULT_ABSTAIN_THRESH).get(intent, 0.30)
    abstained = task_result.abstained or cal < thresh

    cal_info = {
        "raw_confidence": round(float(task_result.confidence), 4),
        "temperature": (temperatures or DEFAULT_TEMPERATURES).get(intent, 1.0),
        "calibrated": cal,
        "gate_degraded": gate_degraded,
        "gsd_fit": round(gsd_fit, 4),
        "conflict_penalty": round(sum(c.penalty for c in conflicts), 4),
        "abstain_threshold": thresh,
        "abstained": abstained,
        "specialist_abstained": task_result.abstained,
        "task": intent,
    }

    if abstained:
        ans = Answer(
            text=(task_result.text if task_result.abstained else
                  "Confidence after calibration is "
                  f"{cal:.0%}, below the {thresh:.0%} threshold for {intent}. "
                  "The supporting evidence is recorded in this trace but no "
                  "answer is emitted. A reasoned abstention is preferable to a "
                  "confident wrong answer on an evaluation set whose annotations "
                  "are undisclosed."),
            confidence=cal,
            intent=intent,
            output_kind=task_result.output_kind,
            abstained=True,
            abstain_reason=(task_result.text if task_result.abstained else
                            f"calibrated confidence {cal:.0%} < threshold "
                            f"{thresh:.0%}"),
            visual=task_result.visual,
            raw=task_result.raw,
            conflicts=conflicts,
            calibration=cal_info,
            contributing=[task_result.model_id],
        )
    else:
        ans = Answer(
            text=task_result.text,
            confidence=cal,
            intent=intent,
            output_kind=task_result.output_kind,
            abstained=False,
            visual=task_result.visual,
            raw=task_result.raw,
            conflicts=conflicts,
            calibration=cal_info,
            contributing=[task_result.model_id],
        )

    dt = (time.perf_counter() - t0) * 1000
    trace.add(TraceStep(
        seq=0, stage=3, name="integrate",
        intent=intent,
        tool_name=task_result.model_id,
        tool_version=task_result.model_version,
        confidence=cal,
        output_kind=task_result.output_kind,
        output_summary=(
            f"raw={task_result.confidence:.3f} -> calibrated={cal:.3f}; "
            f"{'ABSTAINED' if abstained else 'answered'}; "
            f"{len(conflicts)} conflict(s)"),
        latency_ms=dt,
        detail=cal_info,
    ))
    return ans

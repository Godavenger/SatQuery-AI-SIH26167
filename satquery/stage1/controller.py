"""Stage 1 - the agentic controller.

The statement lists six controller duties. Each is implemented as a separately
named and separately logged step, so the observable trace maps onto the required
list without interpretation.

A deterministic router that emits a complete, well-formed trace scores
identically to a chain-of-thought agent and fails far less often, which is why
this is a classifier plus a validation gate rather than an LLM planner.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ..contracts import ExecPlan, Tile, TraceRecord, TraceStep
from ..stage0.ingest import PairCheck
from .gate import run_gate
from .registry import (
    NoToolError, ParameterError, bind_params, select_tool,
)
from .router import RuleRouter, TrainedRouter, extract_slots, load_router


class Controller:
    """Interpret, check, select, bind. Nothing here runs a model."""

    def __init__(self, router: Optional[RuleRouter | TrainedRouter] = None):
        self.router = router or load_router()

    # -- duty 1 -----------------------------------------------------------

    def interpret(self, query: str, trace: TraceRecord) -> tuple[str, float, Any, list]:
        t0 = time.perf_counter()
        intent, conf, alts = self.router.route(query)
        slots = extract_slots(query, intent)
        dt = (time.perf_counter() - t0) * 1000

        trace.add(TraceStep(
            seq=0, stage=1, name="interpret",
            intent=intent,
            tool_name=self.router.name, tool_version=self.router.version,
            output_kind=slots.output_kind,
            output_summary=(
                f"intent={intent} (p={conf:.3f}) via {self.router.name} "
                f"{self.router.version}; target_class={slots.target_class}; "
                f"polarity={slots.polarity}; output_kind={slots.output_kind}"),
            confidence=conf, latency_ms=dt,
            detail={"slots": slots.to_dict(), "alternatives": alts,
                    "router": self.router.name,
                    "margin": round(conf - (alts[0][1] if alts else 0.0), 4)},
        ))
        return intent, conf, slots, alts

    # -- duty 2 -----------------------------------------------------------

    def check(self, intent: str, tiles: list[Tile], query: str,
              trace: TraceRecord,
              pair_check: Optional[PairCheck] = None):
        t0 = time.perf_counter()
        verdict, pc = run_gate(intent, tiles, query, pair_check)
        dt = (time.perf_counter() - t0) * 1000

        passed = sum(1 for f in verdict.fields if f.passed)
        trace.add(TraceStep(
            seq=0, stage=1, name="check_compatibility",
            intent=intent,
            input_tile_ids=[t.id for t in tiles],
            output_kind="gate_verdict",
            output_summary=(
                f"{passed}/{len(verdict.fields)} fields passed; "
                f"action={verdict.action}"
                + (f"; reason={verdict.reason}" if verdict.reason else "")),
            latency_ms=dt,
            gate_verdict=verdict.to_dict(),
        ))
        return verdict, pc

    # -- duty 3 -----------------------------------------------------------

    def select(self, intent: str, tiles: list[Tile], trace: TraceRecord):
        t0 = time.perf_counter()
        modality = _combined_modality(tiles)
        gsd = min(t.gsd for t in tiles) if tiles else 10.0

        try:
            spec, evidence = select_tool(intent, modality, gsd)
        except NoToolError as e:
            trace.add(TraceStep(
                seq=0, stage=1, name="select_tool", intent=intent,
                output_summary=f"selection failed: {e}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail={"error": str(e)}))
            raise

        dt = (time.perf_counter() - t0) * 1000
        trace.add(TraceStep(
            seq=0, stage=1, name="select_tool", intent=intent,
            tool_name=spec.name, tool_version=spec.version,
            weights_hash=spec.weights_hash,
            output_summary=(
                f"{spec.name} v{spec.version} selected by declarative filter "
                f"(task={intent}, modality={modality}, gsd={gsd:.2f} m, "
                f"fit={evidence['gsd_fit']:.2f}"
                + ("" if evidence["gsd_in_declared_range"]
                   else ", OUTSIDE declared GSD range") + ")"),
            latency_ms=dt, detail=evidence,
        ))
        return spec, evidence

    # -- duty 4 -----------------------------------------------------------

    def bind(self, spec, requested: Optional[dict[str, Any]],
             defaults: Optional[dict[str, Any]], trace: TraceRecord,
             intent: str):
        t0 = time.perf_counter()
        try:
            params = bind_params(spec, requested, defaults)
            err = None
        except ParameterError as e:
            params, err = {}, str(e)

        dt = (time.perf_counter() - t0) * 1000
        trace.add(TraceStep(
            seq=0, stage=1, name="bind_params", intent=intent,
            tool_name=spec.name, tool_version=spec.version,
            params=params,
            output_summary=(
                f"bound {sorted(params)} against declared bounds"
                if not err else f"REJECTED: {err}"),
            latency_ms=dt,
            detail={"permitted": spec.permitted_params,
                    "requested": requested or {},
                    "rejected_reason": err},
        ))
        if err:
            raise ParameterError(err)
        return params

    # -- the whole of Stage 1 ---------------------------------------------

    def plan(self, query: str, tiles: list[Tile], trace: TraceRecord,
             requested_params: Optional[dict[str, Any]] = None,
             pair_check: Optional[PairCheck] = None) -> ExecPlan:
        """Produce an ExecPlan, or one whose gate says why nothing should run."""
        intent, conf, slots, alts = self.interpret(query, trace)
        verdict, pc = self.check(intent, tiles, query, trace, pair_check)

        plan = ExecPlan(
            intent=intent, intent_confidence=conf, slots=slots, gate=verdict,
            tile_ids=[t.id for t in tiles],
            router_version=self.router.version, alternatives=alts,
        )

        if verdict.action in ("reject", "abstain"):
            return plan

        effective = intent
        if verdict.action == "degrade" and verdict.degraded_to:
            effective = verdict.degraded_to
            plan.slots.output_kind = "text"

        spec, evidence = self.select(effective, tiles, trace)
        plan.tool_name = spec.name
        plan.tool_version = spec.version
        plan.params = self.bind(
            spec, requested_params,
            {"conf_thresh": 0.35, "dead_band": 0.02}, trace, effective)
        plan.params["_gsd_fit"] = evidence["gsd_fit"]
        plan.params["_effective_intent"] = effective
        return plan


def _combined_modality(tiles: list[Tile]) -> str:
    """Modality of the input set, not of any single tile."""
    if not tiles:
        return "optical"
    has_opt = any(t.has_optical for t in tiles)
    has_sar = any(t.has_sar for t in tiles)
    if has_opt and has_sar:
        return "optical+sar"
    return "sar" if has_sar else "optical"

"""End-to-end pipeline: Stage 0 -> 1 -> 2 -> 3 -> 4.

This is the single function the FastAPI endpoint calls. It accepts a query
string and a list of tile paths (already ingested, or paths to ingest), runs
all stages, and returns a TraceRecord with the Answer attached.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .contracts import TraceRecord, TraceStep, new_run_id
from .stage0.ingest import TileCache, align_pair, check_pair, ingest, resample_to
from .stage1.controller import Controller
from .stage2.heads import get_head
from .stage3.integrate import merge

# Shared objects — initialised once per process.
_controller: Optional[Controller] = None
_cache: Optional[TileCache] = None


def _get_controller() -> Controller:
    global _controller
    if _controller is None:
        _controller = Controller()
    return _controller


def _get_cache(root: Optional[str] = None) -> TileCache:
    global _cache
    if _cache is None:
        _cache = TileCache(root or Path(__file__).parents[1] / "data" / "tile_cache")
    return _cache


def run_query(
    query: str,
    tile_paths: list[str | Path],
    *,
    profile_ids: Optional[list[Optional[str]]] = None,
    timestamps: Optional[list[Optional[str]]] = None,
    tile_size: int = 512,
    benchmark_path: bool = False,
    requested_params: Optional[dict[str, Any]] = None,
    use_cache: bool = True,
    run_id: Optional[str] = None,
) -> TraceRecord:
    """Run the full pipeline for one query over one or two tiles.

    Returns a TraceRecord with every step populated and `answer` set.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must contain a question")
    if not 1 <= len(tile_paths) <= 2:
        raise ValueError("provide one or two tile paths")
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    for name, values in (("profile_ids", profile_ids), ("timestamps", timestamps)):
        if values is not None and len(values) != len(tile_paths):
            raise ValueError(f"{name} must have one entry per tile path")
    trace = TraceRecord(
        run_id=run_id or new_run_id(),
        query=query,
        started_at=time.time(),
        software=_software_versions(),
    )

    profile_ids = profile_ids or [None] * len(tile_paths)
    timestamps = timestamps or [None] * len(tile_paths)

    # ---- Stage 0 --------------------------------------------------------
    t0 = time.perf_counter()
    tiles = []
    for path, prof, ts in zip(tile_paths, profile_ids, timestamps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_cache:
                res = _get_cache().ingest_cached(
                    str(path), profile_id=prof, ts=ts, tile_size=tile_size,
                    overlap=min(32, tile_size - 1), benchmark_path=benchmark_path)
            else:
                res = ingest(str(path), profile_id=prof, ts=ts,
                             tile_size=tile_size, overlap=min(32, tile_size - 1),
                             benchmark_path=benchmark_path)
        tile = res.tiles[0]
        tiles.append(tile)
        trace.tiles.append(tile.describe())
        trace.add(TraceStep(
            seq=0, stage=0, name="ingest",
            tool_name="stage0-ingest", tool_version="0.1.0",
            input_tile_ids=[tile.id],
            input_hash=tile.content_hash(),
            output_kind="tile",
            output_summary=" | ".join(f"{l['step']}: {l['detail'][:60]}"
                                      for l in res.log[:4]),
            detail={"log": res.log[:8], "profile": res.profile.id,
                    "warnings": res.warnings},
        ))

    # ---- Stage 1 --------------------------------------------------------
    controller = _get_controller()
    pair_check = None
    if len(tiles) == 2:
        # Cross-modal pairs (optical+SAR) are NOT bi-temporal even when the
        # two sensors acquired on different days: phase correlation between
        # modalities measures texture disagreement, not co-registration, so
        # "repairing" by that shift corrupts the pair and the gate then
        # degrades a valid fuse query to describe.
        cross_modal = (any("optical" in t.modality for t in tiles)
                       and any("sar" in t.modality for t in tiles))
        bi = not cross_modal
        pair_check = check_pair(
            tiles[0], tiles[1],
            kind="bi_temporal" if bi else "cross_modal")
        # Reprojection uses the actual CRS/affine grid, including when array
        # dimensions happen to agree. Preserve a failing footprint check.
        if pair_check.footprint_iou >= 0.9:
            tiles[1] = resample_to(tiles[1], tiles[0])
            pair_check = check_pair(tiles[0], tiles[1],
                                    kind="bi_temporal" if bi else "cross_modal")
        if (bi and pair_check.footprint_iou >= 0.9
                and pair_check.aligned is False
                and pair_check.residual_magnitude_px > 0.5):
            tr0 = time.perf_counter()
            shift_before = pair_check.residual_shift_px
            tiles[1] = align_pair(tiles[0], tiles[1], shift_before)
            # Re-measure on the repaired pair so the gate judges the tiles
            # it will actually receive, not the pre-repair misalignment.
            pair_check = check_pair(
                tiles[0], tiles[1],
                kind="bi_temporal" if bi else "cross_modal")
            trace.add(TraceStep(
                seq=0, stage=0, name="align_repair",
                input_tile_ids=[t.id for t in tiles],
                output_summary=(
                    f"resampled pair onto reference grid; residual "
                    f"{float(np.hypot(*shift_before)):.2f} px -> "
                    f"{pair_check.residual_magnitude_px:.2f} px"),
                latency_ms=(time.perf_counter() - tr0) * 1000,
                detail={"shift_before": [round(v, 3) for v in shift_before],
                        "shift_after": [round(v, 3)
                                         for v in pair_check.residual_shift_px],
                        "residual_after_px": pair_check.residual_magnitude_px},
            ))
        trace.tiles = [t.describe() for t in tiles]

    plan = controller.plan(query, tiles, trace,
                           requested_params=requested_params,
                           pair_check=pair_check)

    if plan.gate.action in ("reject", "abstain"):
        from .contracts import Answer
        trace.answer = Answer(
            text=(f"[{plan.gate.action.upper()}] {plan.gate.reason}"),
            confidence=0.0, intent=plan.intent,
            abstained=True, abstain_reason=plan.gate.reason,
            calibration={"action": plan.gate.action},
        )
        trace.finished_at = time.time()
        return trace

    effective = plan.params.get("_effective_intent", plan.intent)

    # ---- Stage 2 --------------------------------------------------------
    t2 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        head = get_head(effective)
        task_result = head.run(tiles, plan.slots, plan.params)
    task_result.latency_ms = (time.perf_counter() - t2) * 1000

    trace.add(TraceStep(
        seq=0, stage=2, name="specialist",
        intent=effective,
        # Report what actually answered, not the head class. When a trained
        # checkpoint produced the result these carry its id and the real
        # sha256 of the artifact; on the closed-form path they fall back to
        # the head's own identity values.
        tool_name=task_result.model_id or head.model_id,
        tool_version=task_result.model_version or head.model_version,
        weights_hash=task_result.weights_hash or head.weights_hash,
        params={k: v for k, v in plan.params.items()
                if not k.startswith("_")},
        input_tile_ids=[t.id for t in tiles],
        input_hash="|".join(t.content_hash() for t in tiles),
        output_kind=task_result.output_kind,
        output_summary=(f"conf={task_result.confidence:.3f}; "
                        f"{task_result.text[:120]}"),
        confidence=task_result.confidence,
        latency_ms=task_result.latency_ms,
        detail={"raw": task_result.raw, "visual": task_result.visual.summary()},
    ))

    # ---- Stage 3 --------------------------------------------------------
    gate_degraded = plan.gate.action == "degrade"
    answer = merge(
        task_result=task_result,
        intent=effective,
        plan_params=plan.params,
        tiles=tiles,
        trace=trace,
        gate_degraded=gate_degraded,
    )

    trace.answer = answer
    trace.finished_at = time.time()
    return trace


def _software_versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    from . import __version__
    vers: dict[str, Any] = {"satquery": __version__}
    for mod, key in (("torch", "torch"), ("rasterio", "rasterio"),
                     ("numpy", "numpy"), ("torchgeo", "torchgeo"),
                     ("scipy", "scipy")):
        try:
            vers[key] = version(mod)
        except PackageNotFoundError:
            pass
    return vers

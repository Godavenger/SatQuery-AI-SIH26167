"""Typed contracts published at every stage boundary.

These are the integration surface described in section 2 of the pipeline design:
a stub and a trained checkpoint are interchangeable as long as the contract
holds, and no stage reaches around another to inspect raw inputs.

Nothing in this module imports a model or touches the filesystem, so it stays
importable in a bare environment (the frontend contract tests rely on that).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

import numpy as np

# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------

#: The six supported intents (section 2.2). The hard boundary is change vs
#: change-qa: both are bi-temporal, so lexical routing conflates them, yet one
#: wants a description plus a map and the other a three-way categorical.
Intent = Literal["vqa", "describe", "ground", "change", "change-qa", "fuse"]

INTENTS: tuple[str, ...] = ("vqa", "describe", "ground", "change", "change-qa", "fuse")

#: What a task emits. The controller binds this so the trace records the
#: expected shape before the specialist runs, which is what makes a wrong
#: output kind detectable rather than merely wrong.
OutputKind = Literal["text", "box", "mask", "categorical", "number"]

#: Semantic band slots. Physical band indices are mapped onto these by the
#: sensor profile so that nothing downstream branches on dataset or platform.
BAND_SLOTS: tuple[str, ...] = (
    "blue", "green", "red", "nir", "swir1", "swir2", "pan",  # optical
    "VV", "VH", "HH", "HV",                                   # SAR
)

OPTICAL_SLOTS = frozenset({"blue", "green", "red", "nir", "swir1", "swir2", "pan"})
SAR_SLOTS = frozenset({"VV", "VH", "HH", "HV"})

Modality = Literal["optical", "sar", "optical+sar"]


# --------------------------------------------------------------------------
# Stage 0 output — the canonical tile
# --------------------------------------------------------------------------


@dataclass
class SceneMeta:
    """Provenance for the scene a tile was cut from."""

    parent_scene: str
    tile_grid: tuple[int, int]          # (row, col) within the parent scene
    source_path: str
    profile_id: str
    ingest_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tile:
    """The canonical tile contract (section 3.3).

    ``gsd`` is a first-class field rather than metadata because Stage 2
    conditions on it, and ``transform`` is what makes grounding geo-anchored
    instead of pixel-anchored.

    Absent band slots are simply missing from ``bands``. They are never
    zero-filled: a Cartosat panchromatic scene zero-filled into RGB reads as a
    very dark image rather than as an error, and that failure is silent.
    """

    id: str
    sensor_id: str
    bands: dict[str, np.ndarray]
    gsd: float                              # metres per pixel
    crs: Optional[str] = None
    transform: Optional[tuple[float, ...]] = None   # affine (a, b, c, d, e, f)
    bounds: Optional[tuple[float, float, float, float]] = None  # W,S,E,N
    ts: Optional[str] = None                # ISO-8601 acquisition timestamp
    qa_mask: Optional[np.ndarray] = None    # True where cloud/shadow/nodata
    meta: Optional[SceneMeta] = None

    # -- shape helpers ----------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        for arr in self.bands.values():
            return arr.shape[:2]
        return (0, 0)

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    # -- modality ---------------------------------------------------------

    @property
    def has_optical(self) -> bool:
        return any(s in OPTICAL_SLOTS for s in self.bands)

    @property
    def has_sar(self) -> bool:
        return any(s in SAR_SLOTS for s in self.bands)

    @property
    def modality(self) -> str:
        if self.has_optical and self.has_sar:
            return "optical+sar"
        return "sar" if self.has_sar else "optical"

    def band(self, slot: str) -> Optional[np.ndarray]:
        """Return a band by semantic slot, or None if the sensor lacks it."""
        return self.bands.get(slot)

    def require(self, *slots: str) -> list[np.ndarray]:
        """Fetch slots, raising a typed error naming the first missing one."""
        out = []
        for s in slots:
            arr = self.bands.get(s)
            if arr is None:
                raise MissingBandError(
                    f"tile {self.id} (sensor {self.sensor_id}) has no '{s}' band; "
                    f"available: {sorted(self.bands)}"
                )
            out.append(arr)
        return out

    @property
    def valid_mask(self) -> np.ndarray:
        """True where the pixel is usable (not cloud/shadow/nodata)."""
        if self.qa_mask is None:
            return np.ones(self.shape, dtype=bool)
        return ~self.qa_mask.astype(bool)

    @property
    def clear_fraction(self) -> float:
        """Share of the tile not flagged by the quality mask."""
        h, w = self.shape
        if h == 0 or w == 0:
            return 0.0
        return float(self.valid_mask.mean())

    # -- geo --------------------------------------------------------------

    def pixel_to_geo(self, col: float, row: float) -> tuple[float, float]:
        """Map a pixel coordinate to (lon, lat) through the tile transform."""
        if not self.transform:
            return (float(col), float(row))
        a, b, c, d, e, f = self.transform[:6]
        x, y = a * col + b * row + c, d * col + e * row + f
        if self.crs:
            from rasterio.warp import transform
            xs, ys = transform(self.crs, "EPSG:4326", [x], [y])
            return float(xs[0]), float(ys[0])
        return float(x), float(y)

    def box_to_geo(self, x: float, y: float, w: float, h: float) -> dict[str, float]:
        """Convert a pixel box to a geographic box (section 5.3, grounding)."""
        corners = [self.pixel_to_geo(cx, cy) for cx, cy in
                   ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
        lons, lats = zip(*corners)
        return {
            "west": min(lons), "east": max(lons),
            "south": min(lats), "north": max(lats),
        }

    # -- identity ---------------------------------------------------------

    def content_hash(self) -> str:
        """Stable digest of pixel content. Half of the reproducibility pair
        (``input_hash`` + ``weights_hash``) an ISRO/SAC reviewer will ask for."""
        h = hashlib.sha256()
        h.update(self.sensor_id.encode())
        h.update(f"{self.gsd:.6f}".encode())
        h.update(str((self.crs, self.transform, self.ts)).encode())
        for slot in sorted(self.bands):
            h.update(slot.encode())
            h.update(str((self.bands[slot].shape, self.bands[slot].dtype)).encode())
            h.update(np.ascontiguousarray(self.bands[slot]).tobytes())
        h.update(np.ascontiguousarray(self.valid_mask).tobytes())
        return h.hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary for the trace. Never contains pixel data."""
        return {
            "id": self.id,
            "sensor_id": self.sensor_id,
            "bands": sorted(self.bands),
            "gsd_m": round(self.gsd, 4),
            "shape": list(self.shape),
            "modality": self.modality,
            "ts": self.ts,
            "crs": self.crs,
            "clear_fraction": round(self.clear_fraction, 4),
        }


class MissingBandError(KeyError):
    """A model asked for a semantic band this sensor does not carry."""


# --------------------------------------------------------------------------
# Stage 1 output — the execution plan
# --------------------------------------------------------------------------


@dataclass
class GateField:
    """One of the five properties the statement names for compatibility
    checking. Passes are recorded as explicitly as failures — the trace is
    scored on completeness, so a silent pass is a lost mark."""

    name: str           # number | modality | format | metadata | compatibility
    passed: bool
    detail: str
    measured: Optional[dict[str, Any]] = None


@dataclass
class GateVerdict:
    """Result of the five-field compatibility gate (section 4.2)."""

    fields: list[GateField] = field(default_factory=list)
    action: Literal["proceed", "abstain", "reject", "degrade"] = "proceed"
    reason: str = ""
    degraded_to: Optional[str] = None

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.fields)

    def add(self, name: str, passed: bool, detail: str,
            measured: Optional[dict] = None) -> "GateVerdict":
        self.fields.append(GateField(name, passed, detail, measured))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "degraded_to": self.degraded_to,
            "fields": [asdict(f) for f in self.fields],
            "all_passed": self.passed,
        }


@dataclass
class Slots:
    """Structured query slots extracted alongside the intent (section 4.1)."""

    target_class: Optional[str] = None
    temporal_refs: list[str] = field(default_factory=list)
    polarity: Optional[Literal["increased", "decreased", "unchanged"]] = None
    modality_hint: Optional[str] = None
    output_kind: OutputKind = "text"
    count_query: bool = False
    raw_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecPlan:
    """What Stage 1 hands to Stage 2."""

    intent: str
    intent_confidence: float
    slots: Slots
    gate: GateVerdict
    tool_name: Optional[str] = None
    tool_version: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    tile_ids: list[str] = field(default_factory=list)
    router_version: str = "v1"
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "router_version": self.router_version,
            "alternatives": [[a, round(b, 4)] for a, b in self.alternatives],
            "slots": self.slots.to_dict(),
            "gate": self.gate.to_dict(),
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "params": self.params,
            "tile_ids": self.tile_ids,
        }


# --------------------------------------------------------------------------
# Stage 2 output — one uniform result for every specialist
# --------------------------------------------------------------------------


@dataclass
class Visual:
    """Geo-referenced visual evidence. ``mask`` stays as an array in memory and
    is rendered to PNG only by Stage 4, so the trace holds no pixel payload."""

    kind: Literal["box", "mask", "overlay", "none"] = "none"
    boxes: list[dict[str, Any]] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    mask_classes: dict[str, int] = field(default_factory=dict)
    legend: list[dict[str, Any]] = field(default_factory=list)
    attribution: Optional[np.ndarray] = None
    attribution_method: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_boxes": len(self.boxes),
            "boxes": self.boxes[:12],
            "has_mask": self.mask is not None,
            "mask_classes": self.mask_classes,
            "legend": self.legend,
            "has_attribution": self.attribution is not None,
            "attribution_method": self.attribution_method,
        }


@dataclass
class TaskResult:
    """The uniform specialist return value (section 5).

    ``raw`` carries the numbers the text was derived from, so prose stays
    auditable: a reviewer can check the sentence against the measurement.
    """

    text: str = ""
    visual: Visual = field(default_factory=Visual)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    model_version: str = ""
    weights_hash: str = ""
    output_kind: OutputKind = "text"
    abstained: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "visual": self.visual.summary(),
            "confidence": round(float(self.confidence), 4),
            "raw": _jsonable(self.raw),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "weights_hash": self.weights_hash,
            "output_kind": self.output_kind,
            "abstained": self.abstained,
            "latency_ms": round(self.latency_ms, 2),
        }


# --------------------------------------------------------------------------
# Stage 1 registry — declarative tool specs
# --------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Declarative capability record (section 4.3).

    Selection is a filter over these specs, never a hardcoded branch, and
    ``gsd_range`` is what makes the Cartosat resolution gap visible in the
    trace instead of silent.
    """

    name: str
    version: str
    tasks: list[str]
    modalities: list[str]
    gsd_range: tuple[float, float]
    permitted_params: dict[str, Any] = field(default_factory=dict)
    weights_hash: str = "uninitialised"
    device: str = "cpu"
    load_cost_mb: int = 0
    adapted: bool = False           # remote-sensing adapted (the PS clause)
    notes: str = ""

    def supports(self, task: str, modality: str) -> bool:
        if task not in self.tasks:
            return False
        if modality == "optical+sar":
            return "optical+sar" in self.modalities
        return modality in self.modalities or "optical+sar" in self.modalities

    def gsd_fit(self, gsd: float) -> float:
        """0..1 score for how well this tool matches the input resolution.

        Outside the declared range the score decays instead of hard-failing, so
        the controller can still answer at 0.65 m with a 10 m-trained model and
        the trace shows that it did, along with how far out of range it was.
        """
        lo, hi = self.gsd_range
        if lo <= gsd <= hi:
            return 1.0
        edge = lo if gsd < lo else hi
        octaves = abs(np.log2(max(gsd, 1e-6) / max(edge, 1e-6)))
        return float(max(0.0, 1.0 - 0.18 * octaves))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gsd_range"] = list(self.gsd_range)
        return d


# --------------------------------------------------------------------------
# Stage 3 output — the integrated answer
# --------------------------------------------------------------------------


@dataclass
class Conflict:
    """A disagreement between two specialists over the same tile.

    This is the one behaviour an orchestrated system has that a monolithic
    model cannot, so it is treated as a headline capability (section 6.2).
    """

    between: tuple[str, str]
    detail: str
    penalty: float


@dataclass
class Answer:
    """What Stage 3 publishes and Stage 4 records."""

    text: str
    confidence: float
    intent: str
    output_kind: OutputKind = "text"
    abstained: bool = False
    abstain_reason: str = ""
    visual: Visual = field(default_factory=Visual)
    raw: dict[str, Any] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    contributing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "intent": self.intent,
            "output_kind": self.output_kind,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "visual": self.visual.summary(),
            "raw": _jsonable(self.raw),
            "conflicts": [
                {"between": list(c.between), "detail": c.detail,
                 "penalty": round(c.penalty, 4)} for c in self.conflicts
            ],
            "calibration": _jsonable(self.calibration),
            "contributing": self.contributing,
        }


# --------------------------------------------------------------------------
# Stage 4 output — the auditable trace
# --------------------------------------------------------------------------


@dataclass
class TraceStep:
    """One observable step (section 7).

    The statement scores the observable execution trace and explicitly does not
    evaluate internal reasoning text, so none is stored: the record stays
    compact and every field in it is one the reviewer asked for.
    """

    seq: int
    stage: int                      # 0..6, mirrors the controller duties
    name: str
    intent: Optional[str] = None
    tool_name: Optional[str] = None
    tool_version: Optional[str] = None
    weights_hash: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    input_tile_ids: list[str] = field(default_factory=list)
    input_hash: Optional[str] = None
    output_kind: Optional[str] = None
    output_summary: str = ""
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    gate_verdict: Optional[dict[str, Any]] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "seq": self.seq,
            "stage": self.stage,
            "name": self.name,
            "intent": self.intent,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "weights_hash": self.weights_hash,
            "params": _jsonable(self.params),
            "input_tile_ids": self.input_tile_ids,
            "input_hash": self.input_hash,
            "output_kind": self.output_kind,
            "output_summary": self.output_summary,
            "confidence": None if self.confidence is None else round(float(self.confidence), 4),
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.gate_verdict is not None:
            d["gate_verdict"] = self.gate_verdict
        if self.detail:
            d["detail"] = _jsonable(self.detail)
        return d


@dataclass
class TraceRecord:
    """The single source of truth behind the live view, the JSON view and the
    PDF report. One source means the report can never disagree with the
    screen."""

    run_id: str
    query: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    answer: Optional[Answer] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    tiles: list[dict[str, Any]] = field(default_factory=list)
    software: dict[str, Any] = field(default_factory=dict)

    def add(self, step: TraceStep) -> TraceStep:
        step.seq = len(self.steps)
        self.steps.append(step)
        return step

    @property
    def total_latency_ms(self) -> float:
        if self.finished_at is None:
            return (time.time() - self.started_at) * 1000.0
        return (self.finished_at - self.started_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "tiles": self.tiles,
            "software": self.software,
            "steps": [s.to_dict() for s in self.steps],
            "answer": self.answer.to_dict() if self.answer else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars and arrays into JSON-safe values.

    Arrays are summarised rather than serialised: the trace is meant to be
    read, and a 512x512 mask inlined into JSON makes it unreadable.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size <= 24:
            return _jsonable(obj.tolist())
        return {"__array__": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def new_run_id() -> str:
    """Short, sortable, collision-resistant run identifier."""
    return f"run_{int(time.time() * 1000):013d}_{hashlib.sha1(str(time.perf_counter()).encode()).hexdigest()[:6]}"

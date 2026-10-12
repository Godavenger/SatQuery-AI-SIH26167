"""Stage 1 - tool registry, declarative selection, parameter binding.

Selection is a filter over specs, never a hardcoded branch (section 4.3). That
matters for two reasons: adding a specialist is a registry entry rather than an
``if``, and ``gsd_range`` makes the Cartosat resolution gap visible in the
trace instead of silent.

Parameter binding is enforced rather than documented (section 4.4). The
statement requires the system to "configure only permitted task parameters", so
the binder writes only keys declared in ``permitted_params``, range-checks every
value, and raises on anything else. One mechanism, two properties: security and
compliance.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts import ToolSpec

# --------------------------------------------------------------------------
# the shipped registry
# --------------------------------------------------------------------------
#
# gsd_range reflects what the model was actually trained to handle: the shared
# backbone is adapted on BigEarthNet at 10 m but with multi-scale augmentation
# across a 0.3-10 m equivalent range, so sub-metre Cartosat input is inside the
# declared envelope rather than an extrapolation. Where a head is genuinely
# resolution-bound the range says so, and the selector records the penalty.

DEFAULT_PARAMS = {
    "conf_thresh": [0.1, 0.9],
    "topk": [1, 50],
    "dead_band": [0.0, 0.1],
}

REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    REGISTRY[spec.name] = spec
    return spec


def _build_default_registry() -> None:
    register(ToolSpec(
        name="satquery-vqa",
        version="0.1.0",
        tasks=["vqa"],
        modalities=["optical", "sar", "optical+sar"],
        gsd_range=(0.3, 12.0),
        permitted_params={**DEFAULT_PARAMS, "index_thresh": [-1.0, 1.0]},
        device="auto",
        load_cost_mb=340,
        adapted=True,
        notes=("Adapted backbone plus a closed-form answering head. Presence, "
               "count, comparison and area questions are answered from measured "
               "class statistics rather than generated, so every sentence has a "
               "number behind it in TaskResult.raw."),
    ))
    register(ToolSpec(
        name="satquery-caption",
        version="0.1.0",
        tasks=["describe"],
        modalities=["optical", "sar", "optical+sar"],
        gsd_range=(0.3, 12.0),
        permitted_params=DEFAULT_PARAMS,
        device="auto",
        load_cost_mb=340,
        adapted=True,
        notes="Caption head on the shared adapted backbone.",
    ))
    register(ToolSpec(
        name="satquery-ground",
        version="0.1.0",
        tasks=["ground"],
        modalities=["optical", "sar", "optical+sar"],
        gsd_range=(0.3, 12.0),
        permitted_params={**DEFAULT_PARAMS, "min_area_px": [16, 20000],
                          "index_thresh": [-1.0, 1.0],
                          "explanation_method": ("box", "shap", "lime")},
        device="auto",
        load_cost_mb=180,
        adapted=True,
        notes=("Areal classes are segmented and returned as geo-referenced masks "
               "plus component boxes. A box is the wrong primitive for water and "
               "vegetation, so both are emitted and the mask is authoritative."),
    ))
    register(ToolSpec(
        name="satquery-change",
        version="0.1.0",
        tasks=["change", "change-qa"],
        modalities=["optical", "sar", "optical+sar"],
        gsd_range=(0.3, 12.0),
        permitted_params={**DEFAULT_PARAMS, "cva_thresh": [0.5, 6.0],
                          "min_area_px": [16, 20000]},
        device="auto",
        load_cost_mb=220,
        adapted=True,
        notes=("One model serves both tasks: change-qa is derived from the same "
               "mask by class-wise area differencing with a dead-band, which is "
               "more robust than generative answering and keeps the two outputs "
               "consistent by construction."),
    ))
    register(ToolSpec(
        name="satquery-fuse",
        version="0.1.0",
        tasks=["fuse"],
        modalities=["optical+sar"],
        gsd_range=(0.3, 12.0),
        permitted_params={**DEFAULT_PARAMS, "sar_weight": [0.0, 1.0]},
        device="auto",
        load_cost_mb=260,
        adapted=True,
        notes=("Dual-encoder optical+SAR with mid-level feature fusion. Declared "
               "optical+sar only, so the selector cannot pick it for a single "
               "tile - the gate degrades to single-modality instead."),
    ))


_build_default_registry()


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


class NoToolError(Exception):
    """No registered spec satisfies the request. Raised rather than silently
    falling back, so the trace records why nothing ran."""


def select_tool(task: str, modality: str, gsd: float,
                registry: Optional[dict[str, ToolSpec]] = None
                ) -> tuple[ToolSpec, dict[str, Any]]:
    """Filter the registry declaratively and return the best fit plus the
    selection evidence that goes into the trace."""
    reg = registry if registry is not None else REGISTRY

    considered, eligible = [], []
    for spec in reg.values():
        ok_task = task in spec.tasks
        ok_mod = spec.supports(task, modality)
        fit = spec.gsd_fit(gsd)
        considered.append({
            "tool": spec.name, "task_match": ok_task,
            "modality_match": ok_mod, "gsd_fit": round(fit, 4),
        })
        if ok_task and ok_mod:
            eligible.append((fit, spec))

    if not eligible:
        raise NoToolError(
            f"no tool for task={task} modality={modality}; "
            f"candidates: {[c['tool'] for c in considered]}")

    eligible.sort(key=lambda kv: -kv[0])
    fit, spec = eligible[0]

    lo, hi = spec.gsd_range
    in_range = lo <= gsd <= hi
    evidence = {
        "selected": spec.name,
        "selection_rule": "declarative filter over ToolSpec; ranked by gsd_fit",
        "gsd_m": round(gsd, 4),
        "gsd_range": [lo, hi],
        "gsd_in_declared_range": in_range,
        "gsd_fit": round(fit, 4),
        "considered": considered,
        "adapted": spec.adapted,
    }
    if not in_range:
        evidence["resolution_gap_note"] = (
            f"input at {gsd:.2f} m is outside the declared {lo}-{hi} m envelope; "
            "confidence is penalised and the gap is recorded rather than hidden")
    return spec, evidence


# --------------------------------------------------------------------------
# parameter binding
# --------------------------------------------------------------------------


class ParameterError(ValueError):
    """A parameter was not permitted, or was out of its declared bounds."""


def bind_params(spec: ToolSpec, requested: Optional[dict[str, Any]] = None,
                defaults: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Bind only permitted, in-range parameters.

    Unknown keys raise rather than being dropped: silently ignoring an
    unrecognised parameter is how a demo appears to honour a setting it never
    applied.
    """
    requested = dict(requested or {})
    bound: dict[str, Any] = {}

    for key, value in (defaults or {}).items():
        if key in spec.permitted_params:
            bound[key] = value

    for key, value in requested.items():
        if key not in spec.permitted_params:
            raise ParameterError(
                f"'{key}' is not a permitted parameter for {spec.name}; "
                f"permitted: {sorted(spec.permitted_params)}")
        bounds = spec.permitted_params[key]
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lo, hi = bounds
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ParameterError(
                    f"'{key}' must be numeric for {spec.name}, got {value!r}")
            if not (lo <= v <= hi):
                raise ParameterError(
                    f"'{key}'={v} out of declared bounds [{lo}, {hi}] "
                    f"for {spec.name}")
            bound[key] = int(v) if isinstance(lo, int) and isinstance(hi, int) else v
        else:
            if value not in bounds:
                raise ParameterError(
                    f"'{key}'={value!r} not in permitted set {bounds}")
            bound[key] = value

    return bound


def registry_snapshot() -> list[dict[str, Any]]:
    """Registry state for the trace, the /health endpoint and the UI."""
    return [s.to_dict() for s in REGISTRY.values()]

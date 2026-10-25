"""Stage 1 - query interpretation and slot extraction (section 4.1).

Two router versions share one interface:

* ``v1`` - keyword and pattern rules. Deterministic, zero-dependency, and the
  fallback whenever the trained model is unavailable.
* ``v2`` - a trained 6-way classifier over char+word n-grams. It answers the
  "no generic LLM" clause more cleanly than prompt engineering does, because it
  is a component that was *trained* and its accuracy is reportable.

The hard boundary is ``change`` against ``change-qa``: both are bi-temporal, so
lexical routing conflates them, yet one wants a description plus a map and the
other a three-way categorical backed by areas. The seed corpus in
``scripts/train_router.py`` is deliberately weighted toward that boundary and
the confusion matrix for it is published by the evaluation harness.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..contracts import INTENTS, Slots

# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------

#: Classes the grounding head can actually localise. Anything outside this set
#: earns a reasoned abstention with the nearest supported class suggested,
#: rather than a confident box around nothing.
SUPPORTED_CLASSES: dict[str, tuple[str, ...]] = {
    "water": ("water", "water body", "waterbody", "river", "lake", "pond",
              "reservoir", "canal", "channel", "flood", "flooding", "flooded",
              "sea", "ocean", "lagoon", "wetland", "inundation", "waterlogged"),
    "vegetation": ("vegetation", "green", "greenery", "forest", "tree", "trees",
                   "woodland", "canopy", "jungle", "foliage", "plantation",
                   "grass", "grassland", "shrub", "mangrove"),
    "built_up": ("built-up", "built up", "builtup", "urban", "building",
                 "buildings", "settlement", "settlements", "city", "town",
                 "residential", "industrial", "construction", "concrete",
                 "development", "housing", "infrastructure", "impervious"),
    "cropland": ("crop", "crops", "cropland", "farm", "farmland", "field",
                 "fields", "agriculture", "agricultural", "paddy", "plantation",
                 "harvest", "cultivated", "arable"),
    "bare_soil": ("bare", "bare soil", "barren", "soil", "sand", "desert",
                  "dune", "exposed", "fallow", "dry land", "rock", "rocky"),
    "snow_ice": ("snow", "ice", "glacier", "glacial", "frozen", "snowpack",
                 "icecap", "firn"),
    "burn_scar": ("burn", "burnt", "burned", "fire", "wildfire", "scar",
                  "charred", "scorched", "burn scar"),
}

#: Classes a reviewer may plausibly ask for that this build cannot localise.
#: Naming them explicitly is what lets the gate abstain with a reason instead
#: of returning an empty mask that looks like a negative finding.
UNSUPPORTED_CLASSES: dict[str, str] = {
    "ship": "water", "ships": "water", "vessel": "water", "boat": "water",
    "aircraft": "built_up", "plane": "built_up", "airplane": "built_up",
    "vehicle": "built_up", "car": "built_up", "truck": "built_up",
    "road": "built_up", "roads": "built_up", "highway": "built_up",
    "bridge": "built_up", "runway": "built_up", "airport": "built_up",
    "stadium": "built_up", "tank": "built_up", "storage tank": "built_up",
    "tower": "built_up", "chimney": "built_up", "windmill": "built_up",
    "solar panel": "built_up", "harbour": "water", "port": "water",
}

POLARITY_PATTERNS = (
    ("increased", r"\b(increas\w*|grown|grew|expand\w*|rise|risen|rising|"
                  r"more|higher|gain\w*|up|greater|added)\b"),
    ("decreased", r"\b(decreas\w*|shrunk|shrank|shrink\w*|reduc\w*|declin\w*|"
                  r"fell|fall\w*|less|lower|lost|loss|down|smaller|receded)\b"),
    ("unchanged", r"\b(unchanged|same|stable|constant|no change|remain\w*|"
                  r"steady|unaltered)\b"),
)

MODALITY_PATTERNS = (
    ("optical+sar", r"\b(sar|radar|fusion|fuse|fused|both|together|"
                    r"combin\w*|optical and|jointly|complementary)\b"),
    ("sar", r"\b(sar|radar|backscatter|vv|vh|microwave|all-weather)\b"),
    ("optical", r"\b(optical|multispectral|true colour|true color|rgb|visible)\b"),
)

PRESENCE_CUE = re.compile(
    r"\b(check if|checking if|is there|are there|any)\b")


# --------------------------------------------------------------------------
# slot extraction
# --------------------------------------------------------------------------


def _canonical_class(query: str) -> tuple[Optional[str], Optional[str], bool]:
    """Resolve a target class from the query.

    Returns ``(canonical, requested_surface, supported)``. Longest surface form
    wins so "water body" beats "water" and "storage tank" beats "tank".
    """
    q = query.lower()
    best: tuple[int, Optional[str], Optional[str], bool] = (0, None, None, True)

    for canon, surfaces in SUPPORTED_CLASSES.items():
        for s in surfaces:
            # optional plural: "forests"/"lakes" should resolve like the
            # singular surface, otherwise count queries lose their target
            if re.search(rf"\b{re.escape(s)}(?:e?s)?\b", q) and len(s) > best[0]:
                best = (len(s), canon, s, True)

    for surface, nearest in UNSUPPORTED_CLASSES.items():
        if re.search(rf"\b{re.escape(surface)}(?:e?s)?\b", q) \
                and (best[3] or len(surface) > best[0]):
            best = (len(surface), nearest, surface, False)

    return best[1], best[2], best[3]


def extract_slots(query: str, intent: str) -> Slots:
    """Pull structured slots out of the query (section 4.1)."""
    q = query.lower().strip()

    canon, surface, supported = _canonical_class(q)

    polarity = None
    for name, pat in POLARITY_PATTERNS:
        if re.search(pat, q):
            polarity = name
            break

    modality_hint = None
    for name, pat in MODALITY_PATTERNS:
        if re.search(pat, q):
            modality_hint = name
            break

    temporal = re.findall(
        r"\b(?:19|20)\d{2}\b|\b(?:before|after|prior|earlier|later|previous|"
        r"recent|past|monsoon|pre-monsoon|post-monsoon|kharif|rabi|"
        r"jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|"
        r"oct\w*|nov\w*|dec\w*)\b", q)
    temporal_refs = [t if isinstance(t, str) else "".join(t) for t in temporal]
    temporal_refs = [t for t in temporal_refs if t]

    count_query = bool(re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", q))

    output_kind = {
        "ground": "box", "change": "mask", "change-qa": "categorical",
        "fuse": "mask", "describe": "text", "vqa": "text",
    }.get(intent, "text")
    if intent == "vqa" and count_query:
        output_kind = "number"
    if intent == "vqa" and re.search(r"\b(is|are|does|do|has|have|can)\b.*\?*$", q) \
            and re.search(r"\byes\b|\bno\b|\bpresent\b|\bany\b|\bthere\b", q):
        output_kind = "categorical"

    return Slots(
        target_class=canon,
        temporal_refs=temporal_refs[:6],
        polarity=polarity,
        modality_hint=modality_hint,
        output_kind=output_kind,          # type: ignore[arg-type]
        count_query=count_query,
        raw_query=query.strip(),
    )


def class_supported(query: str) -> tuple[bool, Optional[str], Optional[str]]:
    """``(supported, requested_surface, nearest_supported)`` for the gate."""
    canon, surface, supported = _canonical_class(query)
    if surface is None:
        return True, None, None
    return supported, surface, canon


# --------------------------------------------------------------------------
# router v1 - rules
# --------------------------------------------------------------------------


class RuleRouter:
    """Keyword and pattern rules over the six intents.

    Ordering matters: the bi-temporal family is resolved before the
    single-image family, because "has the built-up area increased" contains
    both a change cue and a VQA cue and the change reading is the correct one.
    """

    version = "v1"
    name = "router-rules"

    _CHANGE_CUE = re.compile(
        r"\b(chang\w*|difference|differ|compare|comparison|between\s+(?:these\s+)?"
        r"(?:two|both|the two)|before\s+and\s+after|bi-?temporal|over time|"
        r"since|from\s+\d{4}\s+to\s+\d{4}|two dates|both dates|earlier|"
        r"expand\w*|shrunk|shrank|grown|grew|declin\w*|convert\w*)\b")
    _QA_CUE = re.compile(
        r"\b(increas\w*|decreas\w*|unchanged|remain\w*|same|stable|"
        r"how much|how many|what percentage|what share|by how much|"
        r"ratio|proportion|which class|dominant)\b")
    _GROUND_CUE = re.compile(
        r"\b(highlight|locate|where is|where are|where exactly|show me where|"
        r"check(?:ing)? for|detect|show me|"
        r"mark|outline|delineate|box|bounding box|segment|point out|"
        r"identify the location|pinpoint|find the|region of)\b")
    _FUSE_CUE = re.compile(
        r"\b(fuse|fusion|together|jointly|combin\w*|both sensors|"
        r"optical and sar|sar and optical|complementary|"
        r"using (?:the )?optical and|cloud[- ]?(?:free|pierc\w*|penetrat\w*))\b")
    _DESCRIBE_CUE = re.compile(
        r"\b(describe|description|caption|summar\w*|what does this (?:image|"
        r"scene) show|what is (?:shown|visible|in) (?:in )?this|overview|"
        r"tell me about|land-?cover and)\b")
    _VQA_CUE = re.compile(
        r"^\s*(what|which|how|is|are|does|do|has|have|can|was|were|any)\b|"
        r"\b(check if|checking if|is there|are there)\b|\?")

    def route(self, query: str) -> tuple[str, float, list[tuple[str, float]]]:
        q = query.lower().strip()
        scores: dict[str, float] = {i: 0.0 for i in INTENTS}

        bi_temporal = bool(self._CHANGE_CUE.search(q))
        if bi_temporal:
            if self._QA_CUE.search(q):
                scores["change-qa"] += 0.72
                scores["change"] += 0.24
            else:
                scores["change"] += 0.70
                scores["change-qa"] += 0.20
        elif re.search(r"\b(increas\w*|decreas\w*|unchanged|remain\w*|stable)\b", q) and re.search(
                r"\b(built-?up|water|forest|vegetation|crop|urban|snow|ice)\b", q):
            # polarity language without an explicit change cue is still
            # change-qa when it names a land-cover class
            scores["change-qa"] += 0.45

        if self._FUSE_CUE.search(q):
            scores["fuse"] += 0.68
        if self._GROUND_CUE.search(q):
            scores["ground"] += 0.66
        if self._DESCRIBE_CUE.search(q):
            scores["describe"] += 0.62
        if self._VQA_CUE.search(q):
            scores["vqa"] += 0.34

        if not any(scores.values()):
            scores["describe"] = 0.30

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        total = sum(v for _, v in ranked) or 1.0
        intent, raw = ranked[0]
        conf = min(0.96, 0.45 + 0.55 * raw / total)
        alts = [(k, round(v / total, 4)) for k, v in ranked[1:4] if v > 0]
        return intent, conf, alts


# --------------------------------------------------------------------------
# router v2 - trained classifier
# --------------------------------------------------------------------------


class TrainedRouter:
    """A trained 6-way classifier loaded from disk.

    Falls back to ``RuleRouter`` when the artefact is absent, so a missing
    checkpoint degrades the router rather than breaking the pipeline.
    """

    version = "v2"
    name = "router-trained"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.metrics: dict[str, Any] = {}
        self._fallback = RuleRouter()
        self.loaded = False
        try:
            with open(self.model_path, "rb") as f:
                blob = pickle.load(f)
            self.pipeline = blob["pipeline"]
            self.metrics = blob.get("metrics", {})
            self.loaded = True
        except Exception:
            pass

    def route(self, query: str) -> tuple[str, float, list[tuple[str, float]]]:
        if not self.loaded:
            intent, conf, alts = self._fallback.route(query)
            return intent, conf, alts

        q = query.lower().strip()
        supported, surface, _ = class_supported(q)
        explicit_grounding = (
            supported and surface is not None
            and bool(self._fallback._GROUND_CUE.search(q))
            and not self._fallback._CHANGE_CUE.search(q))
        if explicit_grounding:
            return "ground", 0.94, [("describe", 0.04), ("vqa", 0.02)]

        supported, surface, _ = class_supported(q)
        presence_query = (
            supported and surface is not None
            and bool(PRESENCE_CUE.search(q))
            and not self._fallback._CHANGE_CUE.search(q)
            and not self._fallback._QA_CUE.search(q))
        if presence_query:
            return "vqa", 0.94, [("ground", 0.04), ("describe", 0.02)]

        proba = self.pipeline.predict_proba([query])[0]
        classes = list(self.pipeline.classes_)
        ranked = sorted(zip(classes, proba), key=lambda kv: -kv[1])
        intent, conf = ranked[0][0], float(ranked[0][1])
        alts = [(str(k), round(float(v), 4)) for k, v in ranked[1:4]]
        return str(intent), conf, alts


def load_router(model_path: str | Path | None = None) -> RuleRouter | TrainedRouter:
    """Prefer the trained router; fall back to rules when it is missing."""
    if model_path:
        r = TrainedRouter(model_path)
        if r.loaded:
            return r
    default = Path(__file__).resolve().parents[2] / "models" / "router_v2.pkl"
    if default.exists():
        r = TrainedRouter(default)
        if r.loaded:
            return r
    return RuleRouter()

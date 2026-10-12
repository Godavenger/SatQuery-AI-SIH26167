"""Stage 2 - specialist task heads.

One uniform signature enforced by SpecialistBase. A stub and a trained
checkpoint are interchangeable as long as the contract holds, and the
controller never changes when a head is swapped.

    run(tiles, slots, params) -> TaskResult

Five heads are implemented here:

    VQAHead       -- closed-form VQA from spectral indices + region stats
    CaptionHead   -- land-cover caption from the same stats
    GroundHead    -- areal-class segmentation (water, vegetation, built-up, ...)
    ChangeHead    -- bi-temporal CVA change detection; change-qa is derived
    FuseHead      -- dual-encoder optical+SAR class masks

Every head derives answers from measured pixel statistics (indices, areas,
class fractions) and records those numbers in TaskResult.raw so every sentence
in TaskResult.text is auditable: a reviewer can check the prose against the
measurement.
"""

from __future__ import annotations

import abc
import re
import hashlib
import time
from typing import Any, Optional

import numpy as np

from ..contracts import Answer, Slots, TaskResult, Tile, Visual

# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------


class SpecialistBase(abc.ABC):
    model_id: str = "specialist-base"
    model_version: str = "0.1.0"

    @property
    def weights_hash(self) -> str:
        return hashlib.sha256(
            f"{self.model_id}:{self.model_version}".encode()).hexdigest()[:16]

    @abc.abstractmethod
    def run(self, tiles: list[Tile], slots: Slots,
            params: dict[str, Any]) -> TaskResult:
        ...

    def _result(self, text: str, conf: float, raw: dict,
                visual: Optional[Visual] = None,
                output_kind: str = "text") -> TaskResult:
        return TaskResult(
            text=text, visual=visual or Visual(), confidence=conf,
            raw=raw, model_id=self.model_id, model_version=self.model_version,
            weights_hash=self.weights_hash, output_kind=output_kind,  # type: ignore[arg-type]
        )

    def _neural_result(self, text: str, conf: float, raw: dict,
                       neural_model_id: str,
                       neural_weights_hash: Optional[str],
                       visual: Optional[Visual] = None,
                       output_kind: str = "text") -> TaskResult:
        """A result the trained weights produced, labelled as such.

        ``model_id`` is suffixed so the trace, the PDF and the frontend
        (which renders ``ans.contributing[0]``) all name the checkpoint
        that answered rather than the head class. ``weights_hash`` is the
        real sha256 of the artifact, not the identity-string hash
        ``weights_hash`` returns for the closed-form path.
        """
        return TaskResult(
            text=text, visual=visual or Visual(), confidence=conf,
            raw=raw, model_id=f"{self.model_id}+{neural_model_id}",
            model_version=self.model_version,
            weights_hash=neural_weights_hash or self.weights_hash,
            output_kind=output_kind,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# shared spectral utilities
# --------------------------------------------------------------------------

def _band(tile: Tile, *slots: str) -> Optional[np.ndarray]:
    for s in slots:
        if s in tile.bands:
            return tile.bands[s].astype(np.float32)
    return None


def _ndvi(tile: Tile) -> Optional[np.ndarray]:
    nir = _band(tile, "nir")
    red = _band(tile, "red")
    if nir is None or red is None:
        return None
    denom = nir + red
    return np.where(denom > 1e-6, (nir - red) / denom, 0.0)


def _ndwi(tile: Tile) -> Optional[np.ndarray]:
    green = _band(tile, "green")
    nir = _band(tile, "nir")
    if green is None or nir is None:
        return None
    denom = green + nir
    return np.where(denom > 1e-6, (green - nir) / denom, 0.0)


def _ndbi(tile: Tile) -> Optional[np.ndarray]:
    swir = _band(tile, "swir1")
    nir = _band(tile, "nir")
    if swir is None or nir is None:
        return None
    denom = swir + nir
    return np.where(denom > 1e-6, (swir - nir) / denom, 0.0)


def _nbr(tile: Tile) -> Optional[np.ndarray]:
    nir = _band(tile, "nir")
    swir = _band(tile, "swir2", "swir1")
    if nir is None or swir is None:
        return None
    denom = nir + swir
    return np.where(denom > 1e-6, (nir - swir) / denom, 0.0)


def _ndsii(tile: Tile) -> Optional[np.ndarray]:
    """Normalised difference snow/ice index (uses SWIR if available)."""
    green = _band(tile, "green")
    swir = _band(tile, "swir1")
    if green is None:
        return None
    if swir is None:
        # Fall back to visible-only: bright pixels without nir
        red = _band(tile, "red")
        if red is None:
            return None
        b = _band(tile, "blue")
        if b is None:
            return None
        return np.where(b + red > 1e-6, (b - red) / (b + red + 1e-6), 0.0)
    denom = green + swir
    return np.where(denom > 1e-6, (green - swir) / denom, 0.0)


def _sar_water(tile: Tile, thresh: float = -14.0) -> Optional[np.ndarray]:
    vv = _band(tile, "VV", "HH")
    if vv is None:
        return None
    return vv < thresh


def _classify(tile: Tile) -> dict[str, np.ndarray]:
    """Simple class masks from spectral thresholds."""
    masks: dict[str, np.ndarray] = {}
    h, w = tile.shape
    valid = tile.valid_mask

    ndvi = _ndvi(tile)
    ndwi = _ndwi(tile)
    ndbi = _ndbi(tile)
    ndsii = _ndsii(tile)
    nbr = _nbr(tile)

    if ndwi is not None:
        masks["water"] = (ndwi > 0.10) & valid
    elif _band(tile, "VV") is not None:
        m = _sar_water(tile)
        if m is not None:
            masks["water"] = m & valid

    if ndvi is not None:
        masks["vegetation"] = (ndvi > 0.25) & valid
        masks["bare_soil"] = (ndvi < 0.10) & valid

    if ndbi is not None:
        masks["built_up"] = (ndbi > 0.05) & (
            ndvi < 0.20 if ndvi is not None else np.ones((h, w), bool)) & valid

    if ndsii is not None:
        masks["snow_ice"] = (ndsii > 0.40) & valid

    if nbr is not None:
        masks["burn_scar"] = (nbr < -0.10) & valid

    return masks


def _class_stats(tile: Tile) -> dict[str, Any]:
    masks = _classify(tile)
    total = tile.valid_mask.sum()
    if total == 0:
        return {}
    stats: dict[str, Any] = {}
    for cls, m in masks.items():
        pct = float(100.0 * m.sum() / total)
        if pct > 0.2:
            stats[cls] = {"fraction": round(pct / 100, 4),
                          "pct": round(pct, 2),
                          "area_px": int(m.sum())}
    return stats


def _dominant(stats: dict[str, Any]) -> Optional[str]:
    if not stats:
        return None
    return max(stats, key=lambda k: stats[k]["fraction"])


def _boxes_from_mask(mask: np.ndarray, tile: Tile,
                     min_area_px: int = 32) -> list[dict[str, Any]]:
    """Connected-components boxes in geographic coordinates."""
    from scipy.ndimage import label as ndlabel
    lbl, n = ndlabel(mask.astype(np.uint8))
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_area_px:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        geo = tile.box_to_geo(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        boxes.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "area_px": int(len(xs)), **geo,
        })
    return sorted(boxes, key=lambda b: -b["area_px"])[:8]


def _rgb_for_explanation(tile: Tile) -> Optional[np.ndarray]:
    """Return a stable 0--1 RGB image for image-level explainers.

    SHAP and LIME explain the auditable spectral grounding detector, not a
    black-box caption. The image-level score is the fraction of target-class
    evidence, so every attribution pixel can be traced back to the detector
    that produced the mask.
    """
    red = _band(tile, "red")
    green = _band(tile, "green")
    blue = _band(tile, "blue")
    if red is None or green is None or blue is None:
        return None
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    finite = np.isfinite(rgb)
    if not finite.any():
        return None
    lo, hi = np.percentile(rgb[finite], [1.0, 99.0])
    if hi - lo < 1e-6:
        return None
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def _target_detector_score(images: np.ndarray, target: str) -> np.ndarray:
    """Image-level score used by SHAP and LIME for the requested class."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[None, ...]
    red, green, blue = images[..., 0], images[..., 1], images[..., 2]
    total = red + green + blue + 1e-6
    if target == "vegetation":
        score = (green - red) + (green - blue)
    elif target == "water":
        score = (blue - red) + (green - red) * 0.5
    elif target == "built_up":
        score = np.maximum(red, np.maximum(green, blue)) - green
    elif target == "bare_soil":
        score = red - green
    elif target == "snow_ice":
        score = np.maximum(red, np.maximum(green, blue))
    elif target == "burn_scar":
        score = red - green - blue * 0.25
    else:
        score = green - red
    return np.clip(score.mean(axis=(1, 2)) / (total.mean(axis=(1, 2)) + 1e-6), 0.0, 1.0)


def _shap_sampling_attribution(rgb: np.ndarray,
                               target: str) -> Optional[np.ndarray]:
    """Numba-free one-sample Shapley approximation for image superpixels.

    The installed SHAP package can be incompatible with a newer NumPy. This
    fallback still estimates the same marginal contributions: for every
    superpixel it measures adding that pixel to the empty coalition and
    removing it from the full coalition, then averages the two marginal
    effects as a Shapley-value sample.
    """
    try:
        from skimage.segmentation import slic
        segments = slic(rgb, n_segments=24, compactness=5.0,
                        channel_axis=-1, start_label=0)
    except Exception:                                          # noqa: BLE001
        segments = np.add.outer(
            np.arange(rgb.shape[0]) * 6 // rgb.shape[0],
            np.arange(rgb.shape[1]) * 6 // rgb.shape[1])

    baseline_score = float(_target_detector_score(
        np.zeros_like(rgb)[None, ...], target)[0])
    full_score = float(_target_detector_score(rgb[None, ...], target)[0])
    segment_values = {}
    for segment_id in np.unique(segments):
        selected = segments == segment_id
        images = np.stack([
            np.where(selected[..., None], rgb, 0.0),
            np.where(selected[..., None], 0.0, rgb),
        ])
        scores = _target_detector_score(images, target)
        add_effect = float(scores[0]) - baseline_score
        remove_effect = full_score - float(scores[1])
        segment_values[int(segment_id)] = 0.5 * (add_effect + remove_effect)

    attribution = np.zeros(rgb.shape[:2], dtype=np.float32)
    for segment_id, value in segment_values.items():
        if value > 0:
            attribution[segments == segment_id] = value
    top = float(np.percentile(attribution, 98))
    if top <= 1e-8:
        return np.zeros_like(attribution)
    return np.clip(attribution / top, 0.0, 1.0)


def _explain_target(tile: Tile, target: str,
                    method: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Run SHAP or LIME and return a 2-D positive-evidence map."""
    if method not in ("shap", "lime"):
        return None, None
    rgb = _rgb_for_explanation(tile)
    if rgb is None:
        return None, "RGB bands unavailable for image explanation"
    try:
        if method == "shap":
            import shap
            baseline = np.zeros_like(rgb)
            explainer = shap.Explainer(
                lambda images: _target_detector_score(images, target),
                baseline, algorithm="partition")
            values = explainer(rgb, max_evals=48, batch_size=8).values
            attribution = np.asarray(values, dtype=np.float32).squeeze()
            if attribution.ndim == 3:
                attribution = attribution.mean(axis=-1)
            if attribution.ndim != 2 or attribution.shape != tile.shape:
                return None, "SHAP returned an unexpected attribution shape"
            positive = np.maximum(attribution, 0.0)
            top = float(np.percentile(positive, 98))
            if top <= 1e-8:
                return np.zeros_like(positive), None
            return positive / top, None

        from lime import lime_image
        explainer = lime_image.LimeImageExplainer(random_state=26167)
        def lime_classifier(images: np.ndarray) -> np.ndarray:
            scores = _target_detector_score(images, target)
            return np.column_stack([1.0 - scores, scores])
        explanation = explainer.explain_instance(
            rgb, lime_classifier,
            top_labels=1, hide_color=0, num_samples=48)
        label = int(explanation.top_labels[0])
        _, mask = explanation.get_image_and_mask(
            label, positive_only=True, hide_rest=False)
        return np.asarray(mask, dtype=np.float32), None
    except Exception as e:                                      # noqa: BLE001
        if method == "shap":
            sampled = _shap_sampling_attribution(rgb, target)
            if sampled is not None:
                return sampled, None
        return None, f"{method} explanation failed: {e}"


# --------------------------------------------------------------------------
# trained-head bridge
# --------------------------------------------------------------------------
#
# These wrap satquery.stage2.neural so the heads never see a torch error:
# the neural stack degrades to None on any failure (missing artifact,
# encode error, out-of-vocabulary question) and the caller falls back to
# the closed-form spectral answer. A repo cloned without models/ answers
# exactly as it did before this bridge existed.


def _neural_presence(tile: Tile, target_class: str) -> Optional[dict]:
    try:
        from .neural import get_neural
        return get_neural().presence(tile, target_class)
    except Exception:                                          # noqa: BLE001
        return None


def _neural_lulc(tile: Tile) -> Optional[dict]:
    try:
        from .neural import get_neural
        return get_neural().lulc(tile)
    except Exception:                                          # noqa: BLE001
        return None


def _neural_answer(tile: Tile, question: str,
                   options: list[str]) -> Optional[dict]:
    """Rank fixed options for a free-form question with the trained scorer."""
    try:
        from .neural import get_neural
        return get_neural().answer(tile, question, options)
    except Exception:                                          # noqa: BLE001
        return None


def _neural_sar_lulc(tile: Tile) -> Optional[dict]:
    """Trained S1 (SAR) land-cover reading of a radar tile."""
    try:
        from .neural import get_neural
        return get_neural().sar_lulc(tile)
    except Exception:                                          # noqa: BLE001
        return None


def _neural_sar_presence(tile: Tile,
                         target_class: str) -> Optional[dict]:
    """Trained S1 land-cover presence decision for a radar tile."""
    try:
        from .neural import get_neural
        return get_neural().sar_presence(tile, target_class)
    except Exception:                                          # noqa: BLE001
        return None


_MCQ_SPLIT = re.compile(r"\s*\(?([a-j])\)\s*")


def _parse_mcq(query: str) -> Optional[tuple[str, list[str]]]:
    """Extract (stem, options) when a query carries inline MCQ options.

    Mirrors the annotation format the scorer was trained on
    ("... a) first b) second c) third"). Returns None for ordinary
    questions -- most live queries have no options, and those go through
    the presence / open paths instead.
    """
    m = re.search(r"\s\(?[a-j]\)\s", query)
    if not m:
        return None
    stem = query[:m.start()].strip().rstrip("?").strip()
    rest = " " + query[m.start():].strip()
    parts = _MCQ_SPLIT.split(rest)
    opts = []
    for i in range(1, len(parts) - 1, 2):
        opt = parts[i + 1].strip()
        if opt:
            opts.append(opt)
    if len(opts) < 2:
        return None
    return stem, opts


#: A measured class fraction at or above this percentage counts as real
#: spectral evidence -- enough that the trained classifier is not allowed
#: to silently veto it.
SPECTRAL_PRESENT_PCT = 1.0


def _reconcile(target: str, neural: Optional[dict],
               spec: Optional[dict], scorer: Optional[dict] = None,
               ) -> tuple[bool, str, float, str]:
    """Reconcile the trained heads against the measured fraction.

    Returns ``(present, sentence, confidence, verdict)``.

    Three opinions meet here, with a deliberate precedence:

    1. The measured spectral fraction (pixel counts are directly
       auditable). The BigEarthNet heads were trained on European
       imagery; on South Asian scenes they can mislabel with high
       confidence -- Kaziranga NP reads 0.88 urban while NDWI measures
       1.75% real Brahmaputra channel. A measured fraction at or above
       ``SPECTRAL_PRESENT_PCT`` is therefore never overturned by either
       learned head: the disagreement is reported and costs confidence,
       but the measurement keeps authorship of the answer.
    2. The learned land-cover classifier (caption head), which reads the
       image feature alone and stays sharp across tile sizes.
    3. The trained VQA scorer v2 (MiniLM text, so any phrasing works).
       It never overrides the measurement either; it adjusts confidence
       with its agreement and, when it is the ONLY opinion that sees the
       class, that is reported as the sole evidence -- never hidden.

    The converse is where the classifier earns its keep: when the
    measurement is scattered noise, agreement is what turns a spurious
    detection into an honest abstention, and a decisive scorer "yes" with
    no spectral signature is surfaced as "scorer-only" rather than
    flattened into "no" (beaches, for instance, have no NDWI signature
    at all).
    """
    name = target.replace("_", " ")
    spec_pct = float(spec["pct"]) if spec else 0.0
    spec_present = spec_pct >= SPECTRAL_PRESENT_PCT
    p = neural["prob"] if neural else None

    sc_p = scorer["probs"].get("yes") if scorer else None
    sc_decided = bool(scorer and scorer.get("decided"))
    sc_yes = bool(scorer and sc_decided
                  and scorer.get("answer") == "yes")
    sc_no = bool(scorer and sc_decided
                 and scorer.get("answer") == "no")

    def sc_note(expected: bool) -> str:
        if sc_p is None:
            return ""
        if (sc_yes or sc_no) and (sc_yes == expected):
            return f" The trained VQA scorer agrees (p_yes={sc_p:.2f})."
        if not sc_decided:
            return f" The trained VQA scorer is undecided (p_yes={sc_p:.2f})."
        return f" The trained VQA scorer disagrees (p_yes={sc_p:.2f})."

    # -- measurement wins, as before ---------------------------------
    if spec_present and neural is None:
        return (True,
                f"Yes -- {name} is present, covering {spec_pct:.1f}% of "
                f"the tile ({spec['area_px']} px)."
                + sc_note(True),
                min(0.94, 0.65 + 0.29 * spec["fraction"]
                    + (0.05 if sc_yes else 0.0)),
                "spectral-only" if sc_p is None else "measured+scorer")

    if spec_present and neural is not None:
        if neural["present"]:
            text = (f"Yes -- {name} is present. The learned land-cover "
                    f"classifier reports '{neural['label']}' (p={p:.2f}) "
                    f"and the spectral measurement agrees: {spec_pct:.1f}% "
                    f"of the tile ({spec['area_px']} px)."
                    + sc_note(True))
            conf = min(0.95, 0.70 + 0.24 * p + (0.05 if sc_yes else 0.0))
            return (True, text, conf, "agree-present")
        text = (f"Yes -- {name} is measured at {spec_pct:.1f}% of the tile "
                f"({spec['area_px']} px). Note: the learned land-cover "
                f"classifier does not see it (p={p:.2f}); that head was "
                "trained on European imagery and is unreliable on this "
                "landscape, so the direct measurement is reported."
                + sc_note(True))
        conf = min(0.78, 0.58 + 0.20 * spec["fraction"]
                   + (0.04 if sc_yes else 0.0))
        return (True, text, conf, "conflict-measured")

    # -- no spectral signature --------------------------------------
    if neural is not None and neural["present"]:
        text = (f"Yes -- the learned land-cover classifier reports "
                f"'{neural['label']}' (p={p:.2f}), though the spectral "
                f"indices do not localise a significant fraction "
                f"({spec_pct:.1f}%)."
                + (f" The trained VQA scorer confirms (p_yes={sc_p:.2f})."
                   if sc_yes else sc_note(True)))
        conf = min(0.80, 0.52 + 0.28 * p + (0.08 if sc_yes else 0.0))
        return (True, text, conf, "classifier-only")

    if sc_yes:
        # No measurement, no LULC head -- but the VQA scorer, which reads
        # the question AND the image, decides "yes". Report it as the
        # sole evidence with honest, bounded confidence.
        text = (f"Yes -- the trained VQA scorer reports {name} present "
                f"(p_yes={sc_p:.2f}, margin {scorer['margin']:.2f}). "
                "Neither the spectral indices nor the land-cover "
                "classifier localise it, so the finding rests on the "
                "question-answering model alone.")
        return (True, text, min(0.62, 0.45 + 0.25 * sc_p), "scorer-only")

    absent_agree = [neural is None or not neural["present"]]
    if sc_no:
        absent_agree.append(True)
    conf_base = 0.90 if len(absent_agree) == 2 else 0.72
    text = (f"No -- no significant {name} signature detected in this tile"
            + (f"; the learned land-cover classifier agrees (p={p:.2f})"
               if neural is not None else "")
            + (f" and so does the trained VQA scorer (p_yes={sc_p:.2f})"
               if sc_no else "") + ".")
    return (False, text, min(conf_base, 0.70 + 0.20 * (
        1.0 - (p if p is not None else 0.5))), "agree-absent")


# --------------------------------------------------------------------------
# VQA head
# --------------------------------------------------------------------------


class VQAHead(SpecialistBase):
    """Closed-form visual question answering from spectral indices (section 5.3).

    Every question type computable from class statistics is answered from
    measurements rather than from generated text:

        presence  -- "is there water?" -> "yes" or "no" with fraction
        count     -- route to approximate count from connected components
        comparison -- t1 vs t2 not supported in single-tile path
        area      -- return fraction + estimated area in ha at the tile GSD
        open      -- fall back to the caption answer

    This is exactly what the change-qa section calls "more robust than
    generative answering": the number in raw matches the word in text.
    """

    model_id = "satquery-vqa"

    def run(self, tiles: list[Tile], slots: Slots,
            params: dict[str, Any]) -> TaskResult:
        t0 = time.perf_counter()
        tile = tiles[0]
        q = slots.raw_query.lower()
        stats = _class_stats(tile)
        target = slots.target_class

        # -- count question ---------------------------------------------
        # checked before presence: "how mANY ..." contains the substring
        # "any " and would otherwise fall into the presence branch
        if slots.count_query and target:
            mask = _classify(tile).get(target)
            if mask is not None:
                from scipy.ndimage import label as ndlabel
                _, n = ndlabel(mask.astype(np.uint8))
                text = (f"Approximately {n} distinct "
                        f"{target.replace('_', ' ')} region(s) detected "
                        f"covering {stats.get(target, {}).get('pct', 0):.1f}% "
                        "of the tile.")
                conf = 0.68
                return self._result(text, conf, {"count": n, "class": target},
                                    output_kind="number")

        # -- presence question -------------------------------------------
        is_presence = any(kw in q for kw in
                          ("is there", "are there", "any ", "present",
                           "can you see", "does this", "do you see"))
        if is_presence and target:
            # Three opinions meet in one place: the measured spectral
            # fraction (auditable, always wins at >= 1% coverage), the
            # learned land-cover head, and the trained VQA scorer v2 --
            # whose MiniLM text side reads any phrasing, not just
            # BigEarthNet.txt's. See _reconcile for the precedence.
            # SAR-only tiles have no optical path into those heads; the
            # jointly trained S1 head is their learned opinion instead.
            if tile.has_optical:
                neural = _neural_presence(tile, target)
                scorer = _neural_answer(tile, slots.raw_query,
                                        ["yes", "no"])
            else:
                neural = _neural_sar_presence(tile, target)
                scorer = None
            spec = stats.get(target)
            present, text, conf, verdict = _reconcile(
                target, neural, spec, scorer)
            raw = {"neural": neural, "spectral": spec or {},
                   "target": target, "present": present,
                   "reconciliation": verdict, "vqa_scorer": scorer}
            src_id = (scorer or {}).get("model_id") or (neural or {}).get(
                "model_id")
            src_hash = ((scorer or {}).get("weights_hash")
                        or (neural or {}).get("weights_hash"))
            if src_id:
                return self._neural_result(
                    text, conf, raw, src_id, src_hash,
                    output_kind="categorical")
            return self._result(text, conf, raw, output_kind="categorical")

        # -- multiple-choice question --------------------------------------
        # A question carrying inline options ("... a) forest b) urban
        # c) water d) farmland") is what the trained VQA scorer is FOR:
        # it ranks options against the image. The MiniLM text side reads
        # any phrasing, and spectral stats are appended as grounding.
        mcq = _parse_mcq(slots.raw_query)
        if mcq:
            stem, opts = mcq
            ranked = _neural_answer(tile, stem, opts)
            dom = _dominant(stats)
            if ranked and ranked["answer"]:
                text = f"{ranked['answer'].capitalize()}."
                if ranked["decided"]:
                    text += (f" The trained VQA scorer selects this option "
                             f"with p={ranked['prob']:.2f} (margin "
                             f"{ranked['margin']:.2f} over the runner-up).")
                else:
                    text += (f" The scorer's margin is thin "
                             f"({ranked['margin']:.2f}); treat this as its "
                             "best guess among the options.")
                if dom:
                    text += (f" Measured grounding: {dom.replace('_', ' ')} "
                             f"{stats[dom]['pct']:.1f}% of the tile.")
                    others = {k: v["pct"] for k, v in stats.items()
                              if k != dom}
                    if others:
                        tops = sorted(others.items(),
                                      key=lambda kv: -kv[1])[:2]
                        text += "; " + ", ".join(
                            f"{k.replace('_', ' ')} {v:.1f}%"
                            for k, v in tops)
                    text += "."
                conf = (min(0.88, 0.50 + 0.45 * ranked["prob"])
                        if ranked["decided"] else 0.55)
                return self._neural_result(
                    text, conf,
                    {"mcq": {"stem": stem, "options": opts},
                     "vqa_scorer": ranked, "stats": stats,
                     "latency_ms": round((time.perf_counter() - t0) * 1000, 1)},
                    ranked["model_id"], ranked["weights_hash"],
                    output_kind="categorical")

        # -- area question -----------------------------------------------
        is_area = any(kw in q for kw in ("how much", "what area", "what fraction",
                                          "percentage", "share", "proportion"))
        if is_area and target and target in stats:
            pct = stats[target]["pct"]
            gsd = tile.gsd
            ha = stats[target]["area_px"] * (gsd ** 2) / 10_000.0
            text = (f"{target.replace('_', ' ').title()} covers {pct:.1f}% of "
                    f"the tile, approximately {ha:.0f} ha at {gsd:.1f} m GSD.")
            conf = 0.82
            return self._result(text, conf,
                                {"pct": pct, "area_ha": round(ha, 1),
                                 "gsd_m": round(gsd, 2)},
                                output_kind="number")

        # -- open question -- learned land-cover first, measurement second -
        neural_lulc = _neural_lulc(tile)
        dom = _dominant(stats)
        if neural_lulc and neural_lulc["above_threshold"]:
            named = neural_lulc["above_threshold"][:3]
            text = ("The learned land-cover classifier reports "
                    + ", ".join(f"{lab.lower()} (p={p:.2f})"
                                for lab, p in named) + ".")
            if dom:
                text += (f" Spectral grounding: {dom.replace('_', ' ')} "
                         f"{stats[dom]['pct']:.1f}% of the tile.")
            if not stats and tile.has_sar and not tile.has_optical:
                # SAR-only tile: the optical stats are empty, say what the
                # S1 side of the model reads instead of "no class".
                sar = _neural_sar_lulc(tile)
                if sar and sar["above_threshold"]:
                    snamed = sar["above_threshold"][:2]
                    text += (" The Sentinel-1 (SAR) land-cover head reads "
                             + ", ".join(f"{l.lower()} (p={p:.2f})"
                                         for l, p in snamed) + ".")
            conf = min(0.92, 0.55 + 0.40 * named[0][1])
            return self._neural_result(
                text, conf,
                {"neural": neural_lulc, "stats": stats, "dominant": dom,
                 "latency_ms": round((time.perf_counter() - t0) * 1000, 1)},
                neural_lulc["model_id"], neural_lulc["weights_hash"])

        if dom:
            pct = stats[dom]["pct"]
            others = {k: v["pct"] for k, v in stats.items() if k != dom}
            text = (f"The dominant land-cover is "
                    f"{dom.replace('_', ' ')} ({pct:.1f}%)")
            if others:
                parts = sorted(others.items(), key=lambda kv: -kv[1])
                text += ", with " + ", ".join(
                    f"{k.replace('_', ' ')} {v:.1f}%" for k, v in parts[:3])
            text += "."
            conf = 0.75
        elif tile.has_sar and not tile.has_optical:
            # closed-form has no SAR land-cover vocabulary beyond water;
            # the trained S1 head is the informed opinion here
            sar = _neural_sar_lulc(tile)
            if sar and sar["above_threshold"]:
                snamed = sar["above_threshold"][:3]
                text = ("SAR-only tile. The trained Sentinel-1 land-cover "
                        "head reports "
                        + ", ".join(f"{l.lower()} (p={p:.2f})"
                                    for l, p in snamed) + ".")
                conf = min(0.80, 0.50 + 0.35 * snamed[0][1])
                return self._neural_result(
                    text, conf,
                    {"sar_lulc": sar, "stats": stats,
                     "latency_ms": round(
                         (time.perf_counter() - t0) * 1000, 1)},
                    sar["model_id"], sar["weights_hash"])
            text = ("SAR-only tile; no optical bands to classify and the "
                    "SAR land-cover head is not decisive.")
            conf = 0.40
        else:
            text = "No clear spectral class detected in this tile."
            conf = 0.40

        return self._result(
            text, conf,
            {"stats": stats, "dominant": dom, "neural": neural_lulc,
             "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})


# --------------------------------------------------------------------------
# caption head
# --------------------------------------------------------------------------


class CaptionHead(SpecialistBase):
    """Scene caption led by the trained BigEarthNet-19 land-cover head."""

    model_id = "satquery-caption"

    def run(self, tiles: list[Tile], slots: Slots,
            params: dict[str, Any]) -> TaskResult:
        tile = tiles[0]
        stats = _class_stats(tile)
        neural = _neural_lulc(tile) if tile.has_optical else None
        # SAR-only tiles have no optical path into the caption head; the
        # jointly trained S1 head is the learned opinion that applies.
        sar_neural = None
        if not tile.has_optical and tile.has_sar:
            sar_neural = _neural_sar_lulc(tile)
            neural = sar_neural

        classes = sorted(stats.items(), key=lambda kv: -kv[1]["fraction"])

        # Build the sentence: name + fraction + spectral note.
        parts = []
        for cls, s in classes[:4]:
            name = cls.replace("_", " ")
            pct = s["pct"]
            parts.append(f"{name} ({pct:.1f}%)")

        ndvi = _ndvi(tile)
        ndwi = _ndwi(tile)
        notes = []
        if ndvi is not None:
            mean_ndvi = float(np.nanmean(ndvi))
            notes.append(f"mean NDVI {mean_ndvi:.2f}")
        if ndwi is not None:
            mean_ndwi = float(np.nanmean(ndwi))
            notes.append(f"NDWI {mean_ndwi:.2f}")

        sensor_label = tile.sensor_id.replace("_", " ")

        # -- neural-primary: name the learned classes, then the measurement
        named = (neural or {}).get("above_threshold") or []
        if not named and neural:
            # Nothing cleared 0.5; fall back to the single most likely label
            # so the caption still reports what the trained head believes.
            named = neural["top"][:1]
        if named:
            src = ("SAR land-cover head" if sar_neural is not None
                   else "learned land-cover classifier")
            text = (f"This {sensor_label} scene at {tile.gsd:.1f} m GSD is "
                    f"read by the {src} as " +
                    ", ".join(f"{lab.lower()} (p={p:.2f})"
                              for lab, p in named[:3]) + ".")
            if sar_neural is not None:
                text += (" Radar penetrates cloud and acquires day-night, "
                         "so this reading holds even where optical "
                         "imagery would be blind.")
            if parts:
                text += " Measured cover: " + "; ".join(parts) + "."
            if notes:
                text += " Spectral indices: " + ", ".join(notes) + "."
            conf = min(0.90, 0.55 + 0.38 * float(named[0][1]))
            return self._neural_result(
                text, conf,
                {"neural": neural, "stats": stats, "notes": notes,
                 "sar_lulc": sar_neural},
                neural["model_id"], neural["weights_hash"])

        # -- spectral fallback (unchanged behaviour) ----------------------
        if parts:
            text = ("This " + sensor_label +
                    f" scene at {tile.gsd:.1f} m GSD shows: " +
                    "; ".join(parts) + ".")
        else:
            text = (f"Scene at {tile.gsd:.1f} m GSD from {sensor_label}. "
                    "No dominant spectral class resolved.")
        if notes:
            text += " Spectral indices: " + ", ".join(notes) + "."

        conf = 0.78 if classes else 0.42
        return self._result(text, conf, {"stats": stats, "notes": notes,
                                         "sar_lulc": sar_neural})


# --------------------------------------------------------------------------
# grounding head
# --------------------------------------------------------------------------


class GroundHead(SpecialistBase):
    """Areal-class grounding (section 5.3 -- areal classes).

    Grounding is the one intent where the spectral thresholds fail loudly:
    a handful of scattered index-noise pixels clear the total-area test
    while no single connected component is large enough to box, and the
    old wording then reported "detected ... the largest region is boxed"
    over zero regions. The trained BigEarthNet head is consulted as the
    presence prior that catches exactly that case.
    """

    model_id = "satquery-ground"

    def run(self, tiles: list[Tile], slots: Slots,
            params: dict[str, Any]) -> TaskResult:
        tile = tiles[0]
        target = slots.target_class or "water"
        thresh = params.get("conf_thresh", 0.35)
        min_area = int(params.get("min_area_px", 32))

        neural = _neural_presence(tile, target)
        masks = _classify(tile)
        mask = masks.get(target)
        mask_sum = int(mask.sum()) if mask is not None else 0

        if mask is None or mask_sum < min_area:
            note = ""
            if neural is not None:
                note = (f" The learned land-cover classifier agrees "
                        f"(p={neural['prob']:.2f})." if not neural["present"]
                        else (f" Note: the learned classifier expects it "
                              f"(p={neural['prob']:.2f}); the spectral "
                              "indices do not localise it here."))
            return self._result(
                f"No significant '{target.replace('_', ' ')}' region found in "
                "the current tile." + note,
                0.46, {"target": target, "mask_sum": mask_sum,
                       "neural": neural},
                output_kind="box")

        h, w = tile.shape
        gsd = tile.gsd
        pct = float(100.0 * mask_sum / max(tile.valid_mask.sum(), 1))
        ha = float(mask_sum * gsd ** 2 / 10_000.0)
        boxes = _boxes_from_mask(mask, tile, min_area)
        explanation_method = str(params.get("explanation_method", "box")).lower()
        attribution, explanation_error = _explain_target(
            tile, target, explanation_method)

        vis = Visual(kind="overlay" if attribution is not None else "mask",
                     mask=mask.astype(np.uint8),
                     mask_classes={target: 1},
                     boxes=boxes,
                     attribution=attribution,
                     attribution_method=(
                         explanation_method if attribution is not None else None),
                     legend=[
                         {"class": target, "color": "#22c55e" if target == "vegetation" else "#3b82f6"},
                         {"class": f"{explanation_method} evidence", "color": "#f59e0b"},
                     ] if attribution is not None else
                     [{"class": target, "color": "#22c55e" if target == "vegetation" else "#3b82f6"}])

        # -- no component survived the minimum-area filter -----------------
        # The pixels exist but nothing is localisable. Saying "detected" and
        # "boxed" here is what produced the contradictory 0.0% / 0 regions
        # answer; report the scatter honestly and let the low confidence
        # drive Stage 3 to abstain.
        if not boxes:
            text = (f"No localisable '{target.replace('_', ' ')}' region: "
                    f"{mask_sum} scattered pixels ({pct:.2f}% of the tile) "
                    f"pass the spectral test but no connected component "
                    f"reaches the {min_area} px minimum, so nothing can be "
                    "boxed.")
            conf = 0.30
            if neural is not None and not neural["present"]:
                text += (f" The learned land-cover classifier also reports "
                         f"the class absent (p={neural['prob']:.2f}).")
                conf = 0.22
                return self._neural_result(
                    text, conf,
                {"target": target, "pct": round(pct, 3),
                 "mask_sum": mask_sum, "n_regions": 0,
                 "neural": neural, "gsd_m": round(gsd, 2),
                 "explanation_method": explanation_method,
                 "explanation_error": explanation_error},
                    neural["model_id"], neural["weights_hash"],
                    visual=vis, output_kind="box")
            return self._result(
                text, conf,
                {"target": target, "pct": round(pct, 3), "mask_sum": mask_sum,
                 "n_regions": 0, "neural": neural, "gsd_m": round(gsd, 2)},
                visual=vis, output_kind="box")

        # -- a real region was localised -----------------------------------
        text = (f"{target.replace('_', ' ').title()} detected: "
                f"{pct:.1f}% of the tile ({ha:.0f} ha at {gsd:.1f} m GSD), "
                f"{len(boxes)} connected region(s). "
                f"The largest region is boxed on the imagery.")
        conf = min(0.91, 0.55 + 0.36 * pct / 100.0)

        if neural is not None:
            if neural["present"]:
                text += (f" The learned land-cover classifier confirms "
                         f"'{neural['label']}' (p={neural['prob']:.2f}).")
                conf = min(0.94, conf + 0.12)
            else:
                # Cross-model disagreement over a polygon that genuinely
                # exists. The penalty is scaled by how strong the
                # measurement is: a substantial, localised region keeps
                # its answer (the classifier is off-distribution on these
                # landscapes), while a marginal one is damped toward
                # abstention.
                text += (f" Caution: the learned land-cover classifier does "
                         f"NOT see this class (p={neural['prob']:.2f}). That "
                         "head was trained on European imagery; the measured "
                         "region is reported because a pixel count is "
                         "directly auditable.")
                conf = max(0.25, conf - (0.05 if pct >= SPECTRAL_PRESENT_PCT
                                         else 0.20))
            return self._neural_result(
                text, conf,
                {"target": target, "pct": round(pct, 2),
                 "area_ha": round(ha, 1), "n_regions": len(boxes),
                 "neural": neural, "gsd_m": round(gsd, 2),
                 "explanation_method": explanation_method,
                 "explanation_error": explanation_error},
                neural["model_id"], neural["weights_hash"],
                visual=vis, output_kind="box")

        return TaskResult(
            text=text, visual=vis, confidence=conf,
            raw={"target": target, "pct": round(pct, 2),
                 "area_ha": round(ha, 1), "n_regions": len(boxes),
                 "gsd_m": round(gsd, 2),
                 "explanation_method": explanation_method,
                 "explanation_error": explanation_error},
            model_id=self.model_id, model_version=self.model_version,
            weights_hash=self.weights_hash, output_kind="box")


# --------------------------------------------------------------------------
# change head
# --------------------------------------------------------------------------


class ChangeHead(SpecialistBase):
    """Bi-temporal change via change-vector analysis (section 5.3).

    change-qa is derived from the same mask: class-wise area at t1 and t2,
    signed delta, then verbalised with a dead-band for 'unchanged'. This is
    more robust than generative answering and keeps the two outputs consistent
    by construction -- a fact about one implies a fact about the other.
    """

    model_id = "satquery-change"

    def run(self, tiles: list[Tile], slots: Slots,
            params: dict[str, Any]) -> TaskResult:
        t1, t2 = tiles[0], tiles[1]
        dead_band = params.get("dead_band", 0.02)
        cva_thresh = params.get("cva_thresh", 1.5)
        effective_intent = params.get("_effective_intent", "change")

        stats1 = _class_stats(t1)
        stats2 = _class_stats(t2)

        # -- change vector analysis on available index stack ------------
        ch_map = self._cva(t1, t2, cva_thresh)

        # -- per-class area delta ----------------------------------------
        deltas: dict[str, dict[str, Any]] = {}
        all_cls = set(stats1) | set(stats2)
        for cls in all_cls:
            f1 = stats1.get(cls, {}).get("fraction", 0.0)
            f2 = stats2.get(cls, {}).get("fraction", 0.0)
            delta = f2 - f1
            if abs(delta) < dead_band:
                verdict = "unchanged"
            elif delta > 0:
                verdict = "increased"
            else:
                verdict = "decreased"
            deltas[cls] = {
                "t1_pct": round(f1 * 100, 2),
                "t2_pct": round(f2 * 100, 2),
                "delta_pct": round(delta * 100, 2),
                "verdict": verdict,
            }

        changed_pct = float(100.0 * (ch_map > 0).sum() / max(t1.shape[0] * t1.shape[1], 1))
        gsd = t1.gsd
        changed_ha = float((ch_map > 0).sum() * gsd ** 2 / 10_000.0)

        # -- change-qa mode ---------------------------------------------
        if effective_intent == "change-qa":
            target = slots.target_class
            if target and target in deltas:
                d = deltas[target]
                text = (f"The {target.replace('_', ' ')} class "
                        f"{d['verdict']}: "
                        f"{d['t1_pct']:.1f}% -> {d['t2_pct']:.1f}% "
                        f"(delta {d['delta_pct']:+.1f} pp).")
            else:
                # pick the class with the largest absolute change
                if deltas:
                    top = max(deltas, key=lambda k: abs(deltas[k]["delta_pct"]))
                    d = deltas[top]
                    asked = target.replace("_", " ") if target else None
                    if asked:
                        text = (f"The {asked} class is not measurably "
                                f"present in this pair (below the 0.2% "
                                f"detection floor at either date), so its "
                                f"change cannot be quantified. For context, "
                                f"the largest measurable change is "
                                f"{top.replace('_', ' ')} {d['verdict']} by "
                                f"{abs(d['delta_pct']):.1f} pp "
                                f"({d['t1_pct']:.1f}% to {d['t2_pct']:.1f}%).")
                    else:
                        text = (f"Largest change: {top.replace('_', ' ')} "
                                f"{d['verdict']} by {abs(d['delta_pct']):.1f} pp "
                                f"({d['t1_pct']:.1f}% to {d['t2_pct']:.1f}%).")
                else:
                    text = "No measurable class-level change detected."
            conf = 0.81
            vis = Visual(kind="mask", mask=(ch_map * 255).astype(np.uint8),
                         mask_classes={"changed": 1},
                         legend=[{"class": "changed", "color": "#ef4444"}])
            return TaskResult(text=text, visual=vis, confidence=conf,
                              raw={"deltas": deltas, "changed_pct": round(changed_pct, 2),
                                   "changed_ha": round(changed_ha, 1),
                                   "interval_days": abs((d2 - d1).days)
                                   if "d1" in locals() and "d2" in locals() else None},
                              model_id=self.model_id, model_version=self.model_version,
                              weights_hash=self.weights_hash, output_kind="categorical")

        # -- change description mode ------------------------------------
        lines = []
        for cls, d in sorted(deltas.items(), key=lambda kv: -abs(kv[1]["delta_pct"])):
            if abs(d["delta_pct"]) >= 1.0:
                lines.append(f"{cls.replace('_', ' ')} {d['verdict']} "
                             f"({d['t1_pct']:.1f}% -> {d['t2_pct']:.1f}%)")
        interval_days = None
        if lines:
            interval = ""
            try:
                from datetime import datetime
                d1 = datetime.strptime(str(t1.ts)[:10], "%Y-%m-%d")
                d2 = datetime.strptime(str(t2.ts)[:10], "%Y-%m-%d")
                days = abs((d2 - d1).days)
                interval_days = days
                interval = f", {days / 30.44:.1f} months apart"
            except (TypeError, ValueError):
                pass
            text = (f"Bi-temporal pair ({t1.ts} -> {t2.ts}{interval}). "
                    "Class changes (dead-band "
                    f"{dead_band * 100:.0f}pp): " + "; ".join(lines[:4]) +
                    f". {changed_pct:.1f}% of the tile shows spectral change "
                    f"({changed_ha:.0f} ha).")
        else:
            text = (f"No change exceeding the {dead_band * 100:.0f}pp dead-band "
                    "detected between the two dates.")

        conf = min(0.88, 0.55 + 0.33 * min(changed_pct / 30.0, 1.0))
        vis = Visual(kind="mask", mask=(ch_map * 255).astype(np.uint8),
                     mask_classes={"changed": 1},
                     legend=[{"class": "changed", "color": "#ef4444"}])
        return TaskResult(text=text, visual=vis, confidence=conf,
                          raw={"deltas": deltas,
                               "changed_pct": round(changed_pct, 2),
                               "changed_ha": round(changed_ha, 1),
                               "t1_date": t1.ts, "t2_date": t2.ts,
                               "interval_days": interval_days},
                          model_id=self.model_id, model_version=self.model_version,
                          weights_hash=self.weights_hash, output_kind="mask")

    @staticmethod
    def _cva(t1: Tile, t2: Tile, thresh: float) -> np.ndarray:
        """Change vector analysis: L2 norm of the per-band difference."""
        common = sorted(set(t1.bands) & set(t2.bands))
        if not common:
            return np.zeros(t1.shape, dtype=np.float32)
        vecs = []
        for slot in common:
            a1 = t1.bands[slot].astype(np.float32)
            a2 = t2.bands[slot].astype(np.float32)
            vecs.append((a2 - a1) ** 2)
        mag = np.sqrt(sum(vecs) / len(vecs))
        p90 = float(np.percentile(mag, 90))
        if p90 < 1e-8:
            return np.zeros_like(mag)
        norm = mag / p90
        return (norm > thresh / 3.0).astype(np.float32)


# --------------------------------------------------------------------------
# fusion head
# --------------------------------------------------------------------------


class FuseHead(SpecialistBase):
    """Optical + SAR dual-source segmentation (section 5.3 fuse)."""

    model_id = "satquery-fuse"

    def run(self, tiles: list[Tile], slots: Slots,
            params: dict[str, Any]) -> TaskResult:
        sar_weight = params.get("sar_weight", 0.4)
        min_area = int(params.get("min_area_px", 32))

        opt = next((t for t in tiles if t.has_optical), None)
        sar = next((t for t in tiles if t.has_sar), None)

        if opt is None:
            return self._result(
                "No optical tile in pair; cannot produce a fusion result.",
                0.20, {}, output_kind="mask")
        if sar is None:
            return self._result(
                "No SAR tile in pair; fusion result is optical-only.",
                0.62, _class_stats(opt), output_kind="mask")

        opt_stats = _class_stats(opt)
        class_masks: dict[str, np.ndarray] = {}

        # Water: SAR dark return + NDWI consensus
        sar_water = _sar_water(sar)
        ndwi = _ndwi(opt)
        if sar_water is not None and ndwi is not None:
            opt_water = ndwi > 0.10
            class_masks["water"] = (
                (sar_weight * sar_water.astype(np.float32) +
                 (1 - sar_weight) * opt_water.astype(np.float32)) > 0.45)
        elif sar_water is not None:
            class_masks["water"] = sar_water
        elif ndwi is not None:
            class_masks["water"] = ndwi > 0.10

        # Built-up and vegetation from optical
        opt_masks = _classify(opt)
        for cls in ("built_up", "vegetation", "bare_soil", "snow_ice", "burn_scar"):
            if cls in opt_masks:
                class_masks[cls] = opt_masks[cls]

        h, w = opt.shape
        combined = np.zeros((h, w), dtype=np.uint8)
        CLASS_IDS = {"water": 1, "vegetation": 2, "built_up": 3,
                     "bare_soil": 4, "snow_ice": 5, "burn_scar": 6}
        for cls, m in class_masks.items():
            cid = CLASS_IDS.get(cls, 7)
            combined[m & (combined == 0)] = cid

        gsd = opt.gsd
        fracs: dict[str, Any] = {}
        legend = []
        colors = {"water": "#3b82f6", "vegetation": "#22c55e",
                  "built_up": "#94a3b8", "bare_soil": "#d97706",
                  "snow_ice": "#e2e8f0", "burn_scar": "#dc2626"}
        for cls, cid in CLASS_IDS.items():
            m = combined == cid
            if m.any():
                pct = float(100.0 * m.sum() / (h * w))
                ha = float(m.sum() * gsd ** 2 / 10_000.0)
                fracs[cls] = {"pct": round(pct, 2), "area_ha": round(ha, 1)}
                legend.append({"class": cls, "color": colors.get(cls, "#888"),
                                "id": cid})

        vis = Visual(kind="mask", mask=combined, mask_classes=CLASS_IDS,
                     legend=legend)

        # If the optical quality mask is poor, name the SAR contribution.
        sar_carried = (opt.clear_fraction < 0.60 and sar_water is not None)
        text = ("Optical+SAR fusion result. Classes detected: " +
                ", ".join(f"{k.replace('_', ' ')} {v['pct']:.1f}%"
                          for k, v in fracs.items()) + ".")
        if sar_carried:
            text += (f" SAR carried the water mask (optical clear fraction "
                     f"{opt.clear_fraction:.0%}).")
        # The trained optical and SAR land-cover heads each read their own
        # tile; agreement across modalities is real fusion evidence and
        # disagreement is named, not averaged away.
        opt_lulc = _neural_lulc(opt)
        sar_lulc = _neural_sar_lulc(sar) if sar is not None else None
        conf = 0.80 if sar is not None else 0.65
        raw = {"classes": fracs, "sar_carried_water": sar_carried,
               "sar_weight": sar_weight,
               "opt_lulc": opt_lulc, "sar_lulc": sar_lulc}
        if opt_lulc and sar_lulc:
            o_top = opt_lulc["top"][:2]
            s_top = sar_lulc["top"][:2]
            o_set = {l for l, _ in o_top}
            s_set = {l for l, _ in s_top}
            if o_set & s_set:
                common = o_set & s_set
                text += (f" Both trained heads agree on "
                         + ", ".join(l.lower() for l in sorted(common))
                         + f" (optical {o_top[0][1]:.2f} / SAR {s_top[0][1]:.2f}).")
                conf = min(0.92, conf + 0.07)
            else:
                text += (f" Optical head reads {o_top[0][0].lower()} "
                         f"({o_top[0][1]:.2f}); SAR head reads "
                         f"{s_top[0][0].lower()} ({s_top[0][1]:.2f}) -- the "
                         "modalities disagree, so the mask is the evidence.")
                conf = max(0.55, conf - 0.10)
            return self._neural_result(
                text, conf, raw, sar_lulc["model_id"],
                sar_lulc["weights_hash"], visual=vis, output_kind="mask")
        return TaskResult(text=text, visual=vis, confidence=conf,
                          raw=raw,
                          model_id=self.model_id, model_version=self.model_version,
                          weights_hash=self.weights_hash, output_kind="mask")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_HEADS: dict[str, SpecialistBase] = {}


def get_head(task: str) -> SpecialistBase:
    if not _HEADS:
        _HEADS.update({
            "vqa":       VQAHead(),
            "describe":  CaptionHead(),
            "ground":    GroundHead(),
            "change":    ChangeHead(),
            "change-qa": ChangeHead(),
            "fuse":      FuseHead(),
        })
    if task not in _HEADS:
        raise KeyError(f"no specialist for task '{task}'; registered: {sorted(_HEADS)}")
    return _HEADS[task]

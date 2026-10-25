"""Stage 2 - the trained heads, wired into the live query path.

Everything in ``models/`` that training produced is loaded here and made
callable from ``heads.py``:

    models/ben_txt_heads.pth   v2 (2026-09-11 joint run): VQA scorer v2
                               (MiniLM text, FiLM fusion), 19-class caption
                               head, S1 LULC head. Falls back to the v1
                               TF-IDF artifacts when the v2 blob or the
                               sentence-transformers package is absent.
    models/backbone_ft.pth     adapted encoder (via ``get_backbone``)

Why this module exists separately from ``heads.py``: the architectures
below must match the ones that were trained, byte for byte, but
``satquery/`` must never import from ``scripts/`` (the training tree is
not a package and is not shipped). The class definitions are restated
here and pinned by the state-dict shape assertions in ``load``.

Degradation contract
    Every loader returns ``None`` rather than raising when its artifact
    is absent, and ``NeuralStack.available`` reports the truth. A fresh
    clone with no ``models/`` directory keeps answering from the
    closed-form spectral path in ``heads.py`` exactly as before.

Honesty contract
    v1: the TF-IDF embedder was fitted on BigEarthNet.txt phrasing, so a
    live question of unseen words projected to a near-zero vector;
    ``text_coverage`` measured that and ``answer()`` refused rather than
    guessing. v2: MiniLM-L6-v2 embeds ANY phrasing meaningfully, so the
    coverage gate is retired in favour of the scorer's own margin: when
    the top option barely beats the runner-up the scorer has not decided,
    and ``answer()`` reports that instead of pretending. Reported val
    accuracy lives in the blob's ``val_metrics`` and is surfaced through
    ``describe()`` rather than hardcoded.
"""

from __future__ import annotations

import hashlib
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ..contracts import Tile

MODELS = Path(__file__).resolve().parents[2] / "models"
HEADS_PATH = MODELS / "ben_txt_heads.pth"
HEADS_V1_BAK = MODELS / "ben_txt_heads_v1.bak.pth"
EMBED_PATH = MODELS / "ben_txt_text_embed.pkl"

#: BigEarthNet-19 label vocabulary, in the order the caption head's output
#: layer was trained against (``data/ben_txt/manifest.json["labels"]``).
#: Order is load-bearing: index i of the 19-d logit vector is BEN19[i].
BEN19_LABELS: list[str] = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of "
    "natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

#: Maps the canonical slot classes the router emits
#: (``stage1/router.py:SUPPORTED_CLASSES``) onto the BigEarthNet-19 labels
#: the caption head predicts. A class present here can be decided by the
#: trained head; a class absent from BEN-19 entirely -- ``snow_ice`` and
#: ``burn_scar`` have no corresponding label -- maps to an empty set and is
#: handed back to the spectral path rather than forced into a wrong label.
CLASS_TO_BEN19: dict[str, frozenset[str]] = {
    "water": frozenset({
        "Inland waters", "Marine waters",
        "Coastal wetlands", "Inland wetlands"}),
    "vegetation": frozenset({
        "Broad-leaved forest", "Coniferous forest", "Mixed forest",
        "Transitional woodland, shrub", "Agro-forestry areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Natural grassland and sparsely vegetated areas"}),
    "built_up": frozenset({
        "Urban fabric", "Industrial or commercial units"}),
    "cropland": frozenset({
        "Arable land", "Permanent crops", "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas "
        "of natural vegetation"}),
    "bare_soil": frozenset({
        "Beaches, dunes, sands",
        "Natural grassland and sparsely vegetated areas"}),
    "snow_ice": frozenset(),
    "burn_scar": frozenset(),
}

#: How far the LULC probability must sit from 0.5 before the trained head
#: is treated as having decided. Inside this band the head is guessing and
#: the measured spectral answer is the better one.
PRESENCE_MARGIN = 0.15

#: When the scorer's top-option probability is this close to a coin flip,
#: it has not decided; ``answer()`` still reports, with ``decided=False``.
ANSWER_UNDECIDED_BAND = 0.14

#: How many encoded tiles to keep. Queries repeat on the same tile
#: constantly in the demo (one site, many questions) and the encoder is
#: the expensive part.
FEATURE_CACHE_SIZE = 64

#: Cache of MiniLM text embeddings (question phrasings repeat a lot).
TEXT_CACHE_SIZE = 4096

#: dB means the S1 head was trained against (train_joint.py /
#: train_ben_s1.py constants). Live SAR tiles arrive in dB from the
#: Stage-0 radiometry, so the same offsets apply.
S1_VV_MEAN, S1_VH_MEAN = -12.643863677978516, -19.352558135986328


# --------------------------------------------------------------------------
# architectures -- must match scripts/train_joint.py (v2) and
# scripts/train_ben_txt.py (v1)
# --------------------------------------------------------------------------


class VQAScorerV1(nn.Module):
    """v1 scorer: concat [q, f, o] into a 2-layer MLP (TF-IDF text).

    Mirrors ``scripts/train_ben_txt.py:VQAScorer``. The ``nn.Sequential``
    indices matter: the saved state dict has ``net.0`` and ``net.3``, so
    the GELU and Dropout must stay as positions 1 and 2 even though they
    carry no parameters.
    """

    def __init__(self, dim: int = 256, feat_dim: int = 2048,
                 hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + feat_dim + dim, hidden), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, q: torch.Tensor, f: torch.Tensor,
                o: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, f, o], -1)).squeeze(-1)


class VQAScorerV2(nn.Module):
    """v2 scorer: question-conditioned image modulation (FiLM).

    Mirrors ``scripts/train_joint.py:VQAScorerV2`` exactly -- layer names
    and order are load-bearing against the saved state dict.
    """

    def __init__(self, txt_dim: int = 384, feat_dim: int = 2048,
                 hidden: int = 512):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(txt_dim, hidden), nn.GELU())
        self.o_proj = nn.Sequential(nn.Linear(txt_dim, hidden), nn.GELU())
        self.img_proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.film = nn.Linear(hidden, hidden * 2)
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.LayerNorm(hidden * 2),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1))

    def forward(self, q: torch.Tensor, f: torch.Tensor,
                o: torch.Tensor) -> torch.Tensor:
        qe = self.q_proj(q)
        oe = self.o_proj(o)
        ie = self.img_proj(f)
        scale, shift = self.film(qe).chunk(2, dim=-1)
        mod = scale * ie + shift
        return self.net(torch.cat([qe, mod, oe], dim=-1)).squeeze(-1)


def _caption_head(n_labels: int = len(BEN19_LABELS),
                  feat_dim: int = 2048) -> nn.Sequential:
    """Mirrors the caption model (``train_ben_txt`` v1 + ``train_joint``)."""
    return nn.Sequential(
        nn.Linear(feat_dim, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, n_labels))


# --------------------------------------------------------------------------
# the stack
# --------------------------------------------------------------------------


class NeuralStack:
    """Lazy, cached, fail-soft holder for the trained heads."""

    model_id = "satquery-ben-txt"
    model_version = "2.0.0"

    def __init__(self) -> None:
        self.scorer: Optional[nn.Module] = None
        self.caption: Optional[nn.Sequential] = None
        self.s1_head: Optional[nn.Linear] = None
        self.text_mode: Optional[str] = None      # "minilm" | "tfidf"
        self.embed_dim: int = 256
        self.vectorizer: Any = None               # v1 only
        self.svd: Any = None                      # v1 only
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.weights_hash: Optional[str] = None
        self.val_metrics: dict[str, Any] = {}
        self.load_error: Optional[str] = None
        self._st: Any = None                      # sentence-transformers
        self._text_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._feat_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # The FastAPI endpoint is async and uvicorn may run handlers on a
        # worker thread; torch modules are not re-entrant across threads
        # for our purposes, so serialise the forward passes.
        self._lock = threading.Lock()
        self._loaded = False

    # -- loading ----------------------------------------------------------

    def load(self) -> "NeuralStack":
        if self._loaded:
            return self
        self._loaded = True
        try:
            if not (MODELS / "backbone_ft.pth").is_file():
                raise FileNotFoundError("the trained backbone checkpoint is unavailable")
            blob = self._load_blob()
            if blob is None:
                raise FileNotFoundError(f"{HEADS_PATH} not found")
            if blob.get("arch") == "v2":
                self._load_v2(blob)
            else:
                self._load_v1(blob)
            self.weights_hash = _sha(HEADS_PATH)
        except Exception as e:                                  # noqa: BLE001
            self.load_error = f"{type(e).__name__}: {e}"
            self._clear()
            print(f"[neural] trained heads unavailable ({self.load_error}); "
                  "falling back to the closed-form spectral path")
        return self

    def _load_blob(self) -> Optional[dict]:
        if HEADS_PATH.exists():
            return torch.load(HEADS_PATH, map_location=self.device,
                              weights_only=True)
        # the v2 promotion backs v1 up here; a rollback restores it to
        # ben_txt_heads.pth, but until then use it as-is
        if HEADS_V1_BAK.exists():
            print(f"[neural] {HEADS_PATH.name} missing; trying the v1 "
                  f"backup {HEADS_V1_BAK.name}")
            return torch.load(HEADS_V1_BAK, map_location=self.device,
                              weights_only=True)
        return None

    def _load_v2(self, blob: dict) -> None:
        dim = int(blob.get("embed_dim", 384))
        scorer = VQAScorerV2(txt_dim=dim)
        scorer.load_state_dict(blob["vqa_scorer"])
        self.scorer = scorer.to(self.device).eval()

        caption = _caption_head()
        caption.load_state_dict(blob["caption_head"])
        self.caption = caption.to(self.device).eval()

        s1_head = nn.Linear(2048, len(BEN19_LABELS))
        s1_head.load_state_dict(blob["s1_head"])
        self.s1_head = s1_head.to(self.device).eval()

        self.embed_dim = dim
        self.text_mode = "minilm"
        self.val_metrics = dict(blob.get("val_metrics") or {})
        self.model_id = "satquery-joint-v2"
        # warm the sentence-transformers cache right here so the first
        # live query does not pay the model load
        self._sentence_model()

    def _load_v1(self, blob: dict) -> None:
        if not EMBED_PATH.exists():
            raise FileNotFoundError(f"{EMBED_PATH} not found")
        dim = int(blob.get("embed_dim", 256))
        scorer = VQAScorerV1(dim)
        scorer.load_state_dict(blob["vqa_scorer"])
        self.scorer = scorer.to(self.device).eval()

        caption = _caption_head()
        caption.load_state_dict(blob["caption_head"])
        self.caption = caption.to(self.device).eval()

        with open(EMBED_PATH, "rb") as f:
            emb = pickle.load(f)
        self.vectorizer, self.svd = emb["vectorizer"], emb["svd"]
        self.embed_dim = dim
        self.text_mode = "tfidf"
        self.model_id = "satquery-ben-txt"

    def _clear(self) -> None:
        self.scorer = self.caption = self.s1_head = None
        self.text_mode = None
        self.vectorizer = self.svd = None

    @property
    def available(self) -> bool:
        self.load()
        return self.scorer is not None and self.text_mode is not None

    # -- text -------------------------------------------------------

    def _sentence_model(self):
        """sentence-transformers MiniLM, GPU first, CPU fallback."""
        if self._st is not None:
            return self._st
        from sentence_transformers import SentenceTransformer
        local_model = MODELS / "minilm"
        name = str(local_model) if local_model.is_dir() else "sentence-transformers/all-MiniLM-L6-v2"
        options = {"local_files_only": True} if local_model.is_dir() else {}
        try:
            self._st = SentenceTransformer(name, device=str(self.device), **options)
        except Exception:                                       # noqa: BLE001
            self._st = SentenceTransformer(name, device="cpu", **options)
        return self._st

    def text_coverage(self, text: str) -> int:
        """How much the embedder understands of ``text``.

        v1 TF-IDF: the stored nnz of the transformed row -- unseen words
        project to a zero vector, so low coverage disqualifies the scorer.
        v2 MiniLM: any phrasing embeds meaningfully, so coverage is always
        full; the scorer's own margin now carries the uncertainty.
        """
        if self.text_mode == "minilm":
            return 999
        if self.vectorizer is None:
            return 0
        try:
            return int(self.vectorizer.transform([text]).nnz)
        except Exception:                                       # noqa: BLE001
            return 0

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.text_mode == "minilm":
            missing = [t for t in texts if t not in self._text_cache]
            if missing:
                st = self._sentence_model()
                vecs = np.asarray(st.encode(
                    missing, batch_size=64,
                    show_progress_bar=False), dtype=np.float32)
                for t, v in zip(missing, vecs):
                    self._text_cache[t] = v
                    while len(self._text_cache) > TEXT_CACHE_SIZE:
                        self._text_cache.popitem(last=False)
            return np.stack([self._text_cache[t] for t in texts])
        X = self.vectorizer.transform(texts)
        return self.svd.transform(X).astype(np.float32)

    # -- image ------------------------------------------------------

    def features(self, tile: Tile) -> Optional[np.ndarray]:
        """(2048,) encoder feature for ``tile``, cached by content hash."""
        key = tile.content_hash()
        if key in self._feat_cache:
            self._feat_cache.move_to_end(key)
            return self._feat_cache[key]
        try:
            from .backbone import get_backbone
            bb = get_backbone()
            with torch.no_grad():
                f = bb.encode_features(tile).squeeze(0).float().cpu().numpy()
        except Exception as e:                                  # noqa: BLE001
            print(f"[neural] encode failed for tile {tile.id}: "
                  f"{type(e).__name__}: {e}")
            return None
        if not np.all(np.isfinite(f)):
            print(f"[neural] non-finite feature for tile {tile.id}; skipping")
            return None
        self._feat_cache[key] = f
        while len(self._feat_cache) > FEATURE_CACHE_SIZE:
            self._feat_cache.popitem(last=False)
        return f

    # -- inference --------------------------------------------------

    def answer(self, tile: Tile, question: str,
               options: list[str]) -> Optional[dict[str, Any]]:
        """Rank ``options`` for ``question`` over ``tile``.

        Returns ``None`` -- meaning "ask the spectral path instead" --
        when the stack is unavailable or the tile will not encode. With
        MiniLM there is no out-of-vocabulary refusal; a scorer that has
        not decided reports ``decided=False`` and a margin near zero, and
        the caller weighs it accordingly.
        """
        if not self.available or len(options) < 2:
            return None
        cov = self.text_coverage(question)
        if cov < 2:
            return None

        feat = self.features(tile)
        if feat is None:
            return None

        with self._lock, torch.no_grad():
            qe = self._embed([question])
            oe = self._embed(options)
            q = torch.from_numpy(qe).repeat(len(options), 1).to(self.device)
            f = torch.from_numpy(feat).repeat(len(options), 1).to(self.device)
            o = torch.from_numpy(oe).to(self.device)
            logits = self.scorer(q, f, o)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        idx = int(np.argmax(probs))
        order = np.sort(probs)[::-1]
        margin = float(order[0] - order[1]) if len(order) > 1 else 1.0
        return {
            "index": idx,
            "answer": options[idx],
            "prob": float(probs[idx]),
            "margin": margin,
            "decided": bool(margin >= ANSWER_UNDECIDED_BAND),
            "probs": {o: round(float(p), 4) for o, p in zip(options, probs)},
            "text_terms": cov,
            "embedder": self.text_mode,
            "model_id": self.model_id,
            "weights_hash": self.weights_hash,
        }

    def lulc(self, tile: Tile, top_k: int = 4,
             thresh: float = 0.5) -> Optional[dict[str, Any]]:
        """19-class BigEarthNet land-cover probabilities for ``tile``.

        The head is multi-label (sigmoid + BCE), so the probabilities do
        not sum to 1 and ``above_threshold`` can be empty -- which is a
        real answer, not a failure.
        """
        if self.caption is None:
            self.load()
        if self.caption is None:
            return None

        feat = self.features(tile)
        if feat is None:
            return None

        with self._lock, torch.no_grad():
            x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
            p = torch.sigmoid(self.caption(x)).squeeze(0).cpu().numpy()

        order = np.argsort(p)[::-1]
        return {
            "top": [(BEN19_LABELS[i], round(float(p[i]), 4))
                    for i in order[:top_k]],
            "above_threshold": [(BEN19_LABELS[i], round(float(p[i]), 4))
                                for i in order if p[i] >= thresh],
            "model_id": self.model_id,
            "weights_hash": self.weights_hash,
        }

    def presence(self, tile: Tile,
                 target_class: str) -> Optional[dict[str, Any]]:
        """Decide whether ``target_class`` is present, from the LULC head.

        This is the trained path that actually survives the jump from
        120 px BigEarthNet patches to 512 px Sentinel-2 mission tiles.
        The VQA scorer alone does not: it was fitted on the joint
        (question, patch) distribution and the adjacency phrasing that
        dominates BigEarthNet.txt. The caption head reads the image
        feature alone and stays sharp -- 0.99 water on the Sundarbans,
        0.97 urban on Delhi.

        Returns ``None`` when the class has no BEN-19 equivalent, or when
        the head's probability sits inside ``PRESENCE_MARGIN`` of 0.5 and
        it has therefore not decided. Both hand the question back to the
        measured spectral answer.
        """
        labels = CLASS_TO_BEN19.get(target_class)
        if not labels:
            return None

        out = self.lulc(tile, top_k=len(BEN19_LABELS))
        if out is None:
            return None

        probs = dict(out["top"])
        matched = {k: v for k, v in probs.items() if k in labels}
        if not matched:
            return None

        best_label = max(matched, key=lambda k: matched[k])
        p = matched[best_label]
        if abs(p - 0.5) < PRESENCE_MARGIN:
            return None

        return {
            "present": p >= 0.5,
            "prob": float(p),
            "label": best_label,
            "class": target_class,
            "evidence": sorted(matched.items(), key=lambda kv: -kv[1]),
            "model_id": self.model_id,
            "weights_hash": self.weights_hash,
        }

    # -- SAR ----------------------------------------------------------

    def sar_lulc(self, tile: Tile, top_k: int = 4,
                 thresh: float = 0.5) -> Optional[dict[str, Any]]:
        """19-class land-cover probabilities from the S1 (SAR) side.

        The S1 head was trained jointly (train_joint.py) on BigEarthNet v2
        VV/VH pairs, so this works on cloudy scenes where optical answers
        degrade. Bands arrive in dB from the Stage-0 radiometry; the same
        dB means the head was trained against are subtracted here. The
        recipe mirrors training exactly:
        120 px resize -> dB mean removal -> sar_enc -> gsd_cond(10) -> head.
        """
        if self.s1_head is None:
            self.load()
        if self.s1_head is None:
            return None
        vv = tile.bands.get("VV", tile.bands.get("HH"))
        vh = tile.bands.get("VH", tile.bands.get("HV"))
        if vv is None:
            return None
        try:
            vv_t = torch.from_numpy(
                np.ascontiguousarray(vv, dtype=np.float32))[None, None]
            vh_t = (torch.from_numpy(
                np.ascontiguousarray(vh, dtype=np.float32))[None, None]
                if vh is not None else vv_t.clone())
            x = torch.cat([vv_t, vh_t], dim=1)
            if x.shape[-2:] != (120, 120):
                x = torch.nn.functional.interpolate(
                    x, size=(120, 120), mode="bilinear",
                    align_corners=False)
            x = x.to(self.device)
            from .backbone import get_backbone
            bb = get_backbone()
            with self._lock, torch.no_grad():
                f = bb.sar_enc(x[:, 0] - S1_VV_MEAN, x[:, 1] - S1_VH_MEAN)
                f = bb.gsd_cond(f, gsd=10.0)
                p = torch.sigmoid(self.s1_head(f)).squeeze(0).cpu().numpy()
        except Exception as e:                                  # noqa: BLE001
            print(f"[neural] sar_lulc failed for tile {tile.id}: "
                  f"{type(e).__name__}: {e}")
            return None
        order = np.argsort(p)[::-1]
        return {
            "top": [(BEN19_LABELS[i], round(float(p[i]), 4))
                    for i in order[:top_k]],
            "above_threshold": [(BEN19_LABELS[i], round(float(p[i]), 4))
                                for i in order if p[i] >= thresh],
            "model_id": f"{self.model_id}-s1",
            "weights_hash": self.weights_hash,
        }

    def sar_presence(self, tile: Tile,
                     target_class: str) -> Optional[dict[str, Any]]:
        """Same contract as ``presence`` but reading the SAR head."""
        labels = CLASS_TO_BEN19.get(target_class)
        if not labels:
            return None
        out = self.sar_lulc(tile, top_k=len(BEN19_LABELS))
        if out is None:
            return None
        probs = dict(out["top"])
        matched = {k: v for k, v in probs.items() if k in labels}
        if not matched:
            return None
        best_label = max(matched, key=lambda k: matched[k])
        p = matched[best_label]
        if abs(p - 0.5) < PRESENCE_MARGIN:
            return None
        return {
            "present": p >= 0.5,
            "prob": float(p),
            "label": best_label,
            "class": target_class,
            "evidence": sorted(matched.items(), key=lambda kv: -kv[1]),
            "model_id": f"{self.model_id}-s1",
            "weights_hash": self.weights_hash,
        }

    # -- reporting ------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """What ``/health`` shows so the demo can prove the weights are live."""
        self.load()
        from .backbone import get_backbone
        vm = self.val_metrics or {}
        out: dict[str, Any] = {
            "vqa_scorer": {
                "artifact": str(HEADS_PATH.relative_to(MODELS.parent)),
                "loaded": self.scorer is not None,
                "arch": "v2" if self.text_mode == "minilm" else self.text_mode,
                "weights_hash": self.weights_hash,
                "val_accuracy": vm.get("vqa_acc"),
                "s2_exact": vm.get("s2_exact"),
                "s1_exact": vm.get("s1_exact"),
                "euro_acc": vm.get("euro_acc"),
            },
            "caption_head": {
                "artifact": str(HEADS_PATH.relative_to(MODELS.parent)),
                "loaded": self.caption is not None,
                "labels": len(BEN19_LABELS),
            },
            "s1_head": {
                "artifact": str(HEADS_PATH.relative_to(MODELS.parent)),
                "loaded": self.s1_head is not None,
                "labels": len(BEN19_LABELS),
            },
            "text_embedder": {
                "mode": self.text_mode,
                "dim": self.embed_dim if self.text_mode else None,
            },
        }
        if self.load_error:
            out["load_error"] = self.load_error
        try:
            if not (MODELS / "backbone_ft.pth").is_file():
                raise FileNotFoundError("the trained backbone checkpoint is unavailable")
            bb = get_backbone()
            out["backbone"] = {
                "artifact": getattr(bb, "checkpoint_path", None),
                "loaded": getattr(bb, "checkpoint_hash", None) is not None,
                "weights_hash": getattr(bb, "checkpoint_hash", None),
                "backbone_id": getattr(bb, "_backbone_id", None),
            }
        except Exception as e:                                  # noqa: BLE001
            out["backbone"] = {"loaded": False, "error": str(e)}
        return out


def _sha(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


_STACK: Optional[NeuralStack] = None


def get_neural() -> NeuralStack:
    """Return the process-wide trained-head stack, loading once."""
    global _STACK
    if _STACK is None:
        _STACK = NeuralStack().load()
    return _STACK

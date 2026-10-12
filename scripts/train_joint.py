"""Joint multi-task fine-tune: unfrozen optical + SAR encoders, MiniLM text.

The accuracy push of 2026-09-11. Everything trains in ONE loop on the
BigEarthNet.txt S2 + S1 data already on disk, plus an EuroSAT replay
stream:

    VQA (S2)      binary + MCQ questions through the UNFROZEN optical
                  encoder (band_proj -> opt_enc -> gsd_cond -> scorer).
                  The old run froze the encoder and fitted the scorer on
                  cached features; the encoder never saw a VQA gradient.
    LULC (S2)     BigEarthNet-19 multi-label on S2 patches. The caption
                  head architecture is unchanged from v1 so neural.py
                  keeps loading it.
    LULC (S1)     BigEarthNet-19 multi-label on Sentinel-1 VV/VH through
                  the UNFROZEN sar_enc (dB-normalised, gsd_cond applied
                  so the inference recipe matches).
    EuroSAT       10-class replay so the 97.78% fine-tune is not
                  catastrophically forgotten by the multi-task losses.

Design decisions that matter:
    - MiniLM-L6-v2 (384-d) replaces TF-IDF+SVD: unseen question phrasings
      still embed meaningfully, which is what the zero-shot benchmarks
      and live free-form queries need.
    - scorer v2: question-conditioned image modulation (FiLM) + option
      match, ~2M params, trained jointly with the encoder.
    - patch-level val split. The previous random question split leaked
      the same patch's questions across folds; honest numbers may read
      lower even as the model is better.
    - band-dropout augmentation: a fraction of S2 batches keep only a
      random 3-8 band subset and let BandProjector interpolate the rest
      -- the exact input mode the RGB-only benchmarks produce.
    - layer-wise LR decay over the ResNet stages; cosine schedule with
      warmup; fp16 autocast + GradScaler.
    - per-epoch checkpoints in models/joint_ckpt/ (weights only) plus a
      rolling latest_full.pth with optimizer state for --resume. The
      best epoch is promoted to models/backbone_ft.pth +
      models/ben_txt_heads.pth only at the end; the v1 artifacts are
      backed up first.

Usage:
    python scripts/train_joint.py --smoke          # sanity run
    python scripts/train_joint.py                  # full run
    python scripts/train_joint.py --resume         # continue after a crash
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BEN = ROOT / "data" / "ben_txt"
PATCH_DIR = BEN / "patches"
S1_DIR = BEN / "s1_patches"
MANIFEST = BEN / "manifest.json"
ANN = BEN / "annotations.parquet"
NAME_MAP = BEN / "s1_name_map.json"
CKPT_DIR = ROOT / "models" / "joint_ckpt"
BB_CKPT = ROOT / "models" / "backbone_ft.pth"
HEADS_OUT = ROOT / "models" / "ben_txt_heads.pth"
METRICS_OUT = ROOT / "data" / "joint_train_metrics.json"
EMB_CACHE = BEN / "minilm_vqa_emb.npz"

from train_ben_txt import BAND_TO_SLOT, SLOT_ORDER, parse_vqa_example  # noqa: E402
from train_backbone import EUROSAT_CLASSES, SLOT_FROM_EUROSAT_IDX     # noqa: E402

# dB means for Sentinel-1 (BigEarthNet v2 interpolated patches); the same
# constants train_ben_s1.py used and the live SAR ingest expects.
S1_VV_MEAN, S1_VH_MEAN = -12.643863677978516, -19.352558135986328

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TXT_DIM = 384
NEG = -1e4          # option padding logit (finite: label smoothing must
                    # not meet -inf, which makes the loss NaN)

# band-dropout keep-sets that mimic the RGB-only benchmarks
_DROP_SETS = [
    ("red", "green", "blue"),
    ("red", "green", "blue", "nir"),
    ("red", "green", "blue", "nir", "swir1", "swir2"),
]


# --------------------------------------------------------------------------
# scorer v2
# --------------------------------------------------------------------------


class VQAScorerV2(nn.Module):
    """Score a (question, image, option) triple -> one logit.

    The question conditions the image feature through FiLM (scale+shift)
    and the option is matched against the modulated feature. The state
    dict layout is fixed by these names -- satquery/stage2/neural.py
    mirrors this class exactly.
    """

    def __init__(self, txt_dim: int = TXT_DIM, feat_dim: int = 2048,
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


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------


def preload_s2(pids: list[str]) -> dict[str, np.ndarray]:
    """patch_id -> (12, 120, 120) float16 band stack.

    float16 halves RAM (337 KB -> 172 KB per patch) with quantization
    error far below reflectance noise, which is what lets the ~16k-patch
    pass-2 set fit beside the OS in 16 GB.
    """
    out: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, pid in enumerate(pids):
        p = PATCH_DIR / f"{pid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        out[pid] = np.ascontiguousarray(
            np.stack([z[b] for b in BAND_TO_SLOT]).astype(np.float16))
        if (i + 1) % 1000 == 0:
            print(f"  s2 preload {i + 1}/{len(pids)} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"S2 preloaded: {len(out)} patches, "
          f"{sum(a.nbytes for a in out.values()) / 1e9:.2f} GB (fp16)",
          flush=True)
    return out


def preload_s1(pids: list[str],
               name_map: dict[str, str]) -> dict[str, np.ndarray]:
    """patch_id -> (2, 120, 120) float32 dB stack."""
    out: dict[str, np.ndarray] = {}
    for pid in pids:
        name = name_map.get(pid)
        if not name:
            continue
        p = S1_DIR / f"{name}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        out[pid] = np.ascontiguousarray(np.stack(
            [z["VV"].astype(np.float32), z["VH"].astype(np.float32)]))
    print(f"S1 preloaded: {len(out)} patches "
          f"({sum(a.nbytes for a in out.values()) / 1e9:.2f} GB)", flush=True)
    return out


def parse_examples(pids: set[str]):
    """VQA (binary+mcq) train annotations over the given patches.

    Returns (examples, q_patch_ids) in the same iteration order.
    """
    df = pd.read_parquet(ANN)
    df = df[(df["split"] == "train") & df["type"].isin(["binary", "mcq"])]
    examples, q_patch_ids = [], []
    for _, r in df.iterrows():
        pid = str(r["patch_id"])
        if pid not in pids:
            continue
        ex = parse_vqa_example(r)
        if ex is None:
            continue
        examples.append(ex)
        q_patch_ids.append(pid)
    return examples, q_patch_ids


def build_text_embeddings(examples, device: str):
    """MiniLM embeddings for question stems + options, cached on disk."""
    n_opts = sum(len(e[1]) for e in examples)
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE)
        if (z["q"].shape == (len(examples), TXT_DIM)
                and z["o"].shape == (n_opts, TXT_DIM)):
            print(f"MiniLM embeddings cached: q {z['q'].shape}, "
                  f"o {z['o'].shape}", flush=True)
            return z["q"], z["o"]
        print("MiniLM cache stale; rebuilding", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(TEXT_MODEL, device=device)
    q_texts = [e[0] for e in examples]
    o_texts = [o for e in examples for o in e[1]]
    t0 = time.time()
    q = enc.encode(q_texts, batch_size=256, show_progress_bar=False)
    o = enc.encode(o_texts, batch_size=256, show_progress_bar=False)
    print(f"MiniLM embedded {len(q_texts)} questions + {len(o_texts)} "
          f"options in {time.time()-t0:.0f}s", flush=True)
    q = np.asarray(q, dtype=np.float32)
    o = np.asarray(o, dtype=np.float32)
    np.savez(EMB_CACHE, q=q, o=o, model=TEXT_MODEL)
    return q, o


# --------------------------------------------------------------------------
# encoder helpers
# --------------------------------------------------------------------------


def s2_features(bb, patches: list[np.ndarray], device, gsd: float = 10.0,
                band_dropout: float = 0.0, rgb_stretch: float = 0.0,
                rng=None) -> torch.Tensor:
    """(B,12,120,120) patches -> (B, 2048) features.

    Three input modes are sampled PER BATCH (a batch is one stacked
    tensor, so every patch in it shares the mode):

      full       -- all 12 bands at the native 120 px reflectance.
      band-drop  -- a random 3-8 band subset at 120 px; BandProjector
                    interpolates the rest (partial live tiles).
      bench      -- RGB only, resized to 224 px, 1-99 percentile
                    stretched to [0,1] -- the exact preprocessing
                    eval_benchmarks._load_rgb applies to the RGB-only
                    public benchmarks. Without seeing this mode in
                    training the encoder+scorer read benchmark images
                    off-distribution: measured on the bench split,
                    64.6% (reflectance recipe) vs 51.2% (benchmark
                    recipe) with the same weights, which is where the
                    zero-shot regression of the first pass came from.
    """
    x = torch.from_numpy(np.stack(patches)).to(device).float() / 10000.0
    bands = {s: x[:, i] for i, s in enumerate(SLOT_ORDER)}
    mode = "full"
    if rgb_stretch > 0 and rng.random() < rgb_stretch:
        mode = "bench"
    elif band_dropout > 0 and rng.random() < band_dropout:
        mode = "band_drop"
    if mode == "bench":
        rgb = torch.cat([x[:, SLOT_ORDER.index(s):SLOT_ORDER.index(s) + 1]
                         for s in ("red", "green", "blue")], dim=1)
        rgb = torch.nn.functional.interpolate(
            rgb, size=(224, 224), mode="bilinear", align_corners=False)
        lo = torch.quantile(rgb.reshape(len(rgb), -1), 0.01, dim=1)
        hi = torch.quantile(rgb.reshape(len(rgb), -1), 0.99, dim=1)
        rgb = ((rgb - lo[:, None, None, None])
               / (hi - lo).clamp(min=1e-3)[:, None, None, None]).clamp(0, 1)
        bands = {"red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2]}
    elif mode == "band_drop":
        m = rng.randint(0, 5)
        if m < len(_DROP_SETS):
            keep = set(_DROP_SETS[int(m)])
        else:
            keep = set(np.asarray(
                rng.choice(SLOT_ORDER, size=int(rng.randint(4, 9)),
                           replace=False)).tolist())
        bands = {s: v for s, v in bands.items() if s in keep}
    x12 = bb.band_proj(bands)
    f = bb.opt_enc(x12).flatten(1)
    return bb.gsd_cond(f, gsd=gsd)


def s1_features(bb, patches: list[np.ndarray], device) -> torch.Tensor:
    """(B,2,120,120) dB patches -> (B, 2048) gsd-conditioned SAR features."""
    x = torch.from_numpy(np.stack(patches)).to(device)
    vv = x[:, 0] - S1_VV_MEAN
    vh = x[:, 1] - S1_VH_MEAN
    f = bb.sar_enc(vv, vh)
    return bb.gsd_cond(f, gsd=10.0)


def eurosat_forward(bb, euro_head, x13: torch.Tensor) -> torch.Tensor:
    from satquery.stage2.backbone import BACKBONE_SLOTS
    x13 = x13.float() / 10000.0
    bands = {}
    for src_idx, slot_idx in SLOT_FROM_EUROSAT_IDX.items():
        if slot_idx is None:
            continue
        bands[BACKBONE_SLOTS[slot_idx]] = x13[:, src_idx]
    f = bb.opt_enc(bb.band_proj(bands)).flatten(1)
    f = bb.gsd_cond(f, gsd=10.0)
    return euro_head(f)


# --------------------------------------------------------------------------
# optimizer param groups (layer-wise LR decay over ResNet stages)
# --------------------------------------------------------------------------


def param_groups(bb, scorer, cap_head, s1_head, euro_head,
                enc_lr: float, head_lr: float):
    stage_lr = {".0.": 0.5, ".1.": 0.5, ".2.": 0.5, ".3.": 0.5,
                ".4.": 0.6, ".5.": 0.8, ".6.": 1.0, ".7.": 1.4,
                ".8.": 1.0, ".9.": 1.0}
    groups: list[dict] = []

    def add(params, lr, name):
        if params:
            groups.append({"params": params, "lr": lr, "name": name})

    for enc_name in ("opt_enc", "sar_enc"):
        by_stage: dict[str, list] = defaultdict(list)
        for n, p in bb.named_parameters():
            if not n.startswith(enc_name):
                continue
            key = next((k for k in stage_lr if f"{enc_name}{k}" in n), ".9.")
            by_stage[key].append(p)
        for key, params in sorted(by_stage.items()):
            add(params, enc_lr * stage_lr[key], f"{enc_name}{key}")

    add(list(bb.band_proj.parameters()), enc_lr, "band_proj")
    add(list(bb.gsd_cond.parameters()), head_lr * 0.2, "gsd_cond")
    add(list(scorer.parameters()), head_lr, "scorer")
    add(list(cap_head.parameters()), head_lr, "cap_head")
    add(list(s1_head.parameters()), head_lr, "s1_head")
    add(list(euro_head.parameters()), head_lr, "euro_head")
    return groups


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--vqa-patch-batch", type=int, default=6)
    ap.add_argument("--lulc-batch", type=int, default=16)
    ap.add_argument("--euro-batch", type=int, default=16)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--val-pids-file", type=str, default="",
                    help="JSON list of patch ids to hold out as val "
                    "(reuses a previous pass's split so a warm-started "
                    "model is never validated on patches it trained on)")
    ap.add_argument("--init-from", type=str, default="",
                    help="warm-start bb+heads from a checkpoint "
                    "(e.g. models/joint_ckpt/best.pth) without optimizer "
                    "state -- used for pass 2")
    ap.add_argument("--band-dropout", type=float, default=0.25)
    ap.add_argument("--rgb-stretch", type=float, default=0.15,
                    help="fraction of batches in benchmark mode: RGB "
                    "only, 224 px, 1-99 percentile stretch")
    ap.add_argument("--qcap", type=int, default=160)
    ap.add_argument("--limit-patches", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 300 patches, 1 epoch, no artifact writes")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-eurosat", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.limit_patches = 300
        args.epochs = 1
        args.band_dropout = 0.25

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda"
             else ""), flush=True)

    # ---- model ----------------------------------------------------------
    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone().to(device)
    old_sd = None
    init_state = None
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            print(f"--init-from {init_path} not found; aborting", flush=True)
            return 1
        init_state = torch.load(init_path, map_location=device,
                                weights_only=False)
        bb.load_state_dict(init_state["bb"], strict=False)
        print(f"warm-started encoder+heads from {init_path}", flush=True)
    elif BB_CKPT.exists():
        old_sd = torch.load(BB_CKPT, map_location=device, weights_only=True)
        bb.load_state_dict(old_sd, strict=False)
        print("warm-started from backbone_ft.pth (EuroSAT + BEN S1/S2)",
              flush=True)

    scorer = VQAScorerV2().to(device)
    cap_head = nn.Sequential(            # caption head, arch unchanged from v1
        nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 19)).to(device)
    s1_head = nn.Linear(2048, 19).to(device)
    euro_head = nn.Linear(2048, len(EUROSAT_CLASSES)).to(device)

    if init_state is not None:
        scorer.load_state_dict(init_state["scorer"])
        cap_head.load_state_dict(init_state["cap_head"])
        s1_head.load_state_dict(init_state["s1_head"])
        euro_head.load_state_dict(init_state["euro_head"])
        print("all four heads warm-started from the pass checkpoint",
              flush=True)
    elif old_sd is not None:
        def grab(prefix: str):
            cut = len(prefix)
            return {k[cut:]: v for k, v in old_sd.items()
                    if k.startswith(prefix)}
        w = grab("ft_cls_head.")
        # the BEN S2 fine-tune repurposed ft_cls_head for 19 classes; only
        # warm-start the 10-class EuroSAT replay head when shapes match
        if ("weight" in w and tuple(w["weight"].shape)
                == tuple(euro_head.weight.shape)):
            euro_head.weight.data.copy_(w["weight"])
            euro_head.bias.data.copy_(w["bias"])
            print("euro head warm-started from ft_cls_head", flush=True)
        else:
            print("ft_cls_head in checkpoint is not 10-class; EuroSAT "
                  "head starts fresh (replay trains it)", flush=True)
        w = grab("s1_cls_head.")
        if "weight" in w:
            s1_head.weight.data.copy_(w["weight"])
            s1_head.bias.data.copy_(w["bias"])
            print("s1 head warm-started from s1_cls_head", flush=True)
        if HEADS_OUT.exists():
            try:
                v1 = torch.load(HEADS_OUT, map_location=device,
                                weights_only=True)
                cap_head.load_state_dict(v1["caption_head"])
                print("caption head warm-started from ben_txt_heads.pth",
                      flush=True)
            except Exception as e:                              # noqa: BLE001
                print(f"caption warm-start skipped ({e})", flush=True)

    # ---- data -----------------------------------------------------------
    manifest = json.loads(MANIFEST.read_text())
    label_vocab = manifest["labels"]
    name_map = json.loads(NAME_MAP.read_text())

    train_rows = [r for r in manifest["patches"] if r["split"] == "train"
                  and (PATCH_DIR / f"{r['patch_id']}.npz").exists()]
    if args.limit_patches:
        train_rows = train_rows[:args.limit_patches]
    if args.val_pids_file:
        val_pids = set(json.loads(Path(args.val_pids_file).read_text()))
        val_rows = [r for r in train_rows if r["patch_id"] in val_pids]
        tr_rows = [r for r in train_rows if r["patch_id"] not in val_pids]
        print(f"pinned val split ({args.val_pids_file}): "
              f"{len(val_rows)} val | {len(tr_rows)} train patches "
              f"({len(val_pids) - len(val_rows)} pids not on disk)",
              flush=True)
    else:
        rng_split = np.random.RandomState(123)
        train_rows = list(train_rows)
        rng_split.shuffle(train_rows)
        n_val = 40 if args.smoke else max(
            50, int(args.val_frac * len(train_rows)))
        val_rows, tr_rows = train_rows[:n_val], train_rows[n_val:]
        print(f"train patches: {len(tr_rows)} | val patches: {n_val}",
              flush=True)

    all_pids = [r["patch_id"] for r in train_rows]
    s2 = preload_s2(all_pids)
    s1 = preload_s1(all_pids, name_map)

    examples, q_pids = parse_examples(set(all_pids))
    print(f"VQA examples (train split, on-disk patches): {len(examples)}",
          flush=True)
    q_emb, o_emb = build_text_embeddings(examples, device)

    pid_list = [p for p in all_pids if p in s2]
    pid_idx = {p: i for i, p in enumerate(pid_list)}
    pid_to_row = {r["patch_id"]: r for r in train_rows}

    gold = np.array([e[2] for e in examples], dtype=np.int64)
    n_opts = np.array([len(e[1]) for e in examples], dtype=np.int64)
    opt_start = np.concatenate([[0], np.cumsum(n_opts)])[:-1]
    patch_of_q = np.array([pid_idx[p] for p in q_pids], dtype=np.int64)

    tr_patch_ix = np.array([pid_idx[r["patch_id"]] for r in tr_rows
                            if r["patch_id"] in pid_idx], dtype=np.int64)
    val_patch_ix = np.array([pid_idx[r["patch_id"]] for r in val_rows
                             if r["patch_id"] in pid_idx], dtype=np.int64)

    q_by_patch: dict[int, list[int]] = defaultdict(list)
    for qi, pi in enumerate(patch_of_q):
        q_by_patch[int(pi)].append(qi)
    # VQA stream iterates only patches that carry questions
    vqa_tr_ix = np.array([p for p in tr_patch_ix
                          if q_by_patch.get(int(p))], dtype=np.int64)
    vqa_val_ix = np.array([p for p in val_patch_ix
                           if q_by_patch.get(int(p))], dtype=np.int64)
    print(f"patches with VQA questions: train {len(vqa_tr_ix)}, "
          f"val {len(vqa_val_ix)}", flush=True)

    q_t = torch.from_numpy(q_emb).to(device)
    o_t = torch.from_numpy(o_emb).to(device)
    gold_t = torch.from_numpy(gold).to(device)
    n_opts_t = torch.from_numpy(n_opts).to(device)
    opt_start_t = torch.from_numpy(opt_start).to(device)
    patch_t = torch.from_numpy(patch_of_q).to(device)

    def label_vec(row) -> np.ndarray:
        v = np.zeros(len(label_vocab), dtype=np.float32)
        for l in row["labels"]:
            v[label_vocab.index(l)] = 1.0
        return v

    lulc_y = np.zeros((len(pid_list), len(label_vocab)), dtype=np.float32)
    for i, pid in enumerate(pid_list):
        lulc_y[i] = label_vec(pid_to_row[pid])
    lulc_y_t = torch.from_numpy(lulc_y).to(device)

    # ---- EuroSAT replay --------------------------------------------------
    euro_iter = None
    euro_val_loader = None
    if not args.no_eurosat:
        try:
            from torch.utils.data import DataLoader
            from torchgeo.datasets import EuroSAT
            kw = dict(root=str(ROOT / "data" / "eurosat"), download=False,
                      checksum=False)
            euro_train = EuroSAT(split="train", **kw)
            euro_val_ds = EuroSAT(split="val", **kw)
            euro_iter = iter(DataLoader(euro_train, batch_size=args.euro_batch,
                                        shuffle=True, num_workers=0))

            def _euro_cycle():
                nonlocal euro_iter
                euro_iter = iter(DataLoader(euro_train,
                                            batch_size=args.euro_batch,
                                            shuffle=True, num_workers=0))
                return next(euro_iter)

            euro_val_loader = DataLoader(euro_val_ds,
                                         batch_size=args.euro_batch)
            print(f"EuroSAT replay: {len(euro_train)} train / "
                  f"{len(euro_val_ds)} val", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"EuroSAT replay disabled ({type(e).__name__}: {e})",
                  flush=True)
            euro_iter = None

    # ---- optimizer / schedule ---------------------------------------------
    optim = torch.optim.AdamW(
        param_groups(bb, scorer, cap_head, s1_head, euro_head,
                     args.enc_lr, args.head_lr),
        weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    vqa_steps_per_epoch = max(1, len(vqa_tr_ix) // args.vqa_patch_batch)
    aux_per_epoch = sum(1 for s in range(vqa_steps_per_epoch)
                        if s % 4 == 1 or s % 4 == 2
                        or (s % 2 == 1 and euro_iter is not None))
    micro_per_epoch = vqa_steps_per_epoch + aux_per_epoch
    total_micro = micro_per_epoch * args.epochs
    warmup = max(10, int(0.03 * total_micro))

    def lr_mult(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_micro - warmup)
        return 0.05 + 0.475 * (1 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_mult)

    # ---- resume -------------------------------------------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    latest = CKPT_DIR / "latest_full.pth"
    if args.resume and latest.exists():
        st = torch.load(latest, map_location=device, weights_only=False)
        bb.load_state_dict(st["bb"])
        scorer.load_state_dict(st["scorer"])
        cap_head.load_state_dict(st["cap_head"])
        s1_head.load_state_dict(st["s1_head"])
        euro_head.load_state_dict(st["euro_head"])
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch = st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']} "
              f"(composite {st['metrics'].get('composite', '?')})", flush=True)

    # ---- scorer batching ----------------------------------------------------
    def score_with_feats(ix: torch.Tensor, feats: torch.Tensor,
                         pos_lookup: torch.Tensor) -> torch.Tensor:
        """(Q, max_opts) logits for questions ``ix`` against the features of
        the current patch batch, via ``pos_lookup`` mapping global patch
        index -> row in ``feats``."""
        k = int(n_opts_t[ix].max())
        starts = opt_start_t[ix]
        offs = torch.arange(k, device=ix.device)[None]
        oix = (starts[:, None] + offs).clamp(max=o_t.shape[0] - 1)
        valid = offs < n_opts_t[ix][:, None]
        qd = q_t.shape[-1]
        fd = feats.shape[-1]
        f_q = feats[pos_lookup[patch_t[ix]]]        # (Q, fd)
        logits = scorer(
            q_t[ix].unsqueeze(1).expand(-1, k, -1).reshape(-1, qd),
            f_q.unsqueeze(1).expand(-1, k, -1).reshape(-1, fd),
            o_t[oix.reshape(-1)])
        return logits.view(len(ix), k).masked_fill(~valid, NEG)

    def backward_step(loss):
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(
            [p for g in optim.param_groups for p in g["params"]], 1.0)
        scaler.step(optim)
        scaler.update()
        sched.step()

    # ---- validation ----------------------------------------------------------
    @torch.no_grad()
    def eval_epoch() -> dict:
        bb.eval(); scorer.eval(); cap_head.eval()
        s1_head.eval(); euro_head.eval()

        # VQA over val patches
        stats = {"binary": [0, 0], "mcq": [0, 0]}
        for pi0 in range(0, len(vqa_val_ix), args.lulc_batch):
            pis = vqa_val_ix[pi0:pi0 + args.lulc_batch]
            patches = [s2[pid_list[p]] for p in pis]
            with torch.autocast("cuda", enabled=device == "cuda"):
                feats = s2_features(bb, patches, device)
            pos = torch.full((len(pid_list),), -1, dtype=torch.long,
                             device=device)
            pos[torch.from_numpy(pis).to(device)] = torch.arange(
                len(pis), device=device)
            qis = [q for p in pis for q in q_by_patch[int(p)]]
            if not qis:
                continue
            qix = torch.tensor(qis, device=device, dtype=torch.long)
            for qi0 in range(0, len(qix), 256):
                sub = qix[qi0:qi0 + 256]
                with torch.autocast("cuda", enabled=device == "cuda"):
                    logits = score_with_feats(sub, feats, pos)
                pred = logits.argmax(1)
                ok = (pred == gold_t[sub])
                for j, q in enumerate(sub.tolist()):
                    t = "binary" if int(n_opts[q]) == 2 else "mcq"
                    stats[t][0] += int(ok[j].item())
                    stats[t][1] += 1
        tot_c = sum(v[0] for v in stats.values())
        tot_n = sum(v[1] for v in stats.values())
        vqa_acc = tot_c / max(tot_n, 1)

        # S2 LULC exact match on val patches
        s2_ok = s2_n = 0
        for pi0 in range(0, len(val_patch_ix), args.lulc_batch):
            pis = val_patch_ix[pi0:pi0 + args.lulc_batch]
            patches = [s2[pid_list[p]] for p in pis]
            with torch.autocast("cuda", enabled=device == "cuda"):
                f = s2_features(bb, patches, device)
                pred = torch.sigmoid(cap_head(f.float())) > 0.5
            y = lulc_y_t[torch.from_numpy(pis).to(device)].bool()
            s2_ok += int((pred == y).all(1).sum().item())
            s2_n += len(pis)
        s2_acc = s2_ok / max(s2_n, 1)

        # S1 LULC exact match on val patches having an S1 patch
        s1_ok = s1_n = 0
        for r in val_rows:
            pid = r["patch_id"]
            if pid not in s1:
                continue
            with torch.autocast("cuda", enabled=device == "cuda"):
                f = s1_features(bb, [s1[pid]], device)
                pred = torch.sigmoid(s1_head(f.float())) > 0.5
            y = torch.from_numpy(label_vec(r)).to(device).bool().unsqueeze(0)
            s1_ok += int((pred == y).all(1).sum().item())
            s1_n += 1
        s1_acc = s1_ok / max(s1_n, 1)

        # EuroSAT val subset
        euro_acc = None
        if euro_val_loader is not None:
            ok = n = 0
            for i, batch in enumerate(euro_val_loader):
                if i >= 40:
                    break
                x = batch["image"].to(device)
                yl = batch["label"].to(device)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    pred = eurosat_forward(bb, euro_head, x).argmax(1)
                ok += int((pred == yl).sum().item())
                n += yl.numel()
            euro_acc = ok / max(n, 1)

        parts = [vqa_acc, s2_acc, s1_acc] + ([euro_acc] if euro_acc else [])
        return {"vqa_acc": vqa_acc,
                "vqa_binary": stats["binary"], "vqa_mcq": stats["mcq"],
                "s2_exact": s2_acc, "s1_exact": s1_acc,
                "euro_acc": euro_acc,
                "composite": float(np.mean(parts))}

    # ---- epochs ---------------------------------------------------------------
    history = []
    best_composite = -1.0
    best_epoch = None
    rng = np.random.RandomState(42)
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        bb.train(); scorer.train(); cap_head.train()
        s1_head.train(); euro_head.train()
        order = rng.permutation(vqa_tr_ix)
        vqa_loss_sum = vqa_n = 0
        task_n = {"s2": 0, "s1": 0, "euro": 0}
        ep_t0 = time.time()

        for step in range(vqa_steps_per_epoch):
            # ---- VQA micro-batch ----
            pis = order[step * args.vqa_patch_batch:
                        (step + 1) * args.vqa_patch_batch]
            if len(pis) == 0:
                continue
            patches = [s2[pid_list[p]] for p in pis]
            q_list = [q for p in pis for q in q_by_patch[int(p)]]
            if not q_list:
                continue
            if len(q_list) > args.qcap:
                q_list = list(rng.choice(q_list, args.qcap, replace=False))
            with torch.autocast("cuda", enabled=device == "cuda"):
                feats = s2_features(bb, patches, device,
                                    band_dropout=args.band_dropout,
                                    rgb_stretch=args.rgb_stretch, rng=rng)
                pos = torch.full((len(pid_list),), -1, dtype=torch.long,
                                 device=device)
                pos[torch.from_numpy(pis).to(device)] = torch.arange(
                    len(pis), device=device)
                qix = torch.tensor(q_list, device=device, dtype=torch.long)
                logits = score_with_feats(qix, feats, pos)
                # no label smoothing: the smoothed term would put mass on
                # the NEG-padded absent-option slots and drown the loss
                # (measured: ~53 nats of pure padding artifact)
                loss = F.cross_entropy(logits, gold_t[qix])
            backward_step(loss)
            vqa_loss_sum += loss.item() * len(qix)
            vqa_n += len(qix)

            # ---- aux tasks ----
            if step % 4 == 1:      # S2 LULC
                pis2 = rng.choice(tr_patch_ix, size=min(
                    args.lulc_batch, len(tr_patch_ix)), replace=False)
                patches2 = [s2[pid_list[p]] for p in pis2]
                with torch.autocast("cuda", enabled=device == "cuda"):
                    f = s2_features(bb, patches2, device,
                                    band_dropout=args.band_dropout,
                                    rgb_stretch=args.rgb_stretch, rng=rng)
                    logits2 = cap_head(f.float())
                    y2 = lulc_y_t[torch.from_numpy(
                        np.asarray(pis2, dtype=np.int64)).to(device)]
                    loss2 = F.binary_cross_entropy_with_logits(
                        logits2, y2) * 0.5
                backward_step(loss2)
                task_n["s2"] += len(pis2)

            elif step % 4 == 2:    # S1 LULC
                pis1 = rng.choice(tr_patch_ix, size=min(
                    args.lulc_batch, len(tr_patch_ix)), replace=False)
                have = [pid_list[p] for p in pis1 if pid_list[p] in s1]
                if have:
                    patches1 = [s1[p] for p in have]
                    y1 = torch.from_numpy(np.stack(
                        [label_vec(pid_to_row[p]) for p in have])).to(device)
                    with torch.autocast("cuda", enabled=device == "cuda"):
                        f = s1_features(bb, patches1, device)
                        loss1 = F.binary_cross_entropy_with_logits(
                            s1_head(f.float()), y1) * 0.5
                    backward_step(loss1)
                    task_n["s1"] += len(have)

            elif step % 2 == 1 and euro_iter is not None:   # EuroSAT replay
                # every 2nd step at 0.7 weight: pass 1's every-4th at 0.35
                # let the multi-task losses erode EuroSAT to 91.4% test
                # (from 97.8%); the replay is what defends that headline
                # number, so it carries a real share of the schedule.
                batch = next(euro_iter, None)
                if batch is None:
                    batch = _euro_cycle()
                x = batch["image"].to(device)
                yl = batch["label"].to(device)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    loss3 = F.cross_entropy(
                        eurosat_forward(bb, euro_head, x), yl) * 0.7
                backward_step(loss3)
                task_n["euro"] += int(yl.numel())

            if (step + 1) % 100 == 0:
                print(f"  ep {epoch} step {step + 1}/{vqa_steps_per_epoch} "
                      f"vqa loss {vqa_loss_sum / max(vqa_n, 1):.4f} "
                      f"({time.time() - ep_t0:.0f}s)", flush=True)

        # ---- validation ----
        metrics = eval_epoch()
        metrics["epoch"] = epoch
        metrics["vqa_train_loss"] = round(vqa_loss_sum / max(vqa_n, 1), 4)
        metrics["epoch_seconds"] = round(time.time() - ep_t0, 1)
        metrics["aux_batches"] = task_n
        history.append(metrics)
        euro_s = ("None" if metrics["euro_acc"] is None
                  else f"{metrics['euro_acc']:.4f}")
        print(f"EPOCH {epoch}/{args.epochs} vqa {metrics['vqa_acc']:.4f} "
              f"(bin {metrics['vqa_binary'][0]}/{metrics['vqa_binary'][1]}, "
              f"mcq {metrics['vqa_mcq'][0]}/{metrics['vqa_mcq'][1]}) "
              f"| s2 {metrics['s2_exact']:.4f} s1 {metrics['s1_exact']:.4f} "
              f"euro {euro_s} | composite {metrics['composite']:.4f} "
              f"({metrics['epoch_seconds']:.0f}s, "
              f"total {time.time() - t_start:.0f}s)", flush=True)

        # ---- checkpoints ----
        if not args.smoke:
            state = {"bb": bb.state_dict(), "scorer": scorer.state_dict(),
                     "cap_head": cap_head.state_dict(),
                     "s1_head": s1_head.state_dict(),
                     "euro_head": euro_head.state_dict(),
                     "epoch": epoch, "metrics": metrics}
            torch.save(state, CKPT_DIR / f"epoch_{epoch:03d}.pth")
            torch.save({**state, "optim": optim.state_dict(),
                        "sched": sched.state_dict(),
                        "scaler": scaler.state_dict()},
                       CKPT_DIR / "latest_full.pth")
            if metrics["composite"] > best_composite:
                best_composite = metrics["composite"]
                best_epoch = epoch
                torch.save(state, CKPT_DIR / "best.pth")
            METRICS_OUT.write_text(json.dumps(
                {"history": history, "best_epoch": best_epoch,
                 "best_composite": round(best_composite, 4),
                 "config": vars(args)}, indent=2, default=str))

    # ---- promote best epoch to the live artifacts -----------------------------
    if not args.smoke:
        st = torch.load(CKPT_DIR / "best.pth", map_location="cpu",
                        weights_only=False)
        if BB_CKPT.exists():
            shutil.copy2(BB_CKPT, ROOT / "models" / "backbone_ft_v1.bak.pth")
        if HEADS_OUT.exists():
            shutil.copy2(HEADS_OUT,
                         ROOT / "models" / "ben_txt_heads_v1.bak.pth")
        sd_out = st["bb"]
        sd_out["ft_cls_head.weight"] = st["euro_head"]["weight"]
        sd_out["ft_cls_head.bias"] = st["euro_head"]["bias"]
        sd_out["s1_cls_head.weight"] = st["s1_head"]["weight"]
        sd_out["s1_cls_head.bias"] = st["s1_head"]["bias"]
        torch.save(sd_out, BB_CKPT)
        torch.save({"vqa_scorer": st["scorer"],
                    "caption_head": st["cap_head"],
                    "s1_head": st["s1_head"],
                    "embed_dim": TXT_DIM, "arch": "v2",
                    "text_encoder": TEXT_MODEL,
                    "val_metrics": st["metrics"]}, HEADS_OUT)
        print(f"promoted epoch {st['epoch']} "
              f"(composite {st['metrics']['composite']:.4f}) -> "
              f"{BB_CKPT.name} + {HEADS_OUT.name}", flush=True)
        print("v1 artifacts backed up: models/backbone_ft_v1.bak.pth, "
              "models/ben_txt_heads_v1.bak.pth", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

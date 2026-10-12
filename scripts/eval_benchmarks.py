"""Evaluate the fine-tuned backbone + VQA scorer on the public benchmarks
named in the problem statement: RSVQA-LR, VRSBench, CDVQA.

Two protocols per benchmark:
  zero-shot   -- the scorer trained on BigEarthNet.txt answers questions from
                another dataset without seeing it (honest generalization).
  supervised  -- the standard benchmark protocol of the RSVQA/VRSBench/CDVQA
                papers: adapt the scorer on the benchmark's TRAIN split,
                evaluate on its test/dev split.

v2 (2026-09-11 joint run): both protocols use the jointly trained encoder
(models/backbone_ft.pth -- features are re-encoded after every encoder
change, see the note on ``feature_cache``) and the scorer v2 architecture
(MiniLM text, FiLM fusion) from scripts/train_joint.py. The text embedder
is universal -- no per-benchmark TF-IDF fitting -- so supervised adaptation
trains only the scorer on top of MiniLM embeddings.

Usage:
    python scripts/eval_benchmarks.py --bench rsvqa_lr --mode both
    python scripts/eval_benchmarks.py --bench vrsbench --mode both
    python scripts/eval_benchmarks.py --bench cdvqa --mode both
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tarfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from train_joint import VQAScorerV2

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "bench"
HEADS_OUT = ROOT / "models" / "ben_txt_heads.pth"

RSVQA_ALL_IMG = BENCH / "rsvqa_lr_images_all"
RSVQA_SPLITS = {s: BENCH / f"rsvqa_lr_LR_split_{s}_images.json"
                for s in ("train", "test")}
RSVQA_Q = {s: BENCH / f"rsvqa_lr_LR_split_{s}_questions.json"
           for s in ("train", "test")}
RSVQA_A = {s: BENCH / f"rsvqa_lr_LR_split_{s}_answers.json"
           for s in ("train", "test")}
VRS_ANN_VAL = BENCH / "vrsbench_ann" / "Annotations_val"
VRS_ANN_TRAIN = BENCH / "vrsbench_ann" / "Annotations_train"
VRS_EVAL = BENCH / "vrsbench_eval_vqa.json"
VRS_IMG_VAL = BENCH / "vrsbench_val_images"
VRS_IMG_TRAIN = BENCH / "vrsbench_train_images"
CDVQA_SHARDS = BENCH / "cdvqa_shards"
CDVQA_TRAIN_SHARDS = BENCH / "cdvqa_train_shards"

_dev = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE on feature caches: data/bench/feat_*.npz hold encoder features and
# MUST be deleted whenever models/backbone_ft.pth changes, or every eval
# silently scores with the previous encoder. The wrapper below stamps the
# cache filename with the checkpoint's sha so a new encoder cannot reuse
# a stale cache.


# ------------------------------------------------------------------ model
def _norm(s):
    return str(s).strip().lower()


class EmbedWrap:
    """MiniLM-L6-v2 over any question/option text (no fitting per set)."""

    NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DIM = 384

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        try:
            self.st = SentenceTransformer(self.NAME, device=_dev)
        except Exception:                                       # noqa: BLE001
            self.st = SentenceTransformer(self.NAME, device="cpu")

    def encode(self, texts, bs: int = 256) -> np.ndarray:
        return np.asarray(self.st.encode(
            texts, batch_size=bs, show_progress_bar=False),
            dtype=np.float32)


_EMBED: EmbedWrap | None = None


def embedder() -> EmbedWrap:
    global _EMBED
    if _EMBED is None:
        _EMBED = EmbedWrap()
    return _EMBED


def load_zero_shot():
    """Scorer + text embedder trained on BigEarthNet.txt (joint v2)."""
    blob = torch.load(HEADS_OUT, map_location=_dev, weights_only=True)
    if blob.get("arch") != "v2":
        raise RuntimeError(
            f"{HEADS_OUT} is arch={blob.get('arch', 'v1')!r}; this "
            "evaluator expects the joint v2 artifacts. Run "
            "scripts/train_joint.py first.")
    model = VQAScorerV2(txt_dim=int(blob["embed_dim"])).to(_dev)
    model.load_state_dict(blob["vqa_scorer"])
    model.eval()
    return model, embedder()


def _new_scorer(feat_dim=2048):
    return VQAScorerV2(feat_dim=feat_dim).to(_dev)


_bb = None


def backbone():
    global _bb
    if _bb is None:
        from satquery.stage2.backbone import AdaptedBackbone
        ck = torch.load(ROOT / "models" / "backbone_ft.pth",
                        map_location="cpu", weights_only=False)
        bb = AdaptedBackbone()
        bb.load_state_dict({k: v for k, v in ck.items()
                            if not k.endswith(
                                ("ft_cls_head.weight", "ft_cls_head.bias",
                                 "s1_cls_head.weight", "s1_cls_head.bias"))})
        bb.eval().to(_dev)
        _bb = bb
    return _bb


def _cache_tag() -> str:
    """First 8 hex of the checkpoint sha -- cache key per encoder version."""
    import hashlib
    h = hashlib.sha256()
    with open(ROOT / "models" / "backbone_ft.pth", "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:8]


@torch.no_grad()
def encode_rgb(buf):
    """buf: list of (3, H, W) float arrays -> (N, 2048) features.

    RGB rides the band projector (red/green/blue slots; the other nine are
    interpolated), normalised to the encoder's Sentinel-2 statistics.
    The band-dropout augmentation in the joint training run made this
    3-band input mode part of the encoder's training distribution.
    """
    x = torch.from_numpy(np.stack(buf)).to(_dev)
    bb = backbone()
    x12 = bb.band_proj({"red": x[:, 0], "green": x[:, 1], "blue": x[:, 2]})
    with torch.autocast("cuda", enabled=_dev == "cuda"):
        f = bb.opt_enc(x12).flatten(1)
        f = bb.gsd_cond(f, gsd=10.0)
    return f.float().cpu().numpy().astype(np.float32)


def _load_rgb(path):
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-3:
        a = a * 0 + 0.5
    else:
        a = (a - lo) / (hi - lo)
    return np.clip(a, 0, 1).transpose(2, 0, 1)


def feature_cache(paths, cache_file, batch=64):
    """(N, 2048) features for `paths`, computed once and cached per id.

    The cache file is suffixed with the backbone checkpoint sha so a new
    encoder writes a fresh cache instead of silently reusing the old one.
    """
    cache_file = Path(str(cache_file).replace(
        ".npz", f".{_cache_tag()}.npz"))
    ids = [Path(p).name for p in paths]
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        have = dict(zip([str(i) for i in z["id"]], z["feat"]))
        missing = [p for p, i in zip(paths, ids) if i not in have]
        if not missing:
            return np.stack([have[i] for i in ids])
    else:
        have, missing = {}, list(paths)
    out = {}
    for i in range(0, len(missing), batch):
        chunk = missing[i:i + batch]
        feats = encode_rgb([_load_rgb(p) for p in chunk])
        for p, f in zip(chunk, feats):
            out[Path(p).name] = f
        print(f"  encoded {min(i + batch, len(missing))}/{len(missing)}",
              flush=True)
    merged = {**have, **out}
    np.savez(cache_file, id=np.array(list(merged.keys())),
             feat=np.stack(list(merged.values())).astype(np.float32))
    return np.stack([merged[i] for i in ids])


# ------------------------------------------------------- scorer adaptation
def _split_qo(examples, emb: EmbedWrap):
    """Per-example (q_emb, [o_emb...]) with a flat option matrix + offsets."""
    stems = [ex[0] for ex in examples]
    opts_flat = [o for ex in examples for o in ex[1]]
    q = emb.encode(stems)
    o = emb.encode(opts_flat)
    q_emb = np.asarray(q, dtype=np.float32)
    o_emb = np.asarray(o, dtype=np.float32)
    starts, s = [], 0
    for ex in examples:
        starts.append(s)
        s += len(ex[1])
    return q_emb, o_emb, np.asarray(starts, dtype=np.int64)


def adapt_scorer(examples, feats_by_img, epochs=10, batch=512, lr=1e-3,
                 feat_dim=2048, groups=None, emb: EmbedWrap | None = None):
    """Train a fresh VQAScorerV2 on (question, image, options) examples.

    groups: per-example group ids (e.g. CDVQA pair ids). When given, the
    val split holds out entire GROUPS -- random row splits leak identical
    pair features across folds on question-redundant datasets.
    """
    emb = emb or embedder()
    q_emb, o_emb, starts = _split_qo(examples, emb)
    Xf = np.stack([feats_by_img[ex[2]] for ex in examples]).astype(np.float32)
    gold = np.array([ex[3] for ex in examples], dtype=np.int64)
    n_opts = np.array([len(ex[1]) for ex in examples], dtype=np.int64)

    q_t = torch.from_numpy(q_emb).to(_dev)
    o_t = torch.from_numpy(o_emb).to(_dev)
    f_t = torch.from_numpy(Xf).to(_dev)
    opt_start_t = torch.from_numpy(starts).to(_dev)
    n_opts_t = torch.from_numpy(n_opts).to(_dev)
    gold_t = torch.from_numpy(gold).to(_dev)

    model = _new_scorer(feat_dim=feat_dim).train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs * max(1, len(examples) // batch))

    def score_batch(ix):
        k = int(n_opts_t[ix].max())
        offs = torch.arange(k, device=ix.device)[None]
        oix = (opt_start_t[ix][:, None] + offs).clamp(max=o_t.shape[0] - 1)
        valid = offs < n_opts_t[ix][:, None]
        qd = q_t.shape[-1]
        fd = f_t.shape[-1]
        logits = model(q_t[ix].unsqueeze(1).expand(-1, k, -1).reshape(-1, qd),
                       f_t[ix].unsqueeze(1).expand(-1, k, -1).reshape(-1, fd),
                       o_t[oix.reshape(-1)]).view(len(ix), k)
        return logits.masked_fill(~valid, -1e4)

    rng = np.random.default_rng(42)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_val_g = max(1, int(len(uniq) * 0.12))
        val_g = set(uniq[:n_val_g])
        val_mask = np.array([g in val_g for g in groups])
    else:
        order_all = rng.permutation(len(examples))
        n_val = max(512, len(examples) // 12)
        val_mask = np.zeros(len(examples), dtype=bool)
        val_mask[order_all[:n_val]] = True
    val_ix = torch.from_numpy(np.where(val_mask)[0]).long().to(_dev)
    tr_ix = torch.from_numpy(np.where(~val_mask)[0]).long().to(_dev)

    import time
    t0 = time.time()
    best_acc, best_state, bad = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        order = tr_ix[torch.randperm(len(tr_ix), device=_dev)]
        tot, seen = 0.0, 0
        for i in range(0, len(order), batch):
            ix = order[i:i + batch]
            logits = score_batch(ix)
            loss = torch.nn.functional.cross_entropy(logits, gold_t[ix])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tot += loss.item() * len(ix); seen += len(ix)
        sched.step()
        model.eval()
        with torch.no_grad():
            vacc = (score_batch(val_ix).argmax(1)
                    == gold_t[val_ix]).float().mean().item()
        print(f"  ep {ep}/{epochs} loss {tot / max(seen, 1):.4f} "
              f"val {vacc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vacc > best_acc:
            best_acc, bad = vacc, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                print(f"  early stop at ep {ep} (best val {best_acc:.4f})",
                      flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, emb


@torch.no_grad()
def score_one(model, emb, feat, question, options):
    qe = emb.encode([question])
    oe = emb.encode(options)
    q = torch.from_numpy(qe).repeat(len(options), 1).to(_dev)
    f = torch.from_numpy(feat).repeat(len(options), 1).to(_dev)
    o = torch.from_numpy(oe).to(_dev)
    return int(model(q, f, o).argmax())


def run_eval(model, emb, examples, feats_by_img):
    """examples: (img_id, question, options, gold, qtype) -> metrics parts.

    Text is embedded in large batches (MiniLM over one question at a time
    would multiply the run by ~30x for nothing).
    """
    stats = defaultdict(lambda: [0, 0])
    correct = total = skipped = 0
    B = 512
    for i0 in range(0, len(examples), B):
        chunk = examples[i0:i0 + B]
        stems = [e[1] for e in chunk]
        opts_flat = [o for e in chunk for o in e[2]]
        q = torch.from_numpy(emb.encode(stems)).to(_dev)
        o = torch.from_numpy(emb.encode(opts_flat)).to(_dev)
        kmax = max(len(e[2]) for e in chunk)
        rows, valid, oix, ptr = [], [], [], 0
        for e in chunk:
            rows.append(feats_by_img.get(e[0]))
            valid.append([True] * len(e[2]) + [False] * (kmax - len(e[2])))
            oix.extend(range(ptr, ptr + len(e[2])))
            oix.extend([ptr] * (kmax - len(e[2])))
            ptr += len(e[2])
        missing = [j for j, r in enumerate(rows) if r is None]
        if missing:
            skipped += len(missing)
        keep = [j for j in range(len(chunk)) if rows[j] is not None]
        if not keep:
            continue
        f = torch.from_numpy(np.stack([rows[j] for j in keep])).to(_dev)
        qk = q[keep]
        valid_t = torch.tensor([valid[j] for j in keep], device=_dev)
        oix_t = oix
        # recompute option gather for kept rows only
        oix_keep, ptr = [], 0
        for j in keep:
            n = len(chunk[j][2])
            oix_keep.extend(range(ptr, ptr + n))
            oix_keep.extend([ptr] * (kmax - n))
            ptr += n
        oe = o[torch.tensor(oix_keep, device=_dev)]
        qe = qk.unsqueeze(1).expand(-1, kmax, -1).reshape(-1, qk.shape[-1])
        fe = f.unsqueeze(1).expand(-1, kmax, -1).reshape(-1, f.shape[-1])
        logits = model(qe, fe, oe).view(len(keep), kmax)
        logits = logits.masked_fill(~valid_t, -1e9)
        preds = logits.argmax(1).tolist()
        for j, pred in zip(keep, preds):
            e = chunk[j]
            ok = pred == e[3]
            correct += ok
            total += 1
            stats[e[4]][0] += ok
            stats[e[4]][1] += 1
        if total and total % 4096 < len(keep):
            print(f"  scored {total} ({correct / total:.3f})", flush=True)
    return stats, correct, total, skipped


def _summary(name, stats, correct, total, skipped, extra):
    lines = ["=" * 60, f"{name}", "=" * 60]
    acc = correct / max(total, 1)
    lines.append(f"overall: {acc:.4f}  ({correct}/{total}, {skipped} skipped)")
    for k, (c, n) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {k:22s} {c / max(n, 1):.4f}  (n={n})")
    lines.append("=" * 60)
    print("\n".join(lines), flush=True)
    return {"accuracy": round(float(acc), 4), "n_scored": total,
            "n_skipped": skipped,
            "accuracy_by_type": {k: round(c / max(n, 1), 4)
                                 for k, (c, n) in stats.items()},
            "images": extra}


# ---------------------------------------------------------------- RSVQA-LR
def rsvqa_examples(split):
    """RSVQA questions as MCQ: presence/comp -> yes/no, count -> 0..10,
    rural_urban -> rural/urban. Counts >10 are skipped."""
    imgs = json.load(open(RSVQA_SPLITS[split]))["images"]
    qs = json.load(open(RSVQA_Q[split]))["questions"]
    ans = {a["id"]: a for a in json.load(open(RSVQA_A[split]))["answers"]}
    ex = []
    for q in qs:
        if not q["active"]:
            continue
        if not (RSVQA_ALL_IMG / f"{q['img_id']}.tif").exists():
            continue
        a = _norm(ans[q["answers_ids"][0]]["answer"])
        qt = _norm(q["question"].rstrip("?"))
        t = q["type"]
        if t in ("presence", "comp"):
            if a not in ("yes", "no"):
                continue
            ex.append((q["img_id"], qt, ["yes", "no"],
                       0 if a == "yes" else 1, t))
        elif t == "rural_urban":
            if a not in ("rural", "urban"):
                continue
            ex.append((q["img_id"], qt, ["rural", "urban"],
                       0 if a == "rural" else 1, t))
        elif t == "count":
            if not a.isdigit() or int(a) > 10:
                continue
            ex.append((q["img_id"], qt, [str(v) for v in range(11)],
                       int(a), t))
    return ex


def bench_rsvqa_lr(mode):
    results = {}
    train_ex = rsvqa_examples("train")
    test_ex = rsvqa_examples("test")
    print(f"rsvqa_lr: {len(train_ex)} train, {len(test_ex)} test examples")
    all_ids = sorted({e[0] for e in train_ex} | {e[0] for e in test_ex})
    feats = feature_cache([RSVQA_ALL_IMG / f"{i}.tif" for i in all_ids],
                          BENCH / "feat_rsvqa_lr.npz")
    feats_by_img = dict(zip(all_ids, feats))

    if mode in ("zero-shot", "both"):
        model, emb = load_zero_shot()
        st, c, n, sk = run_eval(model, emb, test_ex, feats_by_img)
        results["rsvqa_lr_zero_shot"] = {**_summary(
            "RSVQA-LR test -- ZERO-SHOT (BEN.txt joint scorer)",
            st, c, n, sk, f"{len(all_ids)} images"), "benchmark": "RSVQA-LR"}

    if mode in ("supervised", "both"):
        # adapt: (question, options, img, gold) layout for adapt_scorer
        tr = [(e[1], e[2], e[0], e[3]) for e in train_ex]
        model, emb = adapt_scorer(tr, feats_by_img)
        st, c, n, sk = run_eval(model, emb, test_ex, feats_by_img)
        results["rsvqa_lr_supervised"] = {**_summary(
            "RSVQA-LR test -- SUPERVISED (train-split adapted)",
            st, c, n, sk, f"{len(all_ids)} images"), "benchmark": "RSVQA-LR"}
        _save(model, BENCH / "rsvqa_lr_adapted.pkl")
    return results


# --------------------------------------------------------------- VRSBench
def _vrs_option_pool(ann_dir, rng, per_type=40):
    """type -> candidate answer list (official-style distractor pools)."""
    pools = defaultdict(lambda: defaultdict(int))
    for fp in ann_dir.glob("*.json"):
        for a in json.load(open(fp))["qa_pairs"]:
            pools[a["type"]][_norm(a["answer"])] += 1
    out = {}
    for t, cnt in pools.items():
        ranked = [a for a, _ in sorted(cnt.items(), key=lambda kv: -kv[1])]
        # keep the most common answers as the distractor vocabulary
        out[t] = ranked[:per_type]
    return out


def _vrs_options(gold, pool, rng, k=4):
    """gold + (k-1) distinct same-type distractors, gold first."""
    cands = [a for a in pool if a != gold]
    rng.shuffle(cands)
    return [gold] + cands[:k - 1]


def vrs_examples(split, seed=42):
    """(img_id, question, options, gold, type); gold is always option 0.
    Options follow the official MCQ protocol: ground truth plus same-answer-
    type distractors drawn from the split's answer pool.
    """
    rng = np.random.default_rng(seed)
    ex = []
    if split == "test":
        pools = _vrs_option_pool(VRS_ANN_VAL, rng)
        by_img = defaultdict(list)
        for x in json.load(open(VRS_EVAL)):
            by_img[x["image_id"]].append(x)
        for img_id, rows in sorted(by_img.items()):
            if not (VRS_ANN_VAL / (img_id[:-4] + ".json")).exists():
                continue
            for x in rows:
                gold = _norm(x["ground_truth"])
                opts = _vrs_options(gold, pools[x["type"]], rng)
                ex.append((img_id, _norm(x["question"].rstrip("?")),
                           opts, 0, x["type"]))
        return ex
    pools = _vrs_option_pool(VRS_ANN_TRAIN, rng)
    for fp in sorted(VRS_ANN_TRAIN.glob("*.json")):
        img_id = fp.stem + ".png"
        if not (VRS_IMG_TRAIN / img_id).exists():
            continue
        for a in json.load(open(fp))["qa_pairs"]:
            gold = _norm(a["answer"])
            opts = _vrs_options(gold, pools[a["type"]], rng)
            ex.append((img_id, _norm(a["question"].rstrip("?")), opts,
                       0, a["type"]))
    return ex


def bench_vrsbench(mode, limit=0):
    results = {}
    test_ex = vrs_examples("test")
    if limit:
        test_ex = test_ex[:limit]
    print(f"vrsbench: {len(test_ex)} eval examples")
    val_ids = sorted({e[0] for e in test_ex})
    feats = feature_cache([VRS_IMG_VAL / i for i in val_ids],
                          BENCH / "feat_vrsbench_val.npz")
    feats_by_img = dict(zip(val_ids, feats))

    if mode in ("zero-shot", "both"):
        model, emb = load_zero_shot()
        st, c, n, sk = run_eval(model, emb, test_ex, feats_by_img)
        results["vrsbench_zero_shot"] = {**_summary(
            "VRSBench dev -- ZERO-SHOT (BEN.txt joint scorer)",
            st, c, n, sk, f"{len(val_ids)} images"), "benchmark": "VRSBench"}

    if mode in ("supervised", "both"):
        tr = vrs_examples("train")
        print(f"vrsbench train: {len(tr)} examples")
        tr_ids = sorted({e[0] for e in tr})
        trf = feature_cache([VRS_IMG_TRAIN / i for i in tr_ids],
                            BENCH / "feat_vrsbench_train.npz")
        tr_feats = dict(zip(tr_ids, trf))
        tr_adapt = [(e[1], e[2], e[0], e[3]) for e in tr]
        model, emb = adapt_scorer(tr_adapt, tr_feats)
        st, c, n, sk = run_eval(model, emb, test_ex, feats_by_img)
        results["vrsbench_supervised"] = {**_summary(
            "VRSBench dev -- SUPERVISED (train-split adapted)",
            st, c, n, sk, f"{len(val_ids)} images"), "benchmark": "VRSBench"}
        _save(model, BENCH / "vrsbench_adapted.pkl")
    return results


# ----------------------------------------------------------------- CDVQA
def _cdvqa_iter(shard_dir, limit_shards=0):
    shards = sorted(shard_dir.glob("*.tar"))
    if limit_shards:
        shards = shards[:limit_shards]
    for sp in shards:
        try:
            t = tarfile.open(sp)
            keys = sorted({n.rsplit(".", 2)[0] for n in t.getnames()
                           if n.endswith(".json")})
        except (tarfile.ReadError, tarfile.TarError, EOFError,
                OSError) as e:
            # OSError covers files still being written (curl holds them)
            print(f"  skipping partial/corrupt shard {sp.name} ({e})",
                  flush=True)
            continue
        for k in keys:
            try:
                j = json.load(t.extractfile(k + ".json"))
            except (KeyError, tarfile.TarError, json.JSONDecodeError,
                    AttributeError, OSError, EOFError):
                continue
            yield t, k, j


def cdvqa_examples(shard_dir, limit_shards=0, max_pairs=0):
    ex, imgs = [], []
    pair_groups = []
    for t, k, j in _cdvqa_iter(shard_dir, limit_shards):
        conv = j["conversations"]
        question = conv[0]["value"].replace(
            "Image 1: <image>\nImage 2: <image>\n", "").rstrip("?")
        a = _norm(conv[1]["value"])
        if a not in ("yes", "no"):
            continue
        ex.append((k, _norm(question), ["yes", "no"],
                   0 if a == "yes" else 1,
                   j["meta"].get("question_type", "unknown")))
        pair_groups.append(j["meta"].get("image_id", k))
        if max_pairs and len(ex) >= max_pairs:
            break
    return ex, pair_groups


def bench_cdvqa(mode, limit_shards=0, limit_train=0):
    results = {}
    test_ex, _ = cdvqa_examples(CDVQA_SHARDS, limit_shards)
    print(f"cdvqa: {len(test_ex)} test examples "
          f"(from {len(sorted(CDVQA_SHARDS.glob('*.tar')))} shards)")
    if not test_ex:
        print("no CDVQA shards found — download first")
        return results

    if mode in ("zero-shot", "both"):
        # mean-pooled pair features keep the scorer's 2048-d input
        test_feats = cdvqa_pair_features(CDVQA_SHARDS, limit_shards,
                                         BENCH / "feat_cdvqa_test_mean.npz",
                                         mode="mean")
        feats_by = {k: v for k, v in test_feats.items()
                    if any(k == e[0] for e in test_ex)}
        model, emb = load_zero_shot()
        st, c, n, sk = run_eval(model, emb, test_ex, feats_by)
        results["cdvqa_zero_shot"] = {**_summary(
            "CDVQA test -- ZERO-SHOT (BEN.txt joint scorer, mean-pooled pair)",
            st, c, n, sk, f"{len(feats_by)} pairs"), "benchmark": "CDVQA"}

    if mode in ("supervised", "both"):
        tr_shards = CDVQA_TRAIN_SHARDS
        if not any(tr_shards.glob("*.tar")):
            print("no CDVQA train shards — supervised skipped")
        else:
            tr_ex, tr_groups = cdvqa_examples(tr_shards, 0, limit_train)
            tr_feats = cdvqa_pair_features(tr_shards, 0,
                                           BENCH / "feat_cdvqa_train.npz",
                                           limit_train)
            tr_feats = _change_vec(tr_feats)  # [f0-f1, f0+f1]
            tr_adapt = [(e[1], e[2], e[0], e[3]) for e in tr_ex
                        if e[0] in tr_feats]
            tr_group_ids = [g for e, g in zip(tr_ex, tr_groups)
                            if e[0] in tr_feats]
            print(f"cdvqa train: {len(tr_adapt)} examples / "
                  f"{len(set(tr_group_ids))} pairs (change-vector features)")
            model, emb = adapt_scorer(tr_adapt, tr_feats,
                                       epochs=24, feat_dim=4096,
                                       groups=tr_group_ids)
            test_feats = _change_vec(cdvqa_pair_features(
                CDVQA_SHARDS, limit_shards,
                BENCH / "feat_cdvqa_test_concat.npz", mode="concat"))
            feats_by = {k: v for k, v in test_feats.items()
                        if any(k == e[0] for e in test_ex)}
            st, c, n, sk = run_eval(model, emb, test_ex, feats_by)
            results["cdvqa_supervised"] = {**_summary(
                "CDVQA test -- SUPERVISED (train-shard adapted, change vector)",
                st, c, n, sk, f"{len(feats_by)} pairs"),
                "benchmark": "CDVQA"}
            _save(model, BENCH / "cdvqa_adapted.pkl")
    return results


def _change_vec(feats):
    """4096-d concat [f0, f1] -> change-vector [f0-f1, f0+f1].

    A change question is about the difference; handing the MLP the raw
    concat forces it to learn aligned +/- weights across 2048 dim pairs
    (which 1.4k examples could not do — loss sat at ln2). The explicit
    difference is linearly accessible to the first layer.
    """
    out = {}
    for k, v in feats.items():
        h = v.shape[0] // 2
        out[k] = np.concatenate([v[:h] - v[h:], v[:h] + v[h:]]).astype(
            np.float32)
    return out


@torch.no_grad()
def cdvqa_pair_features(shard_dir, limit_shards, cache_file, max_pairs=0,
                        mode="concat"):
    """key -> pair feature. concat=4096-d (supervised pair scorer),
    mean=2048-d (zero-shot scorer expects single-scene dims).

    The cache file is stamped with the encoder checkpoint sha (see
    ``feature_cache``) so it cannot be reused across encoder versions.
    """
    cache_file = Path(str(cache_file).replace(
        ".npz", f".{_cache_tag()}.npz"))
    have = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        have = dict(zip([str(i) for i in z["id"]], z["feat"]))
    out, n = {}, 0
    for t, k, j in _cdvqa_iter(shard_dir, limit_shards):
        if k in have:
            continue
        if max_pairs and n >= max_pairs:
            break
        try:
            f0 = encode_rgb([_load_rgb_img(t.extractfile(k + ".0.img").read())])[0]
            f1 = encode_rgb([_load_rgb_img(t.extractfile(k + ".1.img").read())])[0]
        except (KeyError, tarfile.TarError):
            continue
        out[k] = (np.concatenate([f0, f1], 0) if mode == "concat"
                  else (f0 + f1) / 2.0)
        n += 1
        if n % 200 == 0:
            print(f"  pair-encoded {n}", flush=True)
    merged = {**have, **out}
    if out:
        np.savez(cache_file, id=np.array(list(merged.keys())),
                 feat=np.stack(list(merged.values())).astype(np.float32))
    return merged


def _load_rgb_img(raw):
    im = Image.open(BytesIO(raw)).convert("RGB").resize(
        (224, 224), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-3:
        a = a * 0 + 0.5
    else:
        a = (a - lo) / (hi - lo)
    return np.clip(a, 0, 1).transpose(2, 0, 1)


def _save(model, path):
    with open(path, "wb") as f:
        pickle.dump({"model": model.state_dict(), "arch": "v2",
                     "text": EmbedWrap.NAME}, f)
    print(f"  saved adapted scorer -> {path}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="all",
                    choices=["rsvqa_lr", "vrsbench", "cdvqa", "all"])
    ap.add_argument("--mode", default="both",
                    choices=["zero-shot", "supervised", "both"])
    ap.add_argument("--limit", type=int, default=0,
                    help="cap eval examples (smoke run)")
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="cap CDVQA train examples")
    ap.add_argument("--out", default=str(ROOT / "data" / "bench_metrics.json"))
    args = ap.parse_args()

    todo = (["rsvqa_lr", "vrsbench", "cdvqa"]
            if args.bench == "all" else [args.bench])
    results = {}
    for b in todo:
        print(f"\n>>> {b} ({args.mode})", flush=True)
        if b == "rsvqa_lr":
            results.update(bench_rsvqa_lr(args.mode))
        elif b == "vrsbench":
            results.update(bench_vrsbench(args.mode, args.limit))
        else:
            results.update(bench_cdvqa(args.mode, args.limit_shards,
                                       args.limit_train))

    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

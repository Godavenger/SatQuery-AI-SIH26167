"""Train BigEarthNet.txt text heads on top of the adapted backbone.

Phase 0 (--finetune-encoder): continue the encoder's fine-tuning on the
BigEarthNet-19 multi-label LULC task over the extracted patches, updating
models/backbone_ft.pth. This makes the adaptation claim EuroSAT ->
BigEarthNet sequential.

Phase 1: feature cache - the updated encoder runs once over every selected
patch; 2048-d features + label vectors are cached to data/ben_txt/features.npz.

Phase 2: heads on cached features:
    - VQA scorer  : unified multiple-choice answerer. Binary questions get
      options ["yes","no"]; MCQ options are parsed from the input text.
      A 2-layer MLP scores (question, image, option) triples; trained with
      cross-entropy over each question's options.
    - caption head: multi-label sigmoid classifier over the 19-class LULC
      vocabulary, used to seed the caption template.

Artifacts:
    models/backbone_ft.pth          - encoder updated (phase 0)
    models/ben_txt_heads.pth        - VQA scorer + caption head state
    models/ben_txt_text_embed.pkl   - TF-IDF vectorizer + SVD
    data/ben_txt/features.npz       - cached features
    data/ben_txt/train_metrics.json

Usage:
    python scripts/train_ben_txt.py --finetune-encoder --vqa-epochs 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
BEN = ROOT / "data" / "ben_txt"
PATCH_DIR = BEN / "patches"
FEAT_OUT = BEN / "features.npz"
MANIFEST = BEN / "manifest.json"
ANN = BEN / "annotations.parquet"
CKPT = ROOT / "models" / "backbone_ft.pth"
HEADS_OUT = ROOT / "models" / "ben_txt_heads.pth"
EMBED_OUT = ROOT / "models" / "ben_txt_text_embed.pkl"

BAND_TO_SLOT = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08", "B09": "nir09",
    "B11": "swir1", "B12": "swir2",
}
SLOT_ORDER = ["coastal", "blue", "green", "red", "rededge1", "rededge2",
              "rededge3", "nir", "nir08", "nir09", "swir1", "swir2"]


# --------------------------------------------------------------------------
# phase 0: encoder fine-tune on BigEarthNet-19 multi-label LULC
# --------------------------------------------------------------------------

def load_patch_tensor(pid: str) -> dict[str, torch.Tensor]:
    z = np.load(PATCH_DIR / f"{pid}.npz")
    bands = {}
    for k in z.files:
        slot = BAND_TO_SLOT[k]
        bands[slot] = torch.from_numpy(z[k].astype(np.float32) / 10000.0)[None]
    return bands


def available_patches(manifest: dict) -> list[dict]:
    """Manifest rows whose S2 npz actually exists on disk.

    The v2 tar omits ~3% of metadata-listed patches (quality-filtered,
    mostly snowy Finland scenes), so training and evaluation must filter
    to what was extracted.
    """
    rows = [r for r in manifest["patches"]
            if (PATCH_DIR / f"{r['patch_id']}.npz").exists()]
    dropped = len(manifest["patches"]) - len(rows)
    if dropped:
        print(f"note: {dropped} manifest patches missing on disk; "
              f"using {len(rows)}")
    return rows


def finetune_encoder(epochs: int, batch: int) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone().to(device)
    if CKPT.exists():
        sd = torch.load(CKPT, map_location=device, weights_only=True)
        bb.load_state_dict(sd, strict=False)
        print("resumed from EuroSAT fine-tuned checkpoint")

    manifest = json.loads(MANIFEST.read_text())
    label_vocab = manifest["labels"]
    rows = [r for r in available_patches(manifest)
            if r["split"] == "train"]
    rng = np.random.RandomState(42)
    rng.shuffle(rows)
    n_val = max(300, int(0.1 * len(rows)))
    val_rows, tr_rows = rows[:n_val], rows[n_val:]

    head = nn.Linear(bb.FEAT_DIM, len(label_vocab)).to(device)
    if "ft_cls_head.weight" in (sd if CKPT.exists() else {}):
        try:
            head.weight.data.copy_(sd["ft_cls_head.weight"].to(device))
            head.bias.data.copy_(sd["ft_cls_head.bias"].to(device))
            print("resumed BEN head from previous run")
        except Exception:
            pass

    for p in bb.sar_enc.parameters():
        p.requires_grad = False
    enc_params = [p for p in bb.parameters() if p.requires_grad]
    optim = torch.optim.AdamW([
        {"params": enc_params, "lr": 1e-5},
        {"params": head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs * max(1, len(tr_rows) // batch))

    def label_vec(r):
        v = np.zeros(len(label_vocab), dtype=np.float32)
        for l in r["labels"]:
            v[label_vocab.index(l)] = 1.0
        return v

    def run_epoch(split, train_mode):
        bb.train(train_mode); head.train(train_mode)
        data = tr_rows if train_mode else val_rows
        order = rng.permutation(len(data)) if train_mode else np.arange(len(data))
        tot, seen, correct, nlab = 0.0, 0, 0, 0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            feats, ys = [], []
            for j in idx:
                bands = load_patch_tensor(data[j]["patch_id"])
                feats.append(bands)
                ys.append(label_vec(data[j]))
            # batch the band tensors
            batched = {s: torch.cat([f[s] for f in feats], 0).to(device)
                       for s in SLOT_ORDER}
            y = torch.from_numpy(np.stack(ys)).to(device)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                x = bb.band_proj(batched)
                f = bb.opt_enc(x).flatten(1)
                f = bb.gsd_cond(f, gsd=10.0)
                logits = head(f)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), y)
            if train_mode:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_params + list(head.parameters()), 1.0)
                optim.step()
                sched.step()
            tot += loss.item() * len(idx); seen += len(idx)
            pred = (torch.sigmoid(logits) > 0.5)
            correct += (pred.bool() == y.bool()).all(dim=1).sum().item()
            nlab += len(idx)
        return tot / max(seen, 1), correct / max(nlab, 1)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_exact = run_epoch(tr_rows, True)
        va_loss, va_exact = run_epoch(val_rows, False)
        print(f"enc ep {ep}/{epochs} train loss {tr_loss:.4f} "
              f"exact-match {tr_exact:.3f} | val loss {va_loss:.4f} "
              f"exact-match {va_exact:.3f} ({time.time() - t0:.0f}s)",
              flush=True)

    bb = bb.cpu().eval()
    sd_out = bb.state_dict()
    sd_out["ft_cls_head.weight"] = head.weight.data.cpu()
    sd_out["ft_cls_head.bias"] = head.bias.data.cpu()
    torch.save(sd_out, CKPT)
    print(f"saved updated encoder checkpoint -> {CKPT}")
    return va_exact


# --------------------------------------------------------------------------
# phase 1: feature cache
# --------------------------------------------------------------------------

@torch.no_grad()
def build_features(device: str) -> dict[str, np.ndarray]:
    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone().to(device)
    if CKPT.exists():
        sd = torch.load(CKPT, map_location=device, weights_only=True)
        bb.load_state_dict(sd, strict=False)
    bb.eval()

    manifest = json.loads(MANIFEST.read_text())
    feats, labels, pids = [], [], []
    t0 = time.time()
    for i, row in enumerate(available_patches(manifest)):
        bands = load_patch_tensor(row["patch_id"])
        batched = {s: bands[s].to(device) for s in SLOT_ORDER}
        x = bb.band_proj(batched)
        f = bb.opt_enc(x).flatten(1)
        f = bb.gsd_cond(f, gsd=10.0)
        feats.append(f.cpu().numpy().astype(np.float16))
        pids.append(row["patch_id"])
        if (i + 1) % 500 == 0:
            print(f"  features {i + 1}/{len(manifest['patches'])} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out = {"pid": np.array(pids),
           "feat": np.concatenate(feats, 0)}
    np.savez_compressed(FEAT_OUT, **out)
    print(f"feature cache saved -> {FEAT_OUT} ({out['feat'].shape})")
    return out


# --------------------------------------------------------------------------
# phase 2: heads
# --------------------------------------------------------------------------

MCQ_SPLIT = re.compile(r"\s+([a-j]\))\s*")


def parse_vqa_example(r) -> tuple[str, list[str], int] | None:
    """Return (question, options, gold_index) or None if unparseable."""
    inp = str(r["input"])
    out = str(r["output"]).strip().lower()
    if r["type"] == "binary":
        if out not in ("yes", "no"):
            return None
        return inp, ["yes", "no"], 0 if out == "yes" else 1
    if r["type"] == "mcq":
        m = re.search(r"\s+([a-j]\)\s)", inp)
        if not m:
            return None
        stem = inp[:m.start()].strip()
        rest = " " + inp[m.start():].strip()
        parts = MCQ_SPLIT.split(rest)
        # parts: ['', 'a)', 'text', 'b)', 'text', ...]
        opts, letters = [], []
        for i in range(1, len(parts) - 1, 2):
            letters.append(parts[i].strip(")"))
            opts.append(parts[i + 1].strip())
        if not opts or out not in letters:
            return None
        return stem, opts, letters.index(out)
    return None


class VQAScorer(nn.Module):
    """Score (question, image feature, option) -> logit."""

    def __init__(self, dim: int, feat_dim: int = 2048, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + feat_dim + dim, hidden), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, q, f, o):
        return self.net(torch.cat([q, f, o], -1)).squeeze(-1)


def build_embeddings(examples):
    texts = []
    for stem, opts, _ in examples:
        texts.append(stem)
        texts.extend(opts)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                          max_features=8000, min_df=2)
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=256, random_state=42)
    E = svd.fit_transform(X)
    print(f"text embed: {X.shape} -> {E.shape} "
          f"(svd explains {svd.explained_variance_ratio_.sum():.1%})")
    return vec, svd, E


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune-encoder", action="store_true")
    ap.add_argument("--enc-epochs", type=int, default=3)
    ap.add_argument("--vqa-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.finetune_encoder:
        finetune_encoder(args.enc_epochs, 32)

    if FEAT_OUT.exists() and not args.rebuild_features:
        z = np.load(FEAT_OUT, allow_pickle=True)
        cache = {"pid": z["pid"], "feat": z["feat"]}
        print(f"loaded feature cache: {cache['feat'].shape}")
    else:
        cache = build_features(device)

    pid_to_idx = {str(p): i for i, p in enumerate(cache["pid"])}
    import pandas as pd
    df = pd.read_parquet(ANN)
    train_df = df[(df["split"] == "train") & df["type"].isin(["binary", "mcq"])]
    examples, keep = [], []
    for _, r in train_df.iterrows():
        ex = parse_vqa_example(r)
        if ex is None or str(r["patch_id"]) not in pid_to_idx:
            continue
        examples.append(ex)
        keep.append(r)
    print(f"parsed VQA examples: {len(examples)}/{len(train_df)}")

    vec, svd, E = build_embeddings(examples)
    # index text embeddings
    e_ptr, n_texts = 0, 0
    q_emb = np.zeros((len(examples), 256), dtype=np.float32)
    o_emb, o_ptr = [], []
    for i, (stem, opts, _) in enumerate(examples):
        q_emb[i] = E[e_ptr]; e_ptr += 1
        for j in range(len(opts)):
            o_emb.append(E[e_ptr]); e_ptr += 1
        o_ptr.append((n_texts, len(opts)))
        n_texts += len(opts)
    o_emb = np.asarray(o_emb, dtype=np.float32)
    f_emb = np.zeros((len(examples), 2048), dtype=np.float32)
    for i, r in enumerate(keep):
        f_emb[i] = cache["feat"][pid_to_idx[str(r["patch_id"])]].astype(np.float32)
    gold = np.array([ex[2] for ex in examples], dtype=np.int64)
    n_opts = np.array([len(ex[1]) for ex in examples], dtype=np.int64)
    opt_start = np.concatenate([[0], np.cumsum(n_opts)])[:-1]

    model = VQAScorer(256).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.vqa_epochs)

    q_t = torch.from_numpy(q_emb).to(device)
    f_t = torch.from_numpy(f_emb).to(device)
    o_t = torch.from_numpy(o_emb).to(device)
    gold_t = torch.from_numpy(gold).to(device)
    n_opts_t = torch.from_numpy(n_opts).to(device)
    opt_start_t = torch.from_numpy(opt_start).to(device)
    N = len(examples)

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    n_val = max(1000, int(0.08 * N))
    val_ix, tr_ix = perm[:n_val], perm[n_val:]

    def score_questions(ix: torch.Tensor) -> torch.Tensor:
        """Return (B, max_opts) logits, NaN-pad absent options."""
        k = int(n_opts_t[ix].max())
        rows = torch.arange(len(ix), device=ix.device)
        starts = opt_start_t[ix]
        offs = torch.arange(k, device=ix.device)[None]
        oix = starts[:, None] + offs
        valid = offs < n_opts_t[ix][:, None]
        oix = oix.clamp(max=o_t.shape[0] - 1)
        pair_q = q_t[ix].unsqueeze(1).expand(-1, k, -1)
        pair_f = f_t[ix].unsqueeze(1).expand(-1, k, -1)
        pair_o = o_t[oix.reshape(-1)].view(len(ix), k, -1)
        logits = model(pair_q.reshape(-1, 256),
                       pair_f.reshape(-1, 2048),
                       pair_o.reshape(-1, 256)).view(len(ix), k)
        return logits.masked_fill(~valid, float("-inf")), valid

    t0 = time.time()
    for ep in range(1, args.vqa_epochs + 1):
        model.train()
        order = rng.permutation(tr_ix)
        tot, seen = 0.0, 0
        for i in range(0, len(order), args.batch):
            ix = torch.from_numpy(order[i:i + args.batch]).long().to(device)
            logits, valid = score_questions(ix)
            loss = nn.functional.cross_entropy(logits, gold_t[ix])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tot += loss.item() * len(ix); seen += len(ix)
        sched.step()
        model.eval()
        with torch.no_grad():
            vix = torch.from_numpy(val_ix).long().to(device)
            logits, _ = score_questions(vix)
            acc = (logits.argmax(1) == gold_t[vix]).float().mean().item()
        print(f"vqa ep {ep}/{args.vqa_epochs} loss {tot / seen:.4f} "
              f"val acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # caption head on cached features (multi-label LULC)
    manifest = json.loads(MANIFEST.read_text())
    label_vocab = manifest["labels"]
    tr_rows = [r for r in available_patches(manifest)
               if r["split"] == "train"]
    cap_x = np.stack([cache["feat"][pid_to_idx[r["patch_id"]]].astype(np.float32)
                      for r in tr_rows], 0)
    cap_y = np.zeros((len(tr_rows), len(label_vocab)), dtype=np.float32)
    for i, r in enumerate(tr_rows):
        for l in r["labels"]:
            cap_y[i, label_vocab.index(l)] = 1.0
    cap_model = nn.Sequential(
        nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, len(label_vocab))).to(device)
    optim2 = torch.optim.AdamW(cap_model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.from_numpy(cap_x).to(device)
    Y = torch.from_numpy(cap_y).to(device)
    for ep in range(args.vqa_epochs):
        cap_model.train()
        order = rng.permutation(len(X))
        tot, seen = 0.0, 0
        for i in range(0, len(order), args.batch):
            ixt = torch.from_numpy(order[i:i + args.batch]).long()
            loss = nn.functional.binary_cross_entropy_with_logits(
                cap_model(X[ixt]), Y[ixt])
            optim2.zero_grad(set_to_none=True)
            loss.backward()
            optim2.step()
            tot += loss.item() * len(ixt); seen += len(ixt)
        if (ep + 1) % 4 == 0:
            print(f"caption ep {ep + 1} loss {tot / seen:.4f}", flush=True)

    HEADS_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vqa_scorer": model.state_dict(),
                "caption_head": cap_model.state_dict(),
                "embed_dim": 256}, HEADS_OUT)
    with open(EMBED_OUT, "wb") as f:
        pickle.dump({"vectorizer": vec, "svd": svd}, f)
    print(f"saved heads -> {HEADS_OUT}")

    metrics = {"n_vqa_examples": int(N),
               "n_vqa_options_total": int(n_opts.sum()),
               "vqa_val_accuracy": round(float(acc), 4),
               "n_caption_patches": len(tr_rows),
               "label_vocab_size": len(label_vocab)}
    (BEN / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics:", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())

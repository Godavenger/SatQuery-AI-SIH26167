"""Evaluate the trained heads on the official BigEarthNet.txt bench split.

The bench split is the manually verified VLM evaluation set (1,082 pairs,
15,029 annotations). This script scores every binary and MCQ annotation
with the trained VQA scorer over features encoded by the CURRENT backbone
(re-encoded on every run -- after the joint fine-tune the old
data/ben_txt/features.npz cache describes a checkpoint that no longer
exists), with MiniLM text embeddings for question stems and options.

Usage:
    python scripts/eval_ben_txt.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from train_ben_txt import (BAND_TO_SLOT, BEN, ROOT, SLOT_ORDER,
                            parse_vqa_example)
from train_joint import VQAScorerV2

HEADS_OUT = ROOT / "models" / "ben_txt_heads.pth"
BB_CKPT = ROOT / "models" / "backbone_ft.pth"
PATCH_DIR = BEN / "patches"


def encode_bench_patches(device: str) -> dict[str, np.ndarray]:
    """patch_id -> 2048-d feature from the current checkpoint (fresh pass
    every run; ~1020 patches costs well under a minute on the 4050 and
    the old cache is meaningless after any encoder fine-tune)."""
    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone().to(device)
    if BB_CKPT.exists():
        bb.load_state_dict(torch.load(BB_CKPT, map_location=device,
                                      weights_only=True), strict=False)
    bb.eval()

    manifest = json.loads((BEN / "manifest.json").read_text())
    rows = [r for r in manifest["patches"] if r["split"] == "bench"
            and (PATCH_DIR / f"{r['patch_id']}.npz").exists()]
    feats: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            arrs = []
            for r in chunk:
                z = np.load(PATCH_DIR / f"{r['patch_id']}.npz")
                arrs.append(np.stack(
                    [z[b] for b in BAND_TO_SLOT]).astype(np.float32) / 10000.0)
            x = torch.from_numpy(np.stack(arrs)).to(device)
            bands = {s: x[:, j] for j, s in enumerate(SLOT_ORDER)}
            with torch.autocast("cuda", enabled=device == "cuda"):
                f = bb.opt_enc(bb.band_proj(bands)).flatten(1)
                f = bb.gsd_cond(f, gsd=10.0)
            f = f.float().cpu().numpy()
            for r, v in zip(chunk, f):
                feats[r["patch_id"]] = v
            if (i + 32) % 320 == 0:
                print(f"  encoded {i + 32}/{len(rows)}", flush=True)
    print(f"bench features: {len(feats)} patches", flush=True)
    return feats


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(HEADS_OUT, map_location=device, weights_only=True)
    if blob.get("arch") != "v2":
        print(f"heads artifact is {blob.get('arch', 'v1')}; this evaluator "
              "expects the joint v2 run. Refusing to mix generations.")
        return 1
    model = VQAScorerV2(txt_dim=int(blob["embed_dim"])).to(device)
    model.load_state_dict(blob["vqa_scorer"])
    model.eval()

    feats = encode_bench_patches(device)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(blob.get(
        "text_encoder", "sentence-transformers/all-MiniLM-L6-v2")),
        device=device)

    df = pd.read_parquet(BEN / "annotations.parquet")
    bench = df[(df["split"] == "bench") & df["type"].isin(["binary", "mcq"])]
    print(f"bench VQA annotations: {len(bench)} "
          f"({bench['type'].value_counts().to_dict()})", flush=True)

    # score in batches: embed stems + options once, gather features, rank
    by_type = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    correct = total = missed = 0
    batch: list = []
    t_all = []

    def flush(batch):
        nonlocal correct, total, missed
        if not batch:
            return
        nonlocal t_all
        stems = [b[0] for b in batch]
        opts_flat = [o for b in batch for o in b[1]]
        q = st.encode(stems, batch_size=256, show_progress_bar=False)
        o = st.encode(opts_flat, batch_size=256, show_progress_bar=False)
        q_t = torch.from_numpy(np.asarray(q, dtype=np.float32)).to(device)
        o_t = torch.from_numpy(np.asarray(o, dtype=np.float32)).to(device)
        kmax = max(len(b[1]) for b in batch)
        f_rows, valid = [], []
        ptr = 0
        for b in batch:
            f_rows.append(b[3])
            valid.append([True] * len(b[1])
                         + [False] * (kmax - len(b[1])))
            ptr += len(b[1])
        f = torch.from_numpy(np.stack(f_rows)).to(device)
        valid_t = torch.tensor(valid, device=device)
        qe = q_t.unsqueeze(1).expand(-1, kmax, -1).reshape(-1, q_t.shape[-1])
        fe = f.unsqueeze(1).expand(-1, kmax, -1).reshape(-1, f.shape[-1])
        oix = []
        start = 0
        for b in batch:
            oix.extend(range(start, start + len(b[1])))
            oix.extend([start] * (kmax - len(b[1])))   # pad, masked below
            start += len(b[1])
        oix_t = torch.tensor(oix, device=device)
        oe = o_t[oix_t]
        with torch.no_grad():
            logits = model(qe, fe, oe).view(len(batch), kmax)
        logits = logits.masked_fill(~valid_t, -1e9)
        preds = logits.argmax(1).tolist()
        for b, pred in zip(batch, preds):
            ok = pred == b[2]
            correct += ok
            total += 1
            by_type[b[4]][0] += ok
            by_type[b[4]][1] += 1
            by_cat[b[5]][0] += ok
            by_cat[b[5]][1] += 1

    for _, r in bench.iterrows():
        ex = parse_vqa_example(r)
        f = feats.get(str(r["patch_id"]))
        if ex is None or f is None:
            missed += 1
            continue
        batch.append((ex[0], ex[1], ex[2], f.astype(np.float32),
                      r["type"], r["category"]))
        if len(batch) >= 512:
            flush(batch)
            batch = []
            if total % 2048 < 512:
                print(f"  scored {total} ({correct / max(total, 1):.3f})",
                      flush=True)
    flush(batch)

    acc = correct / max(total, 1)
    print("\n" + "=" * 60)
    print("BIGEARTHNET.TXT BENCH SPLIT -- VQA ACCURACY (v2)")
    print("=" * 60)
    print(f"overall: {acc:.4f}  ({correct}/{total}, {missed} unparseable)")
    for k, (c, n) in sorted(by_type.items()):
        print(f"  by type  {k:12s} {c / max(n, 1):.4f}  (n={n})")
    for k, (c, n) in sorted(by_cat.items()):
        print(f"  by cat   {k:12s} {c / max(n, 1):.4f}  (n={n})")
    print("=" * 60)

    metrics = {
        "benchmark": "BigEarthNet.txt bench split (manually verified)",
        "arch": "v2",
        "n_scored": total,
        "n_unparseable": missed,
        "accuracy": round(float(acc), 4),
        "accuracy_by_type": {k: round(c / max(n, 1), 4)
                             for k, (c, n) in by_type.items()},
        "accuracy_by_category": {k: round(c / max(n, 1), 4)
                                 for k, (c, n) in by_cat.items()},
    }
    out = ROOT / "data" / "ben_bench_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

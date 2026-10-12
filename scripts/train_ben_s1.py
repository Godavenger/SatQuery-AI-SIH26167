"""Fine-tune the SAR encoder branch on BigEarthNet v2 S1 patches.

Loads models/backbone_ft.pth (carrying the EuroSAT+BigEarthNet-S2 optical
fine-tune), trains ONLY the sar_enc branch on VV/VH -> BigEarthNet-19
multi-label LULC, and saves the checkpoint back so the fuse head's SAR
side benefits. The optical encoder is frozen throughout.

S1 v2 patches are delivered in dB scale; the SAR normalisation (dB mean
removal) matches the SAREncoder input contract used at inference
(sar_linear_to_db radiometry in Stage 0).

Usage:
    python scripts/train_ben_s1.py --epochs 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
BEN = ROOT / "data" / "ben_txt"
S1_DIR = BEN / "s1_patches"
MANIFEST = BEN / "manifest.json"
NAME_MAP = BEN / "s1_name_map.json"
CKPT = ROOT / "models" / "backbone_ft.pth"
METRICS_OUT = BEN / "s1_train_metrics.json"

# dB stats for S1 (BigEarthNet v2 interpolated to 120x120, from the
# official ben_txt_datamodule): subtract before the encoder.
VV_MEAN, VV_STD = -12.643863677978516, None
VH_MEAN, VH_STD = -19.352558135986328, None


def load_s1_tensor(s1_name: str) -> torch.Tensor:
    z = np.load(S1_DIR / f"{s1_name}.npz")
    vv = z["VV"].astype(np.float32)
    vh = z["VH"].astype(np.float32)
    vv = vv - VV_MEAN
    vh = vh - VH_MEAN
    return torch.from_numpy(np.stack([vv, vh], 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone().to(device)
    if CKPT.exists():
        sd = torch.load(CKPT, map_location=device, weights_only=True)
        bb.load_state_dict(sd, strict=False)
        print("loaded backbone_ft.pth (optical + prior fine-tunes)")

    # Freeze everything except the SAR encoder.
    for p in bb.parameters():
        p.requires_grad = False
    for p in bb.sar_enc.parameters():
        p.requires_grad = True

    manifest = json.loads(MANIFEST.read_text())
    label_vocab = manifest["labels"]
    name_map = json.loads(NAME_MAP.read_text())

    rows = []
    for r in manifest["patches"]:
        if r["split"] == "train" and r["patch_id"] in name_map and                 (S1_DIR / f"{name_map[r['patch_id']]}.npz").exists():
            rows.append(r)
    print(f"S1 training patches available: {len(rows)}")

    rng = np.random.RandomState(7)
    rng.shuffle(rows)
    n_val = max(200, int(0.1 * len(rows)))
    val_rows, tr_rows = rows[:n_val], rows[n_val:]

    head = nn.Linear(bb.sar_enc.out_dim, len(label_vocab)).to(device)
    optim = torch.optim.AdamW(
        [{"params": bb.sar_enc.parameters(), "lr": 1e-5},
         {"params": head.parameters(), "lr": 1e-3}],
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * max(1, len(tr_rows) // args.batch))

    def label_vec(r):
        v = np.zeros(len(label_vocab), dtype=np.float32)
        for l in r["labels"]:
            v[label_vocab.index(l)] = 1.0
        return v

    def run_epoch(data, train):
        bb.train(train); head.train(train)
        order = rng.permutation(len(data)) if train else np.arange(len(data))
        tot, seen, exact = 0.0, 0, 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xs = torch.stack([load_s1_tensor(name_map[data[j]["patch_id"]])
                              for j in idx]).to(device)
            ys = torch.from_numpy(np.stack([label_vec(data[j]) for j in idx])).to(device)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                feat = bb.sar_enc(xs[:, 0], xs[:, 1])   # (B, out_dim)
                logits = head(feat)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), ys)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(bb.sar_enc.parameters()) + list(head.parameters()), 1.0)
                optim.step()
                sched.step()
            tot += loss.item() * len(idx); seen += len(idx)
            exact += ((torch.sigmoid(logits) > 0.5) == ys.bool()).all(1).sum().item()
        return tot / max(seen, 1), exact / max(seen, 1)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_ex = run_epoch(tr_rows, True)
        va_loss, va_ex = run_epoch(val_rows, False)
        print(f"s1 ep {ep}/{args.epochs} train loss {tr_loss:.4f} "
              f"exact {tr_ex:.3f} | val loss {va_loss:.4f} exact {va_ex:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Save: optical weights frozen as-is, SAR encoder updated.
    bb = bb.cpu()
    sd_out = bb.state_dict()
    # preserve reference cls heads from previous phases if present
    if CKPT.exists():
        old = torch.load(CKPT, map_location="cpu", weights_only=True)
        for k in ("ft_cls_head.weight", "ft_cls_head.bias"):
            if k in old:
                sd_out[k] = old[k]
    sd_out["s1_cls_head.weight"] = head.weight.data.cpu()
    sd_out["s1_cls_head.bias"] = head.bias.data.cpu()
    torch.save(sd_out, CKPT)
    print(f"saved updated checkpoint (SAR fine-tuned) -> {CKPT}")

    metrics = {"n_train": len(tr_rows), "n_val": len(val_rows),
               "epochs": args.epochs,
               "val_exact_match": round(float(va_ex), 4),
               "val_loss": round(float(va_loss), 4),
               "label_vocab_size": len(label_vocab)}
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print("metrics:", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())

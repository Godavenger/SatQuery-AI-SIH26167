"""Fine-tune the adapted backbone on EuroSAT (Sentinel-2, 10 land-cover classes).

This closes the "backbone fine-tuning - not yet done" gap from CHECKPOINT.md
and produces a real benchmark number (EuroSAT held-out test accuracy) for the
submission's evaluation section.

What is trained:
    - opt_enc        : torchgeo SENTINEL2_ALL_MOCO ResNet50 (low LR, 1e-5)
    - band_proj      : band projector (with the encoder)
    - gsd_cond       : GSD conditioning MLP (higher LR, it is new)
    - cls head       : fresh Linear(2048, 10) for EuroSAT classes

EuroSAT's 13 bands map onto the backbone's 12 semantic slots in
BigEarthNet order; B10 (cirrus) has no slot and is dropped.

Saves:
    models/backbone_ft.pth      - AdaptedBackbone state_dict + cls head
    data/backbone_train_metrics.json

Usage:
    cd C:\\Users\\Sharanya\\Documents\\Hackathon\\SatQuery_AI_SIH26167
    python scripts/train_backbone.py --epochs 6 --batch 32
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "eurosat"
CKPT_OUT = ROOT / "models" / "backbone_ft.pth"
METRICS_OUT = ROOT / "data" / "backbone_train_metrics.json"

EUROSAT_CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
    "Industrial Buildings", "Pasture", "Permanent Crop",
    "Residential Buildings", "River", "Sea & Lake",
]

# EuroSAT 13-band order (B01..B12, B8A) -> BACKBONE_SLOTS index.
# BACKBONE_SLOTS = coastal, blue, green, red, rededge1..3, nir, nir08,
#                  nir09, swir1, swir2  (BigEarthNet order, no B10)
SLOT_FROM_EUROSAT_IDX = {
    0: 0,    # B01 coastal
    1: 1,    # B02 blue
    2: 2,    # B03 green
    3: 3,    # B04 red
    4: 4,    # B05 rededge1
    5: 5,    # B06 rededge2
    6: 6,    # B07 rededge3
    7: 7,    # B08 nir
    8: 9,    # B09 nir09
    9: None,  # B10 cirrus - no slot
    10: 10,  # B11 swir1
    11: 11,  # B12 swir2
    12: 8,   # B8A nir08
}


class EuroSATModel(nn.Module):
    """AdaptedBackbone + linear classifier, fed by raw 13-band DN tensors."""

    def __init__(self, bb):
        super().__init__()
        self.bb = bb
        self.head = nn.Linear(bb.FEAT_DIM, len(EUROSAT_CLASSES))

    def forward(self, x13: torch.Tensor) -> torch.Tensor:
        # x13: (B, 13, H, W) raw DN in [0, ~10000]. Scale to reflectance
        # here; BandProjector applies its own mean/std normalisation.
        x13 = x13.float() / 10000.0
        bands = {}
        for src_idx, slot_idx in SLOT_FROM_EUROSAT_IDX.items():
            if slot_idx is None:
                continue
            from satquery.stage2.backbone import BACKBONE_SLOTS
            bands[BACKBONE_SLOTS[slot_idx]] = x13[:, src_idx]
        feat = self.bb.band_proj(bands)          # (B, 12, H, W), normalised
        feat = self.bb.opt_enc(feat).flatten(1)  # (B, 2048)
        feat = self.bb.gsd_cond(feat, gsd=10.0)  # EuroSAT is 10 m GSD
        return self.head(feat)


def make_loaders(batch: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    from torchgeo.datasets import EuroSAT
    kw = dict(root=str(DATA_DIR), download=True, checksum=False)
    train = EuroSAT(split="train", **kw)
    val = EuroSAT(split="val", **kw)
    test = EuroSAT(split="test", **kw)
    print(f"EuroSAT sizes: train={len(train)} val={len(val)} test={len(test)}")
    common = dict(batch_size=batch, num_workers=2, pin_memory=True)
    return (DataLoader(train, shuffle=True, **common),
            DataLoader(val, **common),
            DataLoader(test, **common))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray]:
    model.eval()
    correct, n = 0, 0
    cm = np.zeros((10, 10), dtype=int)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        n += y.numel()
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    return correct / max(n, 1), cm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--enc-lr", type=float, default=1e-5,
                    help="LR for the pretrained optical encoder")
    ap.add_argument("--new-lr", type=float, default=1e-3,
                    help="LR for new modules (gsd conditioner, cls head)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    from satquery.stage2.backbone import AdaptedBackbone
    bb = AdaptedBackbone().to(device)
    model = EuroSATModel(bb).to(device)

    enc_params = [p for n_, p in model.named_parameters()
                  if n_.startswith("bb.opt_enc") or n_.startswith("bb.band_proj")]
    new_params = [p for n_, p in model.named_parameters()
                  if n_.startswith("bb.gsd_cond") or n_.startswith("head")]
    # SAR encoder is untouched (EuroSAT has no radar).
    optim = torch.optim.AdamW([
        {"params": enc_params, "lr": args.enc_lr},
        {"params": new_params, "lr": args.new_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * 100)  # stepped per 1% of training

    train_loader, val_loader, test_loader = make_loaders(args.batch)
    steps_per_epoch = len(train_loader)

    criterion = nn.CrossEntropyLoss()
    best_val, best_state = 0.0, None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for i, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running_loss += loss.item() * y.numel()
            seen += y.numel()
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} step {i + 1}/{steps_per_epoch} "
                      f"loss {running_loss / seen:.4f}", flush=True)

        val_acc, _ = evaluate(model, val_loader, device)
        train_loss = running_loss / max(seen, 1)
        print(f"epoch {epoch}/{args.epochs}  train loss {train_loss:.4f} "
              f" val acc {val_acc:.4f}  ({time.time() - t_start:.0f}s)",
              flush=True)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    # restore best-on-val weights and report test accuracy
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, cm = evaluate(model, test_loader, device)
    per_class = {EUROSAT_CLASSES[i]: round(cm[i, i] / max(cm[i].sum(), 1), 4)
                 for i in range(10)}
    print("\n" + "=" * 60)
    print(f"EUROSAT HELD-OUT TEST ACCURACY: {test_acc:.4f}")
    for c, a in per_class.items():
        print(f"  {c:24s} {a:.2%}")
    print("=" * 60)

    # Save AdaptedBackbone keys (bb.*) so get_backbone() can load them
    # directly; keep head weights under a namespaced key for reference.
    bb_state = {k[len("bb."):]: v for k, v in model.state_dict().items()
                 if k.startswith("bb.")}
    bb_state["ft_cls_head.weight"] = model.state_dict()["head.weight"]
    bb_state["ft_cls_head.bias"] = model.state_dict()["head.bias"]
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bb_state, CKPT_OUT)
    print(f"Saved backbone checkpoint -> {CKPT_OUT}")

    metrics = {
        "dataset": "EuroSAT (Sentinel-2, 10 classes, 13-band all-bands)",
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "epochs": args.epochs,
        "batch": args.batch,
        "encoder_init": "torchgeo SENTINEL2_ALL_MOCO ResNet50",
        "best_val_accuracy": round(float(best_val), 4),
        "test_accuracy": round(float(test_acc), 4),
        "per_class_recall": per_class,
        "train_seconds": round(time.time() - t_start, 1),
        "gsd_cond": "trained (new module)",
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {METRICS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

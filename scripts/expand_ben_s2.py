"""Expand the BigEarthNet.txt S2 working set with more train patches.

Pass 2 of the accuracy push (2026-09-11): the full 63.5 GB S2 archive is
back on disk, so the training set grows from ~6k to ~16k train patches
while the bench split (the official held-out evaluation) stays untouched.

What it does:
    1. snapshots the current manifest to manifest_v1.json (pass-1
       reference, needed to reconstruct the pass-1 validation patches
       so pass 2 validates on patches the warm-started model has
       never trained on)
    2. selects N new train patches, stratified by (country, snow) like
       the original fetch_ben_txt.py selection, excluding every patch
       already in the manifest
    3. streams the tar once and writes only the missing patches as npz
       (same 120x120 resample recipe as the original extraction)
    4. merges the new rows into data/ben_txt/manifest.json
    5. rebuilds data/ben_txt/annotations.parquet for the union of
       patch ids (streams the 0.47 GB HF parquet)
    6. writes data/ben_txt/val_pids.json -- the exact pass-1 validation
       patch ids, which train_joint.py --val-pids-file then reuses

The tar parts are NOT deleted here; that happens only after the
extraction is verified (patch counts on disk match the manifest).

Usage:
    python scripts/expand_ben_s2.py --add 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "data" / "ben_parts"
OUT_DIR = ROOT / "data" / "ben_txt"
PATCH_DIR = OUT_DIR / "patches"
MANIFEST = OUT_DIR / "manifest.json"
MANIFEST_V1 = OUT_DIR / "manifest_v1.json"
VAL_PIDS = OUT_DIR / "val_pids.json"

from fetch_ben_txt import BAND_TO_SLOT, _ConcatParts, hf_file  # noqa: E402


def select_new(n_add: int, existing: set[str]) -> list[dict]:
    import pandas as pd
    meta = pd.read_parquet(hf_file("torchgeo/bigearthnet", "V2/metadata.parquet"))
    meta = meta[["patch_id", "labels", "split", "country",
                 "contains_seasonal_snow"]]
    cand = meta[(meta["split"] == "train") & (~meta["patch_id"].isin(existing))]
    print(f"candidate new train patches: {len(cand)}")

    rng = np.random.RandomState(7)
    cand = cand.sample(frac=1.0, random_state=7)
    groups: dict[tuple, list] = defaultdict(list)
    for _, r in cand.iterrows():
        groups[(r["country"], bool(r["contains_seasonal_snow"]))].append(r)
    picked = []
    keys = list(groups.keys())
    i = 0
    while len(picked) < n_add:
        progressed = False
        for k in keys:
            if len(picked) >= n_add:
                break
            lst = groups[k]
            if i < len(lst):
                picked.append(lst[i])
                progressed = True
        if not progressed:
            break
        i += 1
    rows = [{"patch_id": r["patch_id"], "split": "train",
             "labels": list(r["labels"]), "country": r["country"]}
            for r in picked]
    print(f"selected {len(rows)} new patches across "
          f"{len(set(r['country'] for r in rows))} countries")
    return rows


def extract(patch_ids: set[str]) -> int:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile

    parts = sorted(PARTS_DIR.glob("BigEarthNet-S2.tar.gz*"))
    if len(parts) != 2:
        print(f"ERROR: expected 2 parts, found {len(parts)}")
        return 1
    wanted = {p for p in patch_ids
              if not (PATCH_DIR / f"{p}.npz").exists()}
    print(f"extracting {len(wanted)} new patches "
          f"({len(patch_ids) - len(wanted)} already on disk)")
    partial: dict[str, dict[str, np.ndarray]] = {}
    n_members = n_files = n_band_errors = 0

    stream = _ConcatParts(parts)
    tf = tarfile.open(fileobj=stream, mode="r|gz")
    for member in tf:
        n_members += 1
        if not member.isfile():
            continue
        name = member.name
        segs = name.strip("./").split("/")
        pid = next((s for s in segs if s in wanted), None)
        if pid is None:
            continue
        band = Path(name).stem.rsplit("_", 1)[-1].upper()
        if band not in BAND_TO_SLOT:
            n_band_errors += 1
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        try:
            buf = MemoryFile(f.read())
            with buf.open() as src:
                arr = src.read(1, out_shape=(120, 120), out_dtype="int32",
                               resampling=Resampling.bilinear)
        except Exception:                                       # noqa: BLE001
            n_band_errors += 1
            continue
        partial.setdefault(pid, {})[band] = arr.astype(np.uint16)
        if len(partial[pid]) == len(BAND_TO_SLOT):
            np.savez_compressed(PATCH_DIR / f"{pid}.npz", **partial.pop(pid))
            n_files += 1
            if n_files % 500 == 0:
                print(f"  {n_files} new patches after {n_members} members "
                      f"({n_band_errors} band errors)", flush=True)
    tf.close()
    n_ok = sum(1 for pid in wanted if (PATCH_DIR / f"{pid}.npz").exists())
    print(f"scan done: {n_members} members, {n_files} written, "
          f"{n_ok}/{len(wanted)} new patches complete, "
          f"{len(partial)} incomplete")
    return 0 if n_ok >= int(0.95 * len(wanted)) else 1


def subset_annotations(patch_ids: set[str]) -> None:
    import pyarrow.parquet as pq
    parquet_path = hf_file("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
                           "BigEarthNet.txt.parquet")
    pf = pq.ParquetFile(parquet_path)
    out_name = OUT_DIR / "annotations.parquet"
    writer = None
    n = 0
    try:
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i)
            mask = np.isin(t.column("patch_id").to_numpy(zero_copy_only=False),
                           list(patch_ids))
            if not mask.any():
                continue
            sel = t.filter(mask)
            if writer is None:
                writer = pq.ParquetWriter(out_name, t.schema)
            writer.write_table(sel)
            n += sel.num_rows
    finally:
        if writer:
            writer.close()
    print(f"annotations rebuilt: {n} rows -> {out_name}")


def reconstruct_val_pids() -> None:
    """Rebuild the exact pass-1 validation set from the v1 snapshot.

    train_joint.py pass 1 shuffled the on-disk train rows of the v1
    manifest with RandomState(123) and took the first max(50, 8%) as
    val. Replaying that on the snapshot yields the identical 467 patch
    ids, which pass 2 then reuses so validation stays honest under the
    warm start.
    """
    v1 = json.loads(MANIFEST_V1.read_text())
    rows = [r for r in v1["patches"] if r["split"] == "train"
            and (PATCH_DIR / f"{r['patch_id']}.npz").exists()]
    rng = np.random.RandomState(123)
    rng.shuffle(rows)
    n_val = max(50, int(0.08 * len(rows)))
    pids = [r["patch_id"] for r in rows[:n_val]]
    VAL_PIDS.write_text(json.dumps(pids, indent=1))
    print(f"val_pids.json written: {len(pids)} pass-1 validation patches")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--add", type=int, default=10000)
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    if not MANIFEST_V1.exists():
        import shutil
        shutil.copy2(MANIFEST, MANIFEST_V1)
        print(f"snapshot: manifest -> {MANIFEST_V1.name}")

    manifest = json.loads(MANIFEST.read_text())
    existing = {r["patch_id"] for r in manifest["patches"]}

    if args.extract_only:
        return extract(existing)

    new_rows = select_new(args.add, existing)

    rc = extract({r["patch_id"] for r in new_rows})
    if rc != 0:
        print("extraction incomplete; manifest NOT modified")
        return rc

    on_disk = {r["patch_id"] for r in new_rows
               if (PATCH_DIR / f"{r['patch_id']}.npz").exists()}
    manifest["patches"].extend(
        [r for r in new_rows if r["patch_id"] in on_disk])
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"manifest merged: {len(manifest['patches'])} total patches "
          f"({sum(1 for r in manifest['patches'] if r['split'] == 'train')} "
          f"train / {sum(1 for r in manifest['patches'] if r['split'] == 'bench')} bench)")

    subset_annotations({r["patch_id"] for r in manifest["patches"]})
    reconstruct_val_pids()
    print("extraction verified; tar parts can now be deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())

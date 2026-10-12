"""Build the BigEarthNet.txt working set: select a stratified subset, extract
only those patches from the 63.5 GB S2 tar stream, and subset annotations.

Disk strategy: the tar parts (63.5 GB) are read once as a stream; only the
selected patches are written (as per-patch .npz, ~350 KB each), then the
parts can be deleted.

Selection:
    - all 1,082 'bench' patches (the official VLM evaluation split)
    - ~6,000 'train' patches stratified by (country, season) for diversity

Outputs:
    data/ben_txt/manifest.json     - selected patch ids, labels, splits
    data/ben_txt/annotations.parquet - text annotations for selected patches
    data/ben_txt/patches/*.npz     - 12-band uint16 arrays per patch

Usage:
    python scripts/fetch_ben_txt.py            # select + extract + cleanup
    python scripts/fetch_ben_txt.py --select-only
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "data" / "ben_parts"
OUT_DIR = ROOT / "data" / "ben_txt"
PATCH_DIR = OUT_DIR / "patches"
STATS = ROOT / "data" / "ben_txt_patch_stats.parquet"

BAND_TO_SLOT = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08", "B09": "nir09",
    "B11": "swir1", "B12": "swir2",
}

N_TRAIN_TARGET = 6000


def hf_file(repo: str, fname: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, fname, repo_type="dataset"))


def select_patches() -> list[dict]:
    """Pick bench patches + stratified train patches with their labels."""
    import pandas as pd
    stats = pd.read_parquet(STATS)
    meta = pd.read_parquet(hf_file("torchgeo/bigearthnet", "V2/metadata.parquet"))
    meta = meta[["patch_id", "labels", "split", "country",
                 "contains_seasonal_snow", "contains_cloud_or_shadow"]]
    df = stats.merge(meta, on="patch_id", suffixes=("", "_meta"))
    df["ben_split"] = df["split"]

    bench = df[df["ben_split"] == "bench"]
    print(f"bench patches: {len(bench)}")

    train = df[df["ben_split"] == "train"].copy()
    # group by country+season via s2 name is unavailable; use country +
    # snow/cloud flags as diversity proxies, round-robin within groups.
    rng = np.random.RandomState(42)
    train = train.sample(frac=1.0, random_state=42)
    groups: dict[tuple, list] = defaultdict(list)
    for _, r in train.iterrows():
        key = (r["country"], bool(r["contains_seasonal_snow"]))
        groups[key].append(r)
    picked = []
    keys = list(groups.keys())
    i = 0
    while len(picked) < N_TRAIN_TARGET:
        progressed = False
        for k in keys:
            if len(picked) >= N_TRAIN_TARGET:
                break
            lst = groups[k]
            if i < len(lst):
                picked.append(lst[i])
                progressed = True
        if not progressed:
            break
        i += 1
    print(f"train patches picked: {len(picked)} "
          f"across {len(set(p['country'] for p in picked))} countries")

    rows = []
    for r in picked:
        rows.append({"patch_id": r["patch_id"], "split": "train",
                     "labels": list(r["labels"]),
                     "country": r["country"]})
    for _, r in bench.iterrows():
        rows.append({"patch_id": r["patch_id"], "split": "bench",
                     "labels": list(r["labels"]),
                     "country": r["country"]})

    # label vocabulary (BigEarthNet-19 nomenclature)
    lab = sorted({l for row in rows for l in row["labels"]})
    print(f"label vocabulary: {len(lab)} classes")
    manifest = {"patches": rows, "labels": lab,
                "band_order": list(BAND_TO_SLOT.keys())}
    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"saved manifest -> {OUT_DIR / 'manifest.json'}")
    return rows


def subset_annotations(patch_ids: set[str]) -> None:
    """Filter the 0.47 GB annotations parquet to the selected patches."""
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
    print(f"annotations subset: {n} rows -> {out_name}")


class _ConcatParts(io.RawIOBase):
    """A read-only file-like view over the two tar parts, sequentially."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.idx = 0
        self.f = open(paths[0], "rb")

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while True:
            chunk = self.f.read(len(b))
            if chunk:
                b[:len(chunk)] = chunk
                return len(chunk)
            self.idx += 1
            if self.idx >= len(self.paths):
                return 0
            self.f.close()
            self.f = open(self.paths[self.idx], "rb")

    def close(self) -> None:
        self.f.close()
        super().close()


def extract_patches(patch_ids: set[str]) -> int:
    """Stream the tar.gz and write selected patches as .npz files.

    v2 band tifs come at mixed native sizes (10m/20m/60m grids); every
    band is resampled to 120x120 to match the torchgeo loader and the
    backbone's band projector.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile

    parts = sorted(PARTS_DIR.glob("BigEarthNet-S2.tar.gz*"))
    if len(parts) != 2:
        print(f"ERROR: expected 2 parts, found {parts}")
        return 1
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(patch_ids)
    partial: dict[str, dict[str, np.ndarray]] = {}
    n_members, n_files, n_band_errors = 0, 0, 0

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
        # files are named <patch_id>_<BAND>.tif -> band is the last
        # underscore segment, not the whole stem
        band = Path(name).stem.rsplit("_", 1)[-1].upper()
        if band not in BAND_TO_SLOT:
            if n_band_errors < 3:
                print(f"  WARN unmatched band name in {name[-60:]}"
                      f" (parsed '{band}')", flush=True)
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
        except Exception:
            n_band_errors += 1
            continue
        partial.setdefault(pid, {})[band] = arr.astype(np.uint16)
        if len(partial[pid]) == len(BAND_TO_SLOT):
            np.savez_compressed(PATCH_DIR / f"{pid}.npz", **partial.pop(pid))
            n_files += 1
            if n_files % 500 == 0:
                print(f"  {n_files} patches complete after scanning "
                      f"{n_members} members "
                      f"({n_band_errors} band errors)", flush=True)
    tf.close()
    print(f"scan done: {n_members} members, {n_files} patches written, "
          f"{n_band_errors} band errors, "
          f"{len(partial)} incomplete")

    n_ok = sum(1 for pid in patch_ids
               if (PATCH_DIR / f"{pid}.npz").exists())
    print(f"complete patches: {n_ok}/{len(patch_ids)}")
    return 0 if n_ok >= int(0.95 * len(patch_ids)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="keep the 63.5 GB tar parts after extraction")
    args = ap.parse_args()

    rows = select_patches()
    patch_ids = {r["patch_id"] for r in rows}
    subset_annotations(patch_ids)

    if args.select_only:
        return 0

    rc = extract_patches(patch_ids)
    if rc == 0 and not args.no_cleanup:
        for p in sorted(PARTS_DIR.glob("BigEarthNet-S2.tar.gz*")):
            p.unlink()
            print(f"deleted {p.name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

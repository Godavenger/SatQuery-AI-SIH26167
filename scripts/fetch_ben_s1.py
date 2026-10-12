"""Extract the selected patches' S1 SAR bands from the 54.8 GB S1 tar
stream, then delete the parts.

Reads the same manifest as the S2 phase (data/ben_txt/manifest.json) and
writes data/ben_txt/s1_patches/<patch_id>.npz with VV/VH float32 arrays
resampled to 120x120 (dB-scaled as delivered in BigEarthNet v2).

Usage:
    python scripts/fetch_ben_s1.py
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "data" / "ben_parts"
OUT_DIR = ROOT / "data" / "ben_txt"
PATCH_DIR = OUT_DIR / "s1_patches"
MANIFEST = OUT_DIR / "manifest.json"

# BigEarthNet S1 files are VV.tif and VH.tif inside each patch dir.
BANDS = {"VV", "VH"}


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


def main() -> int:
    import io, json
    manifest = json.loads(MANIFEST.read_text())
    # S1 patch names differ from S2 patch ids: use the s1_name from the
    # torchgeo metadata instead. Fetch the mapping from the manifest rows
    # we stored (they carry patch_id == S2 name). We need s1_name, which
    # lives in the torchgeo metadata parquet.
    from huggingface_hub import hf_hub_download
    import pandas as pd
    meta_path = hf_hub_download("torchgeo/bigearthnet", "V2/metadata.parquet",
                                repo_type="dataset")
    meta = pd.read_parquet(meta_path, columns=["patch_id", "s1_name"])
    s1_of = dict(zip(meta["patch_id"], meta["s1_name"]))

    s2_dir = OUT_DIR / "patches"
    patch_ids = {r["patch_id"] for r in manifest["patches"]
                 if (s2_dir / f"{r['patch_id']}.npz").exists()}
    print(f"S2 patches available on disk: {len(patch_ids)}")
    s1_wanted = {s1_of[p] for p in patch_ids if p in s1_of}
    print(f"S1 patches wanted: {len(s1_wanted)}")

    parts = sorted(PARTS_DIR.glob("BigEarthNet-S1.tar.gz*"))
    # idempotent resume: if a full extraction already ran, parts were (mostly)
    # deleted and every wanted patch is on disk — just write the name map
    have = {p.stem for p in PATCH_DIR.glob("*.npz")}
    if len(parts) < 2 and have >= s1_wanted:
        print(f"extraction already complete: {len(have & s1_wanted)} patches on disk")
        (OUT_DIR / "s1_name_map.json").write_text(
            json.dumps({p: s1_of[p] for p in patch_ids if p in s1_of}, indent=1))
        return 0
    if len(parts) != 2:
        print(f"ERROR: expected 2 S1 parts, found {[p.name for p in parts]}")
        return 1
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    partial: dict[str, dict[str, np.ndarray]] = {}
    n_members, n_written, n_err = 0, 0, 0
    stream = _ConcatParts(parts)
    tf = tarfile.open(fileobj=stream, mode="r|gz")
    for member in tf:
        n_members += 1
        if not member.isfile():
            continue
        segs = member.name.strip("./").split("/")
        pid = next((s for s in segs if s in s1_wanted), None)
        if pid is None:
            continue
        # files are named <s1_name>_<VV|VH>.tif
        band = Path(member.name).stem.rsplit("_", 1)[-1].upper()
        if band not in BANDS:
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        try:
            buf = MemoryFile(f.read())
            with buf.open() as src:
                arr = src.read(1, out_shape=(120, 120), out_dtype="float32",
                               resampling=Resampling.bilinear)
        except Exception:
            n_err += 1
            continue
        partial.setdefault(pid, {})[band] = arr
        if len(partial[pid]) == len(BANDS):
            np.savez_compressed(PATCH_DIR / f"{pid}.npz", **partial.pop(pid))
            n_written += 1
            if n_written % 500 == 0:
                print(f"  {n_written} S1 patches written after {n_members} "
                      f"members", flush=True)
    tf.close()
    print(f"S1 scan done: {n_members} members, {n_written} patches, "
          f"{n_err} errors, {len(partial)} incomplete")

    # persist the S2->S1 name mapping for the trainer
    s1_subset = {p: s1_of[p] for p in patch_ids if p in s1_of}
    (OUT_DIR / "s1_name_map.json").write_text(
        json.dumps(s1_subset, indent=1))

    ok = n_written >= int(0.95 * len(s1_wanted))
    if ok:
        try:
            stream.close()   # tarfile.close() leaves a passed fileobj open — Windows won't delete a locked file
        except Exception:
            pass
        for p in parts:
            p.unlink()
            print(f"deleted {p.name}")
    else:
        print("NOT deleting parts (extraction incomplete)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

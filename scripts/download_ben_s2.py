"""Download the BigEarthNet v2.0 Sentinel-2 archive parts (63.5 GB total).

Downloads both split-gzip parts of the S2 archive from the torchgeo HF
mirror to data/ben_parts/ with byte-range resume, so a dropped connection
never restarts from zero. The parts are deleted by fetch_ben_txt.py after
selective extraction.

Usage:
    python scripts/download_ben_s2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "ben_parts"

REPO = "torchgeo/bigearthnet"
FILES = [
    ("V2/BigEarthNet-S2.tar.gzaa", 48.32e9),
    ("V2/BigEarthNet-S2.tar.gzab", 15.19e9),
]
CHUNK = 4 * 1024 * 1024


def url_for(fname: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/main/{fname}"


def download(fname: str, dest: Path) -> None:
    part_bytes = dest.stat().st_size if dest.exists() else 0
    done = part_bytes
    url = url_for(fname)
    while True:
        headers = {"Range": f"bytes={part_bytes}-"} if part_bytes else {}
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=60, allow_redirects=True) as r:
                if part_bytes and r.status_code == 200:
                    # server ignored the range; restart
                    r.close()
                    part_bytes = 0
                    continue
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + part_bytes
                mode = "ab" if part_bytes else "wb"
                t0, seen0 = time.time(), part_bytes
                with open(dest, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        part_bytes += len(chunk)
                if part_bytes == seen0:
                    return  # nothing new
        except Exception as e:
            print(f"[{fname}] error at {part_bytes/1e9:.2f} GB: {e}; "
                  f"retrying in 10s", flush=True)
            time.sleep(10)
            continue
        done = part_bytes
        print(f"[{fname}] complete: {done/1e9:.2f} GB", flush=True)
        return
    # unreachable


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, expected in FILES:
        dest = OUT_DIR / Path(fname).name
        if dest.exists() and dest.stat().st_size >= expected * 0.99:
            print(f"[{fname}] already complete", flush=True)
            continue
        print(f"[{fname}] downloading to {dest}", flush=True)
        download(fname, dest)
    print("ALL PARTS DOWNLOADED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

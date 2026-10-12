"""Unattended BigEarthNet.txt pipeline: S2 phase then S1 phase.

Chain (each step skipped if its output already exists, so the script is
safe to re-run after interruption):

  Phase 1 (S2 optical)
    0. wait for the S2 tar parts to finish downloading
    1. fetch_ben_txt.py       - extract 7,084 patches, delete parts
    2. train_ben_txt.py       - encoder fine-tune + VQA/caption heads
    3. eval_ben_txt.py        - official bench split accuracy
  Phase 2 (S1 SAR)
    4. download_ben_s1.py     - 54.8 GB parts
    5. fetch_ben_s1.py        - extract S1 patches, delete parts
    6. train_ben_s1.py        - fine-tune SAR encoder into backbone_ft.pth

Usage:
    python scripts/run_ben_pipeline.py            # full chain
    python scripts/run_ben_pipeline.py --phase 1 # just the S2 phase
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "ben_parts"
BEN = ROOT / "data" / "ben_txt"
S2_PART_SIZES = {"BigEarthNet-S2.tar.gzaa": 48.32e9,
                 "BigEarthNet-S2.tar.gzab": 15.19e9}
LOG = ROOT / "data" / "ben_pipeline.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(script: str, *args: str) -> bool:
    log(f"RUN python scripts/{script} {' '.join(args)}")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                       cwd=str(ROOT))
    ok = r.returncode == 0
    log(f"{'OK' if ok else 'FAILED'} {script} ({time.time() - t0:.0f}s)")
    return ok


def s2_download_complete() -> bool:
    for name, size in S2_PART_SIZES.items():
        p = PARTS / name
        if not p.exists() or p.stat().st_size < size * 0.99:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, default=0,
                    help="run only phase 1 (S2) or 2 (S1); 0 = both")
    ap.add_argument("--skip-encoder-ft", action="store_true")
    args = ap.parse_args()

    do_p1 = args.phase in (0, 1)
    do_p2 = args.phase in (0, 2)

    # ---------------- phase 1 ----------------
    if do_p1:
        n_ben_patches = len(list((BEN / "patches").glob("*.npz"))) \
            if (BEN / "patches").exists() else 0
        if n_ben_patches >= 7084 * 0.95:
            log(f"S2 patches already extracted ({n_ben_patches}); skipping")
        else:
            if not s2_download_complete():
                # download any missing/resumable parts ourselves rather
                # than polling an external downloader process
                if not run("download_ben_s2.py"):
                    return 1
                if not s2_download_complete():
                    log("S2 parts still incomplete after download run")
                    return 1
            if not run("fetch_ben_txt.py"):
                return 1

        if (BEN / "train_metrics.json").exists() and not args.skip_encoder_ft:
            log("training metrics exist; skipping head training "
                "(delete data/ben_txt/train_metrics.json to retrain)")
        else:
            ft = [] if args.skip_encoder_ft else ["--finetune-encoder"]
            if not run("train_ben_txt.py", *ft):
                return 1
        if not run("eval_ben_txt.py"):
            return 1

    # ---------------- phase 2 ----------------
    if do_p2:
        s1_patches = len(list((BEN / "s1_patches").glob("*.npz"))) \
            if (BEN / "s1_patches").exists() else 0
        if s1_patches >= 7084 * 0.95:
            log(f"S1 patches already extracted ({s1_patches}); skipping")
        else:
            if not run("download_ben_s1.py"):
                return 1
            if not run("fetch_ben_s1.py"):
                return 1
        if not run("train_ben_s1.py"):
            return 1

    log("PIPELINE COMPLETE")
    # summary
    for f in ("data/ben_txt/train_metrics.json",
              "data/ben_bench_metrics.json",
              "data/ben_txt/s1_train_metrics.json"):
        p = ROOT / f
        if p.exists():
            log(f"{f}: {p.read_text()[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

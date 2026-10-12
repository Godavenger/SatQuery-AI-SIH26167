"""Parallel-fetch CDVQA shards from the HF mirror with resume.

Each tar is ~80MB and holds ~3 unique image pairs asked ~45 questions
each. For a supervised scorer that actually generalizes we need hundreds
of UNIQUE pairs, i.e. hundreds of shards: 150 train + 80 test shards
give roughly 450 train pairs / 250 test pairs (~7k / 4k questions).

8 concurrent curls (HF throttles unauthenticated harder beyond that);
-C - resumes; already-complete shards are skipped.
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "bench"

JOBS = [
    ("train", BENCH / "cdvqa_train_shards", range(0, 150)),
    ("test", BENCH / "cdvqa_shards", range(20, 100)),
]
BASE = ("https://huggingface.co/datasets/ljx620/CDVQA/"
        "resolve/main/{split}/{split}-{i:05d}.tar")


def fetch(split, d, i):
    out = d / f"{split}-{i:05d}.tar"
    if out.exists() and out.stat().st_size > 60_000_000:
        return (i, "have", 0)
    for attempt in range(4):
        r = subprocess.run(
            ["curl", "-sL", "-C", "-", "--max-time", "1800",
             BASE.format(split=split, i=i), "-o", str(out)],
            capture_output=True)
        ok = out.exists() and out.stat().st_size > 60_000_000
        if ok:
            return (i, "done", out.stat().st_size)
        time.sleep(3 * (attempt + 1))
    return (i, "FAIL", out.stat().st_size if out.exists() else 0)


def main():
    tasks = [(s, d, i) for s, d, rng in JOBS for i in rng]
    print(f"{len(tasks)} shard fetches queued", flush=True)
    t0 = time.time()
    done = fail = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, s, d, i): (s, i) for s, d, i in tasks}
        for f in as_completed(futs):
            s, i = futs[f]
            try:
                idx, st, sz = f.result()
            except Exception as e:
                st, sz = "ERR", str(e)[:40]
            if st in ("done", "have"):
                done += 1
            else:
                fail += 1
            if done % 10 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} done, {fail} failed "
                      f"({el/60:.0f} min, {el/done:.0f}s/shard)", flush=True)
    print(f"BATCH3_DONE ok={done} fail={fail}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

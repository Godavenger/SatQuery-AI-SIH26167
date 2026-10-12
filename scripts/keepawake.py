"""Prevent Windows sleep while long training runs execute.

Calls SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) every
minute so the laptop does not suspend mid-run. Run it in the background
alongside training jobs; kill it when the session ends.

Usage:
    python scripts/keepawake.py
"""
from __future__ import annotations

import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def main() -> int:
    kernel32 = ctypes.windll.kernel32
    while True:
        r = kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if r == 0:
            print("SetThreadExecutionState failed", flush=True)
            return 1
        print(f"keep-awake active {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(60)

if __name__ == "__main__":
    raise SystemExit(main())

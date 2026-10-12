# SatQuery AI — Ask the Satellite

**Smart India Hackathon 2026 · Problem SIH26167 (ISRO / Space Technology)**

An agentic, evidence-first vision-language assistant for satellite imagery. Ask a
natural-language question over any location — *"Is there a water body here?"*,
*"What changed between these two dates?"*, *"Fuse the optical and radar views."* —
and get a calibrated, geo-anchored, auditable answer: an overlay on the map, a
confidence value, a machine-readable trace, and a one-click PDF report.

Six capabilities behind one interface: **VQA, scene description, grounding/detection,
change detection, change-QA, and optical–SAR fusion**, served by a six-stage pipeline
(ingest → agentic controller → specialist heads → calibrated integration → trace/report →
FastAPI backend + 3-D globe frontend).

## Requirements

- **Python 3.12** (verified on 3.12.10)
- A CUDA 12.1 GPU is optional; the router and calibrated heads run on CPU.
- See `requirements.txt`. For a GPU build, install torch first:
  ```
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```
  CPU-only: `pip install -r requirements.txt` (pulls CPU torch wheels).

## Run the demo

```bash
# 1. start the backend
python -m uvicorn satquery.api:app --port 7860

# 2. open the frontend (auto-detects the backend on :7860 and switches to live mode)
start docs/index.html          # Windows
```

Or double-click **`START_DEMO.bat`** (Windows one-click launcher).

> The backend is **unauthenticated and intended for localhost/demo use only** — do not
> expose port 7860 to an untrusted network without adding access control.

> The 3-D globe streams live imagery from NASA GIBS / Esri / USGS / OpenStreetMap, so the
> **Earth view needs an internet connection**. The analyst pipeline (ingest → heads →
> report) runs fully offline once tiles are cached; `satquery/stage2/onnx_infer.py` is a
> torch-free int8-ONNX land-cover path for CPU/offline deployment.

## Data & models (not in the repo)

The large binary assets are fetched separately, not committed:

- `data/{raw,tile_cache,eurosat,bench,ben_txt,ben_parts}/` — satellite tiles, benchmark
  shards and feature caches, acquired via `scripts/fetch_*.py`.
- `models/*.pth`, `models/onnx/*.onnx`, `models/*.pkl` — trained encoders, heads, the
  router, and exported ONNX graphs.

The small `data/*.json` metrics files **are** tracked — they back every number in the
report PDFs under `docs/`.

## Evaluation

```bash
python scripts/run_eval.py --quick                 # pipeline scorecard
python scripts/eval_benchmarks.py --bench all --mode both   # RSVQA-LR / VRSBench / CDVQA
python scripts/run_ablations.py                    # the four design ablations
python scripts/fit_calibration.py                  # confidence calibration (ECE)
```

Benchmark results and the ablation/calibration/extended-coverage study are compiled into
`docs/SatQuery_AI_Benchmark_Results.pdf`; the build summary is
`docs/SatQuery_AI_Completion_Report.pdf`.

## Repository layout

| Path | What it is |
|------|------------|
| `satquery/` | the system — `contracts.py`, `pipeline.py`, `api.py`, `stage0/`–`stage4/` |
| `scripts/` | training, evaluation, data-fetch and report-generation scripts |
| `docs/` | `index.html` frontend, generated PDFs, pitch docs, vendored offline assets |
| `models/` | trained weights, router, ONNX graphs |
| `data/` | metrics JSONs (tracked) + large binary caches (fetched) |
| `CHECKPOINT.md` | dated internal build log |

## License

See `LICENSE`. Vendored third-party assets under `docs/vendor/` (three.js, Inter &
Geist Mono fonts) retain their own licenses (MIT / SIL OFL).

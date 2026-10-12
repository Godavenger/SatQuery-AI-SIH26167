# SatQuery AI — Lead Working Demo (SIH26167)

## Run (10 seconds, offline-capable)
1. Double-click `index.html` — no install. 3D globe uses three.js CDN when online (2D fallback offline); satellite map uses Esri World Imagery when online.
2. Or `python server.py` → http://localhost:8000
3. For real inference, run `uvicorn satquery.api:app --port 7860`, then reopen `docs/index.html`; the header changes to **AI LIVE**.
3. Drag globe → pick a mission site → scroll the satellite map down to houses (z18, Delhi is clearest) → open a feature tab → Run.

## What the lead sees
- **3 input modes**: Single optical / Optical+SAR fusion / Bi-temporal before-after (synthetic canvas tiles stand in for BigEarthNet S1/S2 while real weights train).
- **5 mandatory queries** wired to a rule-based Controller v1 (keyword classifier — roadmap-approved for internal round).
- **6-step trace** lights up live: understand → check → pick → run → fuse → proof. Log format `task|model|params|output` = what ISRO judges audit.
- **Registry**: VQA / Caption / Grounding / Change / Fusion cards go LIVE per query.
- **Evidence**: yellow grounding box, red change zone, confidence donut, downloadable report, `/query` JSON viewer.
- **Historical comparison**: asking what changed over a year (or for the oldest view) resolves the earliest verified Sentinel-2 scene, pairs it with the latest, and returns a side-by-side image with spectral-change pixels marked.
- **Evidence modes**: grounding supports **Box**, **SHAP**, and **LIME**. SHAP falls back to a Numba-free Shapley superpixel approximation if the installed SHAP release is incompatible with the current NumPy.
- **Query-aware labels**: vegetation, built-up, water, and other supported targets use the class requested in the query instead of reusing the default water label.
- **Upload**: ⬆ Upload image button loads a PNG/JPG into single-image mode for live questioning.
- **Feature tabs**: Explore / VQA / Caption / Ground / Change / Fusion — each tab is a separate test bench with its own input + run history. Shared state (site, mode, viewer, trace, confidence) persists across tabs; the main query bar auto-routes and jumps to the matching tab.
- **Satellite streets**: real Esri World Imagery map, site markers clickable, 🏠 flies to house level (z18).
- **Agent pipeline**: query → understand → check → task → model → run → fuse → conf → evidence → answer, with per-task output kind in the trace.

## Main work behind it (for team briefing)
1. **Adaptation (M2)** — RemoteCLIP QLoRA on BigEarthNet 10% → loss curve + adapter. Why: PS rejects generic LLM.
2. **Controller (M1)** — 5-way router + GeoTIFF validation + trace logger. Demo = rules; finale = transformer.
3. **Specialists (M3)** — LLaVA-RS, DINO-RS, ChangeFormer, Fusion-UNet. Each returns text+visual+confidence.
4. **Fusion (M1)** — confidence threshold + abstain; merges pair outputs.
5. **GUI+proof (M4/M6)** — overlay map + trace + PDF; offline USB fallback.

## Next steps for real build
- M5: replace canvas tiles with rasterio 512px BigEarthNet PNGs.
- M2/M3: swap `answers()` stubs for real checkpoints behind same `task` names.
- M1: swap `server.py` stub for FastAPI `/upload` + `/query` (contract already matches).

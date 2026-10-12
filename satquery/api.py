"""FastAPI backend for SatQuery AI (Stage 5 serving layer).

Endpoints match the contract in section 8 of the pipeline design:

    POST /upload           GeoTIFF or benchmark image -> tiles + manifest
    POST /query            natural-language query + tile refs -> answer
    GET  /report/{run_id}  downloadable PDF report
    GET  /health           readiness: lists loaded models with versions

CORS is fully open (allow_origins=["*"]) so the existing index.html can reach
the API from the file:// origin or any local port. This is intended for
localhost / demo use only — the service is unauthenticated, so do not expose it
to an untrusted network without adding access control and tightening CORS.

Run:
    cd C:\\Users\\Sharanya\\Documents\\Hackathon\\SatQuery_AI_SIH26167
    uvicorn satquery.api:app --port 7860 --reload
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rasterio

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = DATA / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

# in-memory run store (survives the session; lost on restart — acceptable for demo)
_RUN_STORE: dict[str, dict] = {}

# Run ids come from new_run_id() = "run_<ms>_<hex>". Validate any client-supplied
# run_id before using it to build a filesystem path, so a crafted value cannot
# escape the runs directory.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")


def _valid_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return run_id

app = FastAPI(
    title="SatQuery AI",
    description="Interactive Vision-Language Assistant for Multimodal Remote Sensing — SIH26167",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve the docs directory as static root (index.html lives there)
DOCS = ROOT / "docs"
if DOCS.exists():
    app.mount("/ui", StaticFiles(directory=str(DOCS), html=True), name="ui")


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


@app.get("/health")
def health():
    from satquery.stage1.registry import REGISTRY, registry_snapshot

    # Report the trained artifacts and whether they actually loaded, so a
    # reviewer can tell a live checkpoint from a silent fallback to the
    # closed-form spectral path.
    models: dict[str, Any] = {}
    try:
        from satquery.stage2.neural import get_neural
        models = get_neural().describe()
    except Exception as e:                                 # noqa: BLE001
        models = {"error": f"{type(e).__name__}: {e}"}
    try:
        from satquery.stage1.router import load_router
        r = load_router()
        models["router"] = {
            "name": r.name, "version": r.version,
            "loaded": getattr(r, "loaded", False),
            "metrics": getattr(r, "metrics", {}),
        }
    except Exception as e:                                 # noqa: BLE001
        models["router"] = {"loaded": False, "error": str(e)}

    return {
        "status": "ok",
        "time": time.time(),
        "version": "0.1.0",
        "tools": registry_snapshot(),
        "models": models,
        "data_manifest": str(DATA / "raw" / "manifest.json")
            if (DATA / "raw" / "manifest.json").exists() else None,
    }


# --------------------------------------------------------------------------
# /upload
# --------------------------------------------------------------------------


class UploadResponse(BaseModel):
    tile_id: str
    sensor_id: str
    gsd_m: float
    bands: list[str]
    shape: list[int]
    modality: str
    ts: Optional[str]
    clear_fraction: float
    cached_path: str
    warnings: list[str]
    log: list[dict]


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    profile_id: Optional[str] = Query(None, description="Sensor profile override"),
    ts: Optional[str] = Query(None, description="Acquisition timestamp ISO-8601"),
    benchmark_path: bool = Query(False, description="Set true for benchmark PNG/JPEG"),
):
    """Stage 0 ingest over HTTP. Accepts GeoTIFF (default) or PNG/JPEG (benchmark_path=true).

    Security:
    - file-size cap (400 MB)
    - band-count cap (16)
    - pixel-dimension cap (20 000 px)
    - decompression-bomb guard
    All caps enforced inside ``satquery.stage0.ingest.IngestLimits``.
    """
    suffix = Path(file.filename or "upload.tif").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        try:
            shutil.copyfileobj(file.file, tmp)
        finally:
            file.file.close()

    try:
        from satquery.stage0.ingest import ingest

        result = ingest(tmp_path, profile_id=profile_id, ts=ts,
                        benchmark_path=benchmark_path)
        tile = result.tiles[0]

        # cache the file under a content-addressed path
        h = tile.content_hash()
        cached = DATA / "uploads" / f"{h}{suffix}"
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path, cached)

        return UploadResponse(
            tile_id=tile.id,
            sensor_id=tile.sensor_id,
            gsd_m=round(tile.gsd, 4),
            bands=sorted(tile.bands),
            shape=list(tile.shape),
            modality=tile.modality,
            ts=tile.ts,
            clear_fraction=round(tile.clear_fraction, 4),
            cached_path=str(cached.relative_to(ROOT)),
            warnings=result.warnings,
            log=result.log,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)


# --------------------------------------------------------------------------
# /query
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    tile_paths: list[str]                  # paths relative to ROOT or absolute
    timestamps: Optional[list[Optional[str]]] = None
    profile_ids: Optional[list[Optional[str]]] = None
    params: Optional[dict[str, Any]] = None
    use_cache: bool = True
    # convenience fields used by the demo site picker
    site_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    layer: Optional[str] = None
    date: Optional[str] = None


class QueryResponse(BaseModel):
    run_id: str
    task: str
    model: str
    answer: str
    confidence: float
    abstained: bool
    output_kind: str
    visual: dict
    visual_url: Optional[str]
    comparison_url: Optional[str]
    raw: dict
    trace: dict
    report_url: str


# --------------------------------------------------------------------------
# on-demand imagery for ad-hoc pinned locations (shift-clicked points)
# --------------------------------------------------------------------------

#: window to search for a first-pass fetch: wide enough that every
#: location has some scene, tight enough that it is recent
PIN_WINDOW_T1 = ("2024-01-01", "2024-06-30")
#: offset window for the second tile of a bi-temporal pair
PIN_WINDOW_T2 = ("2024-07-01", "2024-12-31")

PIN_FETCH_SEMAPHORE = asyncio.Semaphore(1)   # one STAC fetch at a time

HISTORICAL_QUERY = re.compile(
    r"\b(over (?:a |one |the )?year|year(?:s)? or so|last year|past year|"
    r"oldest|earliest|latest|histor(?:y|ical)|since \d{4}|"
    r"from \d{4} to \d{4})\b", re.IGNORECASE)


def _historical_change_query(query: str) -> bool:
    return bool(HISTORICAL_QUERY.search(query or ""))


def _manifest_s2_paths(entry: dict[str, Any]) -> list[tuple[str, str]]:
    records = []
    for record in entry.get("s2", []):
        path = ROOT / record.get("path", "")
        if path.exists() and record.get("date"):
            records.append((str(record["date"]), str(path)))
    return sorted(records)


def _pair_spans_year(paths: list[str]) -> bool:
    if len(paths) < 2:
        return False
    dates = []
    for path in paths[:2]:
        try:
            with rasterio.open(path) as src:
                dates.append(str(src.tags().get("ACQUISITION_DATE", "")))
        except Exception:
            return False
    if not all(dates):
        return False
    try:
        first = time.mktime(time.strptime(dates[0], "%Y-%m-%d"))
        second = time.mktime(time.strptime(dates[1], "%Y-%m-%d"))
    except ValueError:
        return False
    return abs(second - first) >= 365.0 * 24 * 60 * 60


def _fetch_historical_pair(req: "QueryRequest") -> list[str]:
    """Fetch and register a verified oldest/latest Sentinel-2 pair."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import fetch_data as fd                                     # noqa

    site = req.site_id or "pin"
    mpath = DATA / "raw" / "manifest.json"
    try:
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    except Exception:
        manifest = {}
    entry = manifest.get(site) or {
        "site": site, "lat": req.lat, "lon": req.lon,
        "mode": "change", "s2": [], "s1": [], "pinned": True,
    }
    lat = float(entry.get("lat", req.lat))
    lon = float(entry.get("lon", req.lon))
    existing = {r.get("path") for r in entry.get("s2", [])}

    oldest_windows = [
        ("2015-06-23", "2017-12-31"),
        ("2018-01-01", "2020-12-31"),
    ]
    latest_windows = [
        ("2025-09-01", date.today().isoformat()),
        ("2024-01-01", date.today().isoformat()),
    ]
    fetched = []
    for tag, windows, order in (
            ("oldest", oldest_windows, "oldest"),
            ("latest", latest_windows, "latest")):
        for window in windows:
            try:
                record = fd.fetch_s2(
                    site, lat, lon, window, tag, order=order)
            except Exception as e:                             # noqa: BLE001
                print(f"[history {site}/{tag}] FAILED: {type(e).__name__}: {e}")
                record = None
            if record and record["path"] not in existing:
                entry.setdefault("s2", []).append(record)
                existing.add(record["path"])
                fetched.append(record)
                break

    entry["mode"] = "change"
    manifest[site] = entry
    try:
        mpath.write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        print(f"[history {site}] manifest update failed: {e}")

    paths = _manifest_s2_paths(entry)
    if len(paths) < 2:
        return []
    return [paths[0][1], paths[-1][1]]


def _classify_intent(query: str) -> str:
    """Offline intent guess so the fetcher knows which tiles to pull."""
    try:
        from satquery.stage1.router import load_router
        return load_router().route(query)[0]
    except Exception:
        return "describe"


def _fetch_pin_tiles(req: "QueryRequest") -> list[str]:
    """Fetch real Sentinel-2 (+SAR when needed) for an arbitrary lat/lon.

    Reuses the scripts/fetch_data.py STAC machinery and registers the
    result in the manifest under the pin id so follow-up questions on
    the same workspace window hit the cache instead of re-fetching.
    Returns absolute tile paths, [] when nothing could be fetched.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import fetch_data as fd                                             # noqa

    intent = _classify_intent(req.query or "describe")
    need_pair = intent in ("change", "change-qa")
    need_sar = intent == "fuse"

    site = req.site_id or "pin"
    entry = {"site": site, "lat": req.lat, "lon": req.lon, "mode":
             "change" if need_pair else "single", "bbox":
             fd.bbox_for(req.lat, req.lon), "s2": [], "s1": [],
             "pinned": True}

    windows = [PIN_WINDOW_T1, PIN_WINDOW_T2] if need_pair else [PIN_WINDOW_T1]
    try:
        for tag, win in zip(("t1", "t2"), windows):
            r = fd.fetch_s2(site, req.lat, req.lon, win, tag)
            if r:
                entry["s2"].append(r)
        if need_sar:
            r = fd.fetch_s1(site, req.lat, req.lon, PIN_WINDOW_T1, "t1")
            if r:
                entry["s1"].append(r)
    except Exception as e:                                 # noqa: BLE001
        print(f"[pin-fetch {site}] FAILED: {type(e).__name__}: {e}")
        return []

    paths = []
    for s2 in entry["s2"]:
        p = ROOT / s2["path"]
        if p.exists():
            paths.append(str(p))
    if need_sar:
        for s1 in entry["s1"]:
            p = ROOT / s1["path"]
            if p.exists():
                paths.append(str(p))

    if not paths:
        return []

    # register in the manifest so subsequent queries on this pin resolve
    # like any mission site
    mpath = DATA / "raw" / "manifest.json"
    try:
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        manifest[site] = entry
        mpath.write_text(json.dumps(manifest, indent=2))
    except Exception as e:                                 # noqa: BLE001
        print(f"[pin-fetch {site}] manifest update failed: {e}")
    return paths


async def _resolve_tiles_async(req: QueryRequest) -> list[str]:
    """Manifest/cache resolution first; on-demand fetch for new pins.

    The on-demand path is guarded by a semaphore (single-flight): the
    frontend can retry while a fetch is in flight and will wait for it
    rather than firing a duplicate STAC crawl.
    """
    cached = _resolve_tile_paths(req)
    if cached and _historical_change_query(req.query):
        if _pair_spans_year(cached):
            return cached
        loop = asyncio.get_running_loop()
        async with PIN_FETCH_SEMAPHORE:
            historical = await loop.run_in_executor(
                None, _fetch_historical_pair, req)
            if historical:
                return historical
    if cached:
        return cached
    if req.lat is None or req.lon is None:
        return []

    loop = asyncio.get_running_loop()
    async with PIN_FETCH_SEMAPHORE:
        # re-check inside the lock: an earlier holder may have just
        # registered this pin
        cached = _resolve_tile_paths(req)
        if cached:
            return cached
        return await loop.run_in_executor(None, _fetch_pin_tiles, req)


# --------------------------------------------------------------------------
# rendered visual evidence
# --------------------------------------------------------------------------

_RUN_MEDIA: dict[str, dict[str, bytes]] = {}


def _normalise_rgb(arrays: list[Any]) -> Any:
    import numpy as np
    rgb = np.stack(arrays, axis=-1).astype(np.float32)
    finite = np.isfinite(rgb)
    if not finite.any():
        raise ValueError("tile contains no finite pixels")
    lo, hi = np.percentile(rgb[finite], [2.0, 98.0])
    if hi - lo < 1e-6:
        lo, hi = float(rgb[finite].min()), float(rgb[finite].max()) + 1.0
    rgb[~finite] = lo
    return np.clip(((rgb - lo) / (hi - lo)) * 255.0, 0, 255).astype("uint8")


def _tile_rgb(path: str | Path) -> Any:
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        if src.count >= 12:
            indexes = (4, 3, 2)
        elif src.count >= 3:
            indexes = (3, 2, 1)
        else:
            indexes = (1, 1, 1)
        arrays = [src.read(index) for index in indexes]
    if len({arr.shape for arr in arrays}) != 1:
        import numpy as np
        shape = arrays[0].shape
        arrays = [np.resize(arr, shape) for arr in arrays]
    return _normalise_rgb(arrays)


def _overlay_evidence(rgb: Any, visual: dict[str, Any]) -> Any:
    import numpy as np
    from PIL import Image

    out = Image.fromarray(rgb, "RGB").convert("RGBA")
    overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    mask = visual.get("mask")
    if mask is not None:
        active = np.asarray(mask) > 0
        overlay[active] = (239, 68, 68, 115)
    attribution = visual.get("attribution")
    if attribution is not None:
        values = np.asarray(attribution, dtype=np.float32)
        if values.max() > 1.0:
            values = values / max(float(values.max()), 1e-6)
        hot = values > 0.45
        overlay[hot] = (245, 158, 11, 145)
    if overlay[..., 3].any():
        out = Image.alpha_composite(out, Image.fromarray(overlay, "RGBA"))
    return out.convert("RGB")


def _png_bytes(image: Any) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    pil_image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_ground_visual(tile_paths: list[str],
                          visual: Any) -> bytes | None:
    if not tile_paths or visual is None:
        return None
    rgb = _tile_rgb(tile_paths[0])
    vis = {
        "mask": getattr(visual, "mask", None),
        "attribution": getattr(visual, "attribution", None),
    }
    return _png_bytes(_overlay_evidence(rgb, vis))


def _render_comparison(tile_paths: list[str], visual: Any) -> bytes | None:
    if len(tile_paths) < 2:
        return None
    import numpy as np
    from PIL import Image, ImageDraw

    size = 480
    images = []
    for path in tile_paths[:2]:
        rgb = _tile_rgb(path)
        vis = {
            "mask": getattr(visual, "mask", None),
            "attribution": getattr(visual, "attribution", None),
        }
        image = _overlay_evidence(rgb, vis).resize(
            (size, size), Image.Resampling.LANCZOS)
        images.append(image)

    canvas = Image.new("RGB", (size * 2 + 24, size + 76), (4, 7, 17))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(images[0], (8, 44))
    canvas.paste(images[1], (size + 16, 44))
    draw.rectangle((size + 11, 44, size + 12, size + 44), fill=(38, 42, 57))

    dates = []
    for path in tile_paths[:2]:
        try:
            import rasterio
            with rasterio.open(path) as src:
                dates.append(src.tags().get("ACQUISITION_DATE", "unknown date"))
        except Exception:
            dates.append("unknown date")
    draw.text((12, 12), f"OLDEST VERIFIED — {dates[0]}", fill=(94, 234, 212))
    draw.text((size + 20, 12), f"LATEST — {dates[1]}", fill=(94, 234, 212))
    draw.rectangle((8, size + 52, 24, size + 68), fill=(239, 68, 68))
    draw.text((32, size + 51), "marked spectral change", fill=(226, 232, 240))
    draw.text((220, size + 51),
              "co-registered pair · verified STAC scene metadata",
              fill=(148, 163, 184))
    return _png_bytes(np.asarray(canvas))


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Run the full pipeline and return a QueryResponse.

    Tiles resolve in three steps: explicit tile_paths, the data manifest
    (mission sites and previously-fetched pins), and finally an on-demand
    Sentinel-2 fetch for ad-hoc pinned locations (shift-clicked points) so
    any point on Earth can be queried without a pre-planned fetch.
    """
    from satquery.pipeline import run_query

    tile_paths = await _resolve_tiles_async(req)
    if not tile_paths:
        raise HTTPException(status_code=422,
                            detail="No tile paths supplied or resolvable "
                                   "from site_id, and on-demand fetch for "
                                   "lat/lon found no imagery")

    try:
        trace = run_query(
            query=req.query,
            tile_paths=tile_paths,
            timestamps=req.timestamps,
            profile_ids=req.profile_ids,
            requested_params=req.params,
            use_cache=req.use_cache,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # persist for /report
    _RUN_STORE[trace.run_id] = trace.to_dict()
    run_file = RUNS / f"{trace.run_id}.json"
    run_file.write_text(json.dumps(trace.to_dict(), indent=2, default=str))

    ans = trace.answer
    media: dict[str, bytes] = {}
    if ans and ans.intent == "ground":
        try:
            rendered = _render_ground_visual(tile_paths, ans.visual)
            if rendered:
                media["visual"] = rendered
        except Exception as e:                                 # noqa: BLE001
            print(f"[visual {trace.run_id}] render failed: {e}")
    if ans and ans.intent in ("change", "change-qa") and len(tile_paths) >= 2:
        try:
            rendered = _render_comparison(tile_paths, ans.visual)
            if rendered:
                media["comparison"] = rendered
        except Exception as e:                                 # noqa: BLE001
            print(f"[comparison {trace.run_id}] render failed: {e}")
    if media:
        _RUN_MEDIA[trace.run_id] = media

    return QueryResponse(
        run_id=trace.run_id,
        task=ans.intent if ans else "",
        model=ans.contributing[0] if ans and ans.contributing else "",
        answer=ans.text if ans else "",
        confidence=round(float(ans.confidence), 4) if ans else 0.0,
        abstained=ans.abstained if ans else True,
        output_kind=ans.output_kind if ans else "text",
        visual=ans.visual.summary() if ans else {},
        visual_url=f"/runs/{trace.run_id}/visual" if "visual" in media else None,
        comparison_url=f"/runs/{trace.run_id}/comparison" if "comparison" in media else None,
        raw={k: v for k, v in (ans.raw or {}).items()
             if not isinstance(v, (bytes, bytearray))},
        trace=trace.to_dict(),
        report_url=f"/report/{trace.run_id}",
    )


@app.get("/runs/{run_id}/visual")
def get_run_visual(run_id: str):
    _valid_run_id(run_id)
    png = _RUN_MEDIA.get(run_id, {}).get("visual")
    if png is None:
        raise HTTPException(status_code=404, detail="Visual evidence not found")
    return Response(png, media_type="image/png")


@app.get("/runs/{run_id}/comparison")
def get_run_comparison(run_id: str):
    _valid_run_id(run_id)
    png = _RUN_MEDIA.get(run_id, {}).get("comparison")
    if png is None:
        raise HTTPException(status_code=404, detail="Comparison image not found")
    return Response(png, media_type="image/png")


@app.get("/report/{run_id}")
def get_report(run_id: str):
    """Render the PDF report for a completed run."""
    _valid_run_id(run_id)
    from satquery.contracts import TraceRecord, Answer
    from satquery.stage4.report import build_pdf

    # try memory first, then disk
    td = _RUN_STORE.get(run_id)
    if td is None:
        path = RUNS / f"{run_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        td = json.loads(path.read_text())

    # reconstruct a minimal TraceRecord from the dict for the PDF renderer
    from satquery.contracts import TraceRecord, TraceStep, Answer, Visual
    tr = TraceRecord(run_id=run_id, query=td.get("query", ""),
                     started_at=td.get("started_at", 0),
                     finished_at=td.get("finished_at"),
                     tiles=td.get("tiles", []),
                     software=td.get("software", {}))
    for sd in td.get("steps", []):
        step = TraceStep(seq=sd.get("seq", 0), stage=sd.get("stage", 0),
                         name=sd.get("name", ""))
        for k, v in sd.items():
            if hasattr(step, k):
                try:
                    setattr(step, k, v)
                except Exception:
                    pass
        tr.steps.append(step)

    if td.get("answer"):
        ad = td["answer"]
        vis = Visual()
        tr.answer = Answer(
            text=ad.get("text", ""),
            confidence=float(ad.get("confidence", 0)),
            intent=ad.get("intent", ""),
            output_kind=ad.get("output_kind", "text"),
            abstained=bool(ad.get("abstained", False)),
            abstain_reason=ad.get("abstain_reason", ""),
            visual=vis,
            raw=ad.get("raw", {}),
            calibration=ad.get("calibration", {}),
            contributing=ad.get("contributing", []),
        )

    try:
        pdf = build_pdf(tr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF render failed: {e}")

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="satquery_{run_id}.pdf"'},
    )


@app.get("/sites")
def list_sites():
    """Manifest of available demo sites with data."""
    mpath = DATA / "raw" / "manifest.json"
    if not mpath.exists():
        return []
    return json.loads(mpath.read_text())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _resolve_tile_paths(req: QueryRequest) -> list[str]:
    """Resolve tile_paths from the request, falling back to manifest lookup."""
    if req.tile_paths:
        return [
            str(ROOT / p) if not Path(p).is_absolute() else p
            for p in req.tile_paths
        ]

    if not req.site_id:
        return []

    mpath = DATA / "raw" / "manifest.json"
    if not mpath.exists():
        return []

    manifest = json.loads(mpath.read_text())
    entry = manifest.get(req.site_id)
    if not entry:
        return []

    intent = _classify_intent(req.query or "describe")
    if intent in ("change", "change-qa"):
        s2_paths = _manifest_s2_paths(entry)
        if len(s2_paths) >= 2:
            # Historical questions deliberately use the widest available pair;
            # ordinary change questions use the earliest/latest cached pair.
            if _historical_change_query(req.query):
                return [s2_paths[0][1], s2_paths[-1][1]]
            return [s2_paths[0][1], s2_paths[-1][1]]

    paths = []
    for s2 in entry.get("s2", []):
        p = ROOT / s2["path"]
        if p.exists():
            paths.append(str(p))
            break

    if len(paths) < 2:
        for s1 in entry.get("s1", []):
            p = ROOT / s1["path"]
            if p.exists():
                paths.append(str(p))

    return paths

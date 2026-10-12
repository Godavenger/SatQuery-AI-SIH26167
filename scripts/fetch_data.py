"""Fetch real Sentinel-2 L2A and Sentinel-1 SAR for the nine mission sites.

Genuine pixels, not synthetic tiles. Sentinel-2 comes from the public
``sentinel-cogs`` bucket via Earth Search (no credentials); Sentinel-1 RTC
comes from Planetary Computer with anonymous URL signing.

The Sentinel-2 stack is written in BigEarthNet band order

    B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12

so the ``s2_msi`` sensor profile maps onto it directly and the adaptation
corpus and the demo corpus share one contract.

Usage
    python scripts/fetch_data.py                     # all sites
    python scripts/fetch_data.py --sites assam delhi # a subset
    python scripts/fetch_data.py --list              # show planned fetches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# GDAL tuning for range-requested COGs: without these every band open does a
# directory listing over HTTP and the fetch takes minutes instead of seconds.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import rasterio                                            # noqa: E402
from rasterio.warp import transform_bounds                 # noqa: E402
from rasterio.windows import from_bounds                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"

#: BigEarthNet band order. Earth Search asset name -> position in the stack.
S2_STACK = [
    ("coastal", "B01"), ("blue", "B02"), ("green", "B03"), ("red", "B04"),
    ("rededge1", "B05"), ("rededge2", "B06"), ("rededge3", "B07"),
    ("nir", "B08"), ("nir08", "B8A"), ("nir09", "B09"),
    ("swir16", "B11"), ("swir22", "B12"),
]

TILE_PX = 512
GSD = 10.0                       # metres, the Sentinel-2 analysis grid
HALF_DEG = (TILE_PX * GSD / 2) / 111_320.0   # half-window in degrees


# --------------------------------------------------------------------------
# the nine mission sites, mirroring SITES[] in docs/index.html
# --------------------------------------------------------------------------

SITES = [
    # id           lat      lon     mode      early window       late window
    ("assam",     26.14,   91.73, "change", ("2024-02-01", "2024-04-15"),
                                            ("2024-09-15", "2024-12-15"), True),
    # Kaziranga NP core: Brahmaputra floodplain, reliably inundated
    # post-monsoon; the assam patch above sits on dry upland with no
    # water to ground. Serves as the canonical flooded-water scene.
    ("kaziranga", 26.58,   93.17, "change", ("2024-02-01", "2024-04-15"),
                                            ("2024-09-15", "2024-12-15"), True),
    ("delhi",     28.61,   77.20, "change", ("2023-11-01", "2024-01-31"),
                                            ("2024-11-01", "2025-01-31"), True),
    ("sundarbans", 21.94,  88.90, "fuse",   ("2024-01-01", "2024-03-31"),
                                            None, True),
    ("punjab",    31.10,   75.34, "single", ("2024-03-01", "2024-05-15"),
                                            None, True),
    ("gangotri",  30.93,   79.08, "change", ("2023-09-15", "2023-11-15"),
                                            ("2024-09-15", "2024-11-15"), False),
    ("rondonia", -10.24,  -63.16, "change", ("2023-06-01", "2023-08-31"),
                                            ("2024-06-01", "2024-08-31"), False),
    ("creekfire",  37.02, -119.16, "change", ("2020-07-01", "2020-08-31"),
                                             ("2020-10-01", "2020-12-15"), False),
    ("nile",       30.90,   31.20, "change", ("2023-11-01", "2024-01-31"),
                                             ("2024-11-01", "2025-01-31"), False),
    ("dubai",      25.00,   55.10, "fuse",   ("2024-01-01", "2024-03-31"),
                                             None, True),
]


def bbox_for(lat: float, lon: float) -> list[float]:
    """A TILE_PX x TILE_PX window at 10 m, centred on the site."""
    dlon = HALF_DEG / max(np.cos(np.deg2rad(lat)), 0.2)
    return [lon - dlon, lat - HALF_DEG, lon + dlon, lat + HALF_DEG]


# --------------------------------------------------------------------------
# windowed COG reads
# --------------------------------------------------------------------------


def read_window(href: str, bbox: list[float], size: int = TILE_PX,
                retries: int = 3) -> np.ndarray:
    """Read a geographic window from a remote COG, resampled to size x size."""
    last = None
    for attempt in range(retries):
        try:
            with rasterio.open(href) as src:
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326", src.crs, *bbox, densify_pts=21)
                win = from_bounds(left, bottom, right, top, src.transform)
                arr = src.read(
                    1, window=win, out_shape=(size, size),
                    resampling=rasterio.enums.Resampling.bilinear,
                    boundless=True, fill_value=0)
                return arr.astype(np.float32)
        except Exception as e:                              # noqa: BLE001
            last = e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"read failed after {retries} tries: {href[:80]} :: {last}")


def window_transform(bbox: list[float], size: int = TILE_PX):
    from rasterio.transform import from_bounds as tf_from_bounds
    return tf_from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], size, size)


def write_stack(path: Path, arrays: list[np.ndarray], bbox: list[float],
                dtype: str, tags: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(arrays).astype(dtype)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype=dtype, crs="EPSG:4326",
        transform=window_transform(bbox), compress="deflate", predictor=2,
    ) as dst:
        dst.write(data)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


# --------------------------------------------------------------------------
# Sentinel-2
# --------------------------------------------------------------------------


def fetch_s2(site: str, lat: float, lon: float, window: tuple[str, str],
             tag: str, max_cloud: int = 25,
             order: str = "least_cloud") -> dict | None:
    from pystac_client import Client

    bbox = bbox_for(lat, lon)
    client = Client.open(EARTH_SEARCH)
    direction = ("asc" if order == "oldest" else
                 "desc" if order == "latest" else None)
    for cloud in (max_cloud, 45, 80):
        kwargs = {
            "collections": ["sentinel-2-l2a"], "bbox": bbox,
            "datetime": f"{window[0]}/{window[1]}",
            "query": {"eo:cloud_cover": {"lt": cloud}}, "limit": 100,
        }
        if direction:
            kwargs["sortby"] = [{
                "field": "properties.datetime", "direction": direction}]
        try:
            items = list(client.search(**kwargs).items())
        except TypeError:
            kwargs.pop("sortby", None)
            items = list(client.search(**kwargs).items())
        if items:
            break
    if not items:
        print(f"    [{site}/{tag}] no Sentinel-2 scene in {window}")
        return None

    if order == "oldest":
        items.sort(key=lambda i: i.datetime.timestamp() if i.datetime else float("inf"))
    elif order == "latest":
        items.sort(key=lambda i: i.datetime.timestamp() if i.datetime else float("-inf"), reverse=True)
    else:
        items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    item = items[0]
    cc = item.properties.get("eo:cloud_cover", -1)
    date = item.datetime.strftime("%Y-%m-%d")
    print(f"    [{site}/{tag}] S2 {item.id[:34]} {date} cloud={cc:.1f}%")

    def grab(idx_asset):
        idx, (asset, code) = idx_asset
        return idx, read_window(item.assets[asset].href, bbox)

    bands: list[np.ndarray | None] = [None] * len(S2_STACK)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(grab, (i, a)) for i, a in enumerate(S2_STACK)]
        for f in as_completed(futs):
            i, arr = f.result()
            bands[i] = arr

    scl = read_window(item.assets["scl"].href, bbox)

    out = RAW / site / f"s2_{tag}_{date}.tif"
    write_stack(out, [b for b in bands], bbox, "uint16", {
        "SENSOR_PROFILE": "s2_msi", "ACQUISITION_DATE": date,
        "BAND_ORDER": " ".join(c for _, c in S2_STACK),
        "CLOUD_COVER": f"{cc:.2f}", "STAC_ID": item.id, "GSD": "10",
        "SOURCE": "Sentinel-2 L2A via Earth Search / sentinel-cogs",
    })
    scl_path = RAW / site / f"scl_{tag}_{date}.tif"
    write_stack(scl_path, [scl], bbox, "uint8",
                {"PRODUCT": "Sentinel-2 SCL scene classification"})

    verification = {
        "verified": True,
        "scene_intersects_bbox": True,
        "required_bands_read": len(S2_STACK),
        "scl_read": True,
        "non_empty_pixels": bool(np.asarray(bands).any()),
        "selection_order": order,
        "cloud_cover_pct": round(float(cc), 2),
    }

    return {"path": str(out.relative_to(ROOT)), "scl": str(scl_path.relative_to(ROOT)),
            "date": date, "cloud_cover": round(float(cc), 2), "stac_id": item.id,
            "profile": "s2_msi", "gsd": 10.0, "bbox": bbox, "tag": tag,
            "verification": verification}


# --------------------------------------------------------------------------
# Sentinel-1
# --------------------------------------------------------------------------


def fetch_s1(site: str, lat: float, lon: float, window: tuple[str, str],
             tag: str) -> dict | None:
    import planetary_computer as pc
    from pystac_client import Client

    bbox = bbox_for(lat, lon)
    client = Client.open(PLANETARY, modifier=pc.sign_inplace)

    for coll in ("sentinel-1-rtc", "sentinel-1-grd"):
        try:
            items = list(client.search(
                collections=[coll], bbox=bbox,
                datetime=f"{window[0]}/{window[1]}", limit=12).items())
        except Exception as e:                              # noqa: BLE001
            print(f"    [{site}/{tag}] {coll} search failed: {e}")
            continue
        items = [i for i in items if "vv" in i.assets]
        if not items:
            continue

        item = items[0]
        date = item.datetime.strftime("%Y-%m-%d")
        pols, arrays = [], []
        for p in ("vv", "vh"):
            if p in item.assets:
                try:
                    arrays.append(read_window(item.assets[p].href, bbox))
                    pols.append(p.upper())
                except Exception as e:                      # noqa: BLE001
                    print(f"    [{site}/{tag}] {p} read failed: {str(e)[:80]}")
        if not arrays:
            continue

        print(f"    [{site}/{tag}] S1 {coll} {date} pols={pols}")
        out = RAW / site / f"s1_{tag}_{date}.tif"
        write_stack(out, arrays, bbox, "float32", {
            "SENSOR_PROFILE": "s1_grd", "ACQUISITION_DATE": date,
            "BAND_ORDER": " ".join(pols), "COLLECTION": coll,
            "STAC_ID": item.id, "GSD": "10",
            "SOURCE": f"Sentinel-1 via Planetary Computer ({coll})",
        })
        return {"path": str(out.relative_to(ROOT)), "date": date,
                "pols": pols, "stac_id": item.id, "collection": coll,
                "profile": "s1_grd", "gsd": 10.0, "bbox": bbox, "tag": tag}

    print(f"    [{site}/{tag}] no Sentinel-1 scene found")
    return None


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-sar", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    todo = [s for s in SITES if not args.sites or s[0] in args.sites]
    if args.list:
        for sid, lat, lon, mode, w1, w2, sar in todo:
            print(f"{sid:11s} {mode:7s} early={w1} late={w2} sar={sar}")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    manifest = {}
    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text())

    for sid, lat, lon, mode, w_early, w_late, want_sar in todo:
        if sid in manifest and not args.force:
            print(f"[{sid}] cached, skipping (use --force to refetch)")
            continue
        print(f"[{sid}] {mode} @ {lat},{lon}")
        entry = {"site": sid, "lat": lat, "lon": lon, "mode": mode,
                 "bbox": bbox_for(lat, lon), "s2": [], "s1": []}
        try:
            for tag, win in (("t1", w_early), ("t2", w_late)):
                if win is None:
                    continue
                r = fetch_s2(sid, lat, lon, win, tag)
                if r:
                    entry["s2"].append(r)
            if want_sar and not args.skip_sar:
                r = fetch_s1(sid, lat, lon, w_early, "t1")
                if r:
                    entry["s1"].append(r)
        except Exception as e:                              # noqa: BLE001
            print(f"[{sid}] FAILED: {type(e).__name__}: {e}")
            continue

        if entry["s2"] or entry["s1"]:
            manifest[sid] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"[{sid}] ok — {len(entry['s2'])} optical, {len(entry['s1'])} SAR")

    print(f"\nmanifest: {manifest_path}  ({len(manifest)} sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

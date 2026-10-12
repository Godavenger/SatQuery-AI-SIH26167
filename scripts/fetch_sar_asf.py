"""Fill the SAR data gap using ASF OPERA RTC-S1 (analysis-ready radar).

gangotri / rondonia / creekfire / nile have S2 optical but no S1 tile:
Planetary Computer's sentinel-1-rtc had no coverage there, so the fuse intent
degrades to optical-only. ASF hosts the OPERA RTC-S1 product -- radiometric-
terrain-corrected gamma-0 VV/VH as cloud-optimised GeoTIFFs -- which DOES
cover all four sites (confirmed: 8-50 scenes near each S2 date). This fetches
the temporally closest RTC scene per site, windows it to the site bbox, and
writes the same s1 tile format fetch_data.py produces (float32 linear power;
Stage-0 ingest converts to dB), then updates data/raw/manifest.json.

Auth -- OPERA RTC COGs are Earthdata-protected. Provide ONE of:
  * env EDL_TOKEN            Earthdata bearer token (preferred; no full
                             download -- COGs are windowed-read in place).
                             Make one at urs.earthdata.nasa.gov -> Generate
                             Token.
  * env EARTHDATA_USERNAME + EARTHDATA_PASSWORD   (asf_search downloads the
                             COGs to a temp dir, then they are windowed and
                             deleted).
  * ~/.netrc with a machine urs.earthdata.nasa.gov entry (same download path).

Usage:
    # in this session, so the token reaches the tool:
    ! set EDL_TOKEN=xxxxx   (PowerShell: $env:EDL_TOKEN="xxxxx")
    python scripts/fetch_sar_asf.py                 # all four gap sites
    python scripts/fetch_sar_asf.py --sites nile    # one site
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.transform import from_bounds as tf_from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"
TILE_PX = 512
GAP_SITES = ["gangotri", "rondonia", "creekfire", "nile"]


# ------------------------------------------------------------------- auth
def get_auth():
    """Return ('token', tok) | ('session', ASFSession) | (None, msg).

    Credentials are read from disk/env so they survive across separate tool
    invocations (unlike a shell `export`): EDL_TOKEN env, then a token file
    (~/.edl_token or ./.edl_token), then EARTHDATA_USERNAME/PASSWORD, then
    ~/.netrc. Prefer creating the file/netrc in your OWN terminal so the
    secret never lands in the session transcript.
    """
    tok = os.environ.get("EDL_TOKEN")
    if not tok:
        for f in (Path.home() / ".edl_token", ROOT / ".edl_token"):
            if f.exists():
                tok = f.read_text().strip()
                break
    if tok:
        return "token", tok.strip()
    import asf_search as asf
    user = os.environ.get("EARTHDATA_USERNAME")
    pw = os.environ.get("EARTHDATA_PASSWORD")
    try:
        if user and pw:
            return "session", asf.ASFSession().auth_with_creds(user, pw)
        if (Path.home() / ".netrc").exists():
            return "session", asf.ASFSession().auth_with_creds(  # netrc lookup
                *_netrc_creds())
    except Exception as e:                                       # noqa: BLE001
        return None, f"Earthdata auth failed: {e}"
    return None, ("no Earthdata credentials found (set EDL_TOKEN, or write "
                  "~/.edl_token, or EARTHDATA_USERNAME/PASSWORD, or ~/.netrc)")


def _netrc_creds():
    import netrc
    auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    if not auth:
        raise RuntimeError("no urs.earthdata.nasa.gov entry in ~/.netrc")
    return auth[0], auth[2]


# --------------------------------------------------------------- geometry
def _wkt(bb):
    w, s, e, n = bb
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _window_transform(bb):
    return tf_from_bounds(bb[0], bb[1], bb[2], bb[3], TILE_PX, TILE_PX)


def read_window(src, bb):
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", src.crs, *bb, densify_pts=21)
    win = from_bounds(left, bottom, right, top, src.transform)
    arr = src.read(1, window=win, out_shape=(TILE_PX, TILE_PX),
                   resampling=rasterio.enums.Resampling.bilinear,
                   boundless=True, fill_value=0)
    return arr.astype(np.float32)


def write_stack(path, arrays, bb, tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(arrays).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype="float32", crs="EPSG:4326",
        transform=_window_transform(bb), compress="deflate", predictor=2,
    ) as dst:
        dst.write(data)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


# --------------------------------------------------------- ASF search/read
def pick_scene(bb, date, window_days=60):
    import asf_search as asf
    c = dt.date.fromisoformat(date)
    res = asf.search(
        dataset="OPERA-S1", processingLevel="RTC", intersectsWith=_wkt(bb),
        start=(c - dt.timedelta(days=window_days)).isoformat(),
        end=(c + dt.timedelta(days=window_days)).isoformat(), maxResults=100)
    best, best_gap = None, 1e9
    for r in res:
        st = r.properties.get("startTime", "")[:10]
        try:
            gap = abs((dt.date.fromisoformat(st) - c).days)
        except ValueError:
            continue
        # need both VV and VH present
        urls = _vv_vh(r)
        if urls[0] and urls[1] and gap < best_gap:
            best, best_gap = (r, urls, st), gap
    return best, best_gap


def _vv_vh(r):
    all_urls = [r.properties.get("url", "")] + list(
        r.properties.get("additionalUrls", []) or [])
    vv = next((u for u in all_urls if u.endswith("_VV.tif")), None)
    vh = next((u for u in all_urls if u.endswith("_VH.tif")), None)
    return vv, vh


def _read_url(url, bb, auth):
    kind, cred = auth
    if kind == "token":
        env = rasterio.Env(
            GDAL_HTTP_HEADERS=f"Authorization: Bearer {cred}",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
        with env, rasterio.open(f"/vsicurl/{url}") as src:
            return read_window(src, bb)
    # session: download to temp, read, delete
    import asf_search as asf
    with tempfile.TemporaryDirectory() as td:
        fn = url.split("/")[-1]
        asf.download_url(url=url, path=td, filename=fn, session=cred)
        with rasterio.open(Path(td) / fn) as src:
            return read_window(src, bb)


def _db_stats(a):
    v = a[a > 0]
    if v.size == 0:
        return "all-zero"
    db = 10.0 * np.log10(v)
    return f"mean {db.mean():.1f} dB, [{np.percentile(db,5):.1f},{np.percentile(db,95):.1f}]"


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="*", default=GAP_SITES)
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true",
                    help="search + report coverage only, no auth/download")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    auth = (None, "dry-run")
    if not args.dry_run:
        auth = get_auth()
        if auth[0] is None:
            print(f"ERROR: {auth[1]}\n\nProvide an Earthdata login and re-run. "
                  "See this script's docstring for the exact steps.")
            return 2
        print(f"authenticated via {auth[0]}", flush=True)

    updated = 0
    for site in args.sites:
        entry = manifest.get(site)
        if not entry:
            print(f"[{site}] not in manifest; skip")
            continue
        bb = entry["bbox"]
        date = entry["s2"][0]["date"] if entry.get("s2") else "2024-01-01"
        got, gap = pick_scene(bb, date, args.window_days)
        if not got:
            print(f"[{site}] no OPERA RTC-S1 with VV+VH near {date}; skip")
            continue
        r, (vv_url, vh_url), sdate = got
        print(f"[{site}] S2={date} -> RTC {sdate} ({gap}d away) "
              f"{r.properties['sceneName']}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        vv = _read_url(vv_url, bb, auth)
        vh = _read_url(vh_url, bb, auth)
        tag = "t1"
        out = RAW / site / f"s1_{tag}_{sdate}.tif"
        write_stack(out, [vv, vh], bb, {
            "SENSOR_PROFILE": "s1_grd", "ACQUISITION_DATE": sdate,
            "SOURCE": "ASF OPERA RTC-S1", "POLARISATIONS": "VV,VH",
            "SCENE": r.properties["sceneName"]})
        print(f"       VV {_db_stats(vv)} | VH {_db_stats(vh)} "
              f"({time.time()-t0:.0f}s) -> {out.relative_to(ROOT)}", flush=True)
        rec = {"path": str(out.relative_to(ROOT)), "date": sdate,
               "tag": tag, "profile": "s1_grd", "gsd": 10.0, "bbox": bb,
               "source": "asf-opera-rtc-s1",
               "scene": r.properties["sceneName"]}
        entry.setdefault("s1", [])
        entry["s1"] = [x for x in entry["s1"] if x.get("tag") != tag] + [rec]
        updated += 1

    if updated and not args.dry_run:
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"\nupdated manifest for {updated} site(s) -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

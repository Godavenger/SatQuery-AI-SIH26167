"""Build a synthetic ISRO-style rehearsal set (analogous data).

The judge-set pairs (Cartosat-2S / RISAT) are undisclosed, so this synthesises
scenes with the *right shapes, dtypes and radiometric conventions* and pushes
them through the real Stage-0 profiles. It is explicitly a rehearsal of the
ingest path, not a stand-in for ISRO data.

Conventions mirrored from the real corpus:
  - optical DN is 10-bit (0..1023), matching CARTOSAT2S_PAN/MX scale 1/1023
  - SAR is linear sigma-nought power (S1 files here: p50 ~= 0.11, p99 ~= 2.7)
  - GeoTIFF in EPSG:4326, like data/raw/*

Footprint: every sensor covers the SAME ground extent (one shared bounding box)
at its OWN pixel size, so a pair passes the gate's IoU >= 0.90 while still
arriving at 0.65 / 1.0 / 1.6 m. Both halves are load-bearing: separate
footprints would reject every fusion case on geometry before exercising
anything downstream, and a shared pixel grid would erase the resolution gap
that GSD conditioning exists to handle.

Written to data/rehearsal/. Nothing existing is modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

OUT = Path(__file__).parents[1] / "data" / "rehearsal"

# A Bengaluru-suburbs-like pinned site: works for India-generalisation checks.
LON0, LAT0 = 77.59, 12.97

# Scene extent is fixed by the PAN sensor: 512 px at 0.65 m ~= 333 m a side.
# Every other sensor then gets however many pixels that extent is at its own
# GSD, which is what makes the resolution difference real rather than declared.
PAN_PX = 512
PAN_GSD_M = 0.65
SCENE_M = PAN_PX * PAN_GSD_M          # 333 m
SPAN_DEG = SCENE_M / 111_320.0


def _size_for(gsd_m: float) -> int:
    return max(8, int(round(SCENE_M / gsd_m)))


def _transform_for(n: int):
    return from_bounds(LON0, LAT0 - SPAN_DEG, LON0 + SPAN_DEG, LAT0, n, n)


def _urban_mask(n: int, seed: int):
    """A plausible land-cover layout: river, urban blocks, vegetation.

    Built on a normalised grid so one seed gives the same *relative* layout at
    every sensor's pixel count.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / max(n - 1, 1)

    river = np.abs((yy - 0.5) - 0.35 * np.sin(3.0 * xx)) < 0.10
    urban = np.zeros((n, n), bool)
    for cy, cx, r in ((0.30, 0.30, 0.14), (0.72, 0.66, 0.18)):
        urban |= (np.abs(yy - cy) < r) & (np.abs(xx - cx) < r * 1.3)
    urban &= ~river

    veg = ~(river | urban)
    return river, urban, veg, rng


def _texture(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Smooth correlated noise, so the scene is not flat white noise."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(rng.standard_normal((n, n)) * scale, sigma=3.0)


def make_pan(path: Path, seed: int = 11) -> dict:
    """Cartosat-2S PAN: ONE band, 10-bit DN, 0.65 m."""
    n = _size_for(PAN_GSD_M)
    river, urban, veg, rng = _urban_mask(n, seed)

    dn = np.zeros((n, n), np.float32)
    dn[veg] = 300.0                      # vegetation: mid-dark
    dn[urban] = 620.0                    # built-up: bright, high variance
    dn[river] = 150.0                    # water: dark
    dn = dn + _texture(n, rng, 90.0)
    # A deliberately bright patch: concrete yard / salt pan, WITH variance.
    # A perfectly uniform patch lands exactly on the 99th percentile and the
    # strict `>` in the PAN brightness proxy then flags nothing, hiding the
    # very failure this rehearsal is meant to surface.
    r0, r1 = int(0.08 * n), int(0.18 * n)
    c0, c1 = int(0.74 * n), int(0.92 * n)
    dn[r0:r1, c0:c1] = 900.0 + _texture(n, rng, 70.0)[r0:r1, c0:c1]
    dn = np.clip(dn, 0, 1023).astype("uint16")

    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="uint16", crs="EPSG:4326",
                       transform=_transform_for(n)) as dst:
        dst.write(dn, 1)
        dst.update_tags(ACQUISITION_DATE="2025-03-14")
    return {"file": path.name, "profile": "cartosat2s_pan", "bands": 1,
            "gsd_m": PAN_GSD_M, "dtype": "uint16", "pixels": n}


def make_mx(path: Path, seed: int = 11, date: str = "2025-03-14") -> dict:
    """Cartosat-2S MX: FOUR bands B/G/R/NIR, 10-bit DN, 1.6 m.

    No SWIR (correct for the sensor) -- downstream NDBI/MNDWI must fall back.
    """
    gsd = 1.6
    n = _size_for(gsd)
    river, urban, veg, rng = _urban_mask(n, seed)

    # reflectance-ish DN per class, B/G/R/NIR
    pal = {
        "veg":   np.array([180.0, 300.0, 210.0, 780.0]),
        "urban": np.array([430.0, 450.0, 470.0, 520.0]),
        "water": np.array([260.0, 300.0, 200.0, 110.0]),
    }
    stack = np.zeros((4, n, n), np.float32)
    for i in range(4):
        layer = np.zeros((n, n), np.float32)
        layer[veg] = pal["veg"][i]
        layer[urban] = pal["urban"][i]
        layer[river] = pal["water"][i]
        stack[i] = layer + _texture(n, rng, 45.0)
    # The bi-temporal pair must actually differ, or "no change detected" is
    # vacuous. Convert one vegetation block to built-up between the dates:
    # that is the crop/construction-edge case the change head exists for.
    if date == "2025-09-10":
        y0, y1 = int(0.55 * n), int(0.78 * n)
        x0, x1 = int(0.10 * n), int(0.36 * n)
        for i in range(4):
            stack[i, y0:y1, x0:x1] = pal["urban"][i]
    stack = np.clip(stack, 0, 1023).astype("uint16")

    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=4,
                       dtype="uint16", crs="EPSG:4326",
                       transform=_transform_for(n)) as dst:
        dst.write(stack)
        dst.update_tags(ACQUISITION_DATE=date)
    return {"file": path.name, "profile": "cartosat2s_mx", "bands": 4,
            "gsd_m": gsd, "dtype": "uint16", "pixels": n}


def make_risat(path: Path, seed: int = 23) -> dict:
    """RISAT: TWO bands HH/HV, linear sigma-nought power, 1.0 m.

    Distribution matched to the real Sentinel-1 GRD files in data/raw
    (p50 ~= 0.11, p99 ~= 2.7) so linear->dB produces a sane range.
    """
    gsd = 1.0
    n = _size_for(gsd)
    river, urban, veg, rng = _urban_mask(n, seed)

    hh = np.zeros((n, n), np.float32)
    hh[veg] = 0.20
    hh[urban] = 1.40                     # double-bounce: strong
    hh[river] = 0.03                     # specular: near-black
    # speckle is multiplicative in SAR, not additive
    hh *= np.exp(_texture(n, rng, 0.45))

    hv = hh * 0.28 * np.exp(_texture(n, rng, 0.30))
    stack = np.clip(np.stack([hh, hv]), 1e-4, None).astype("float32")

    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=2,
                       dtype="float32", crs="EPSG:4326",
                       transform=_transform_for(n)) as dst:
        dst.write(stack)
        dst.update_tags(ACQUISITION_DATE="2025-03-15")
    return {"file": path.name, "profile": "risat_sar", "bands": 2,
            "gsd_m": gsd, "dtype": "float32", "pixels": n}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    man = {"note": "SYNTHETIC rehearsal data - analogous, not ISRO-sourced",
           "site": {"lon": LON0, "lat": LAT0, "label": "Bengaluru outskirts"},
           "scene_extent_m": SCENE_M,
           "files": []}

    man["files"].append(make_pan(OUT / "cartosat2s_pan_2025-03-14.tif"))
    man["files"].append(make_mx(OUT / "cartosat2s_mx_2025-03-14.tif"))
    # a second MX date, for the bi-temporal change path
    man["files"].append(make_mx(OUT / "cartosat2s_mx_2025-09-10.tif",
                                seed=17, date="2025-09-10"))
    man["files"].append(make_risat(OUT / "risat_sar_2025-03-15.tif"))

    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"wrote {len(man['files'])} files to {OUT}")
    for f in man["files"]:
        print(f"  {f['file']:42s} {f['profile']:16s} {f['bands']} band  "
              f"{f['gsd_m']} m  {f['dtype']:8s} {f['pixels']}px")


if __name__ == "__main__":
    main()

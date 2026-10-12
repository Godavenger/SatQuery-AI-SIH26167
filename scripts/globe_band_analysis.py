"""Per-longitude analysis of the GIBS VIIRS base layer vs BlueMarble underpaint.

The globe composites: BlueMarble underpaint -> VIIRS daily layer (alpha-masked
no-data). If VIIRS is nearly opaque everywhere, the visible texture IS VIIRS and
any flatness is real imagery. If VIIRS is sparse/transparent in whole longitude
bands, the back side may be showing washed BlueMarble instead.
"""
import sys
import numpy as np
from PIL import Image

viirs_path, bm_path = sys.argv[1], sys.argv[2]

v = np.array(Image.open(viirs_path).convert("RGBA")).astype(float)
b = np.array(Image.open(bm_path).convert("RGB")).astype(float)
h, w = v.shape[:2]
print(f"viirs {v.shape}  bluemarble {b.shape}")

va = v[..., 3]  # alpha
print(f"VIIRS opaque fraction overall: {(va > 0).mean():.3f}")

# 16 longitude bands of 22.5 deg each
BANDS = 16
print("\nper-longitude-band stats (band = 2048/16 px wide):")
print(f"{'lon range':>16} {'alpha>0':>8} {'V-mean':>7} {'V-std':>7} {'BM-std':>7} {'detail':>7}")
for i in range(BANDS):
    x0, x1 = i * w // BANDS, (i + 1) * w // BANDS
    band_a = va[:, x0:x1]
    band_v = v[:, x0:x1, :3]
    band_b = b[:, x0:x1]
    # horizontal detail of the visible composite in this band (mid-latitudes)
    mid = band_v[256:768]
    mid_b = band_b[256:768]
    d = np.abs(np.diff(mid.mean(axis=2), axis=1)).mean() if (band_a > 0).any() else 0
    db = np.abs(np.diff(mid_b.mean(axis=2), axis=1)).mean()
    lon0 = -180 + i * 360 / BANDS
    lon1 = lon0 + 360 / BANDS
    print(
        f"{lon0:7.1f},{lon1:7.1f} {float((band_a > 0).mean()):8.3f}"
        f" {float(band_v[..., :3][band_a > 0].mean() if (band_a>0).any() else 0):7.1f}"
        f" {float(band_v[..., :3][band_a > 0].std() if (band_a>0).any() else 0):7.1f}"
        f" {float(db):7.2f} {float(d):7.2f}"
    )

# where alpha==0 (no VIIRS data), what does the composite show? = BlueMarble
nodata = va == 0
if nodata.any():
    print(f"\nno-data pixels: {nodata.mean():.3f} of image")
    print("BlueMarble stats in no-data regions:")
    print("  mean", b[nodata].mean(axis=0).round(1), "std", b[nodata].std(axis=0).round(1))
    # per-band no-data fraction
    for i in range(BANDS):
        x0, x1 = i * w // BANDS, (i + 1) * w // BANDS
        fr = nodata[:, x0:x1].mean()
        if fr > 0.05:
            print(f"  band lon {-180 + i * 22.5:6.1f}: no-data {fr:.3f}")

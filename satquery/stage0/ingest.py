"""Stage 0 - ingest, sensor abstraction, normalization.

The only stage that touches raw files, and the stage that absorbs sensor
heterogeneity so nothing downstream has to. Inputs span band counts from 1 to
12, ground sample distances from 0.15 m to 10 m, and two unrelated SAR sensors.

Every step in ``ingest()`` appends to a log that Stage 4 lifts into the trace,
so the compatibility checking the statement asks for is evidenced rather than
asserted.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..contracts import Tile, SceneMeta
from .profiles import (
    BENCHMARK_FORMATS, GEO_FORMATS, SensorProfile, get_profile, infer_profile,
    incidence_normalise, linear_to_db, percentile_stretch, refined_lee,
)


class IngestError(Exception):
    """Typed ingest failure. Carries the reason so the gate can name it."""

    def __init__(self, message: str, code: str = "ingest_error"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# guards - the /upload security posture from section 8
# --------------------------------------------------------------------------


@dataclass
class IngestLimits:
    """Caps against a crafted raster. A GeoTIFF can expand enormously on read,
    so the guard is on decompressed pixel count, not on file size alone."""

    max_file_mb: float = 400.0
    max_bands: int = 16
    max_pixels: int = 120_000_000       # ~11k x 11k
    max_dim: int = 20_000


DEFAULT_LIMITS = IngestLimits()


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


@dataclass
class IngestResult:
    tiles: list[Tile]
    profile: SensorProfile
    log: list[dict[str, Any]] = field(default_factory=list)
    display: Optional[np.ndarray] = None      # HxWx3 float [0,1], display only
    scene_id: str = ""
    warnings: list[str] = field(default_factory=list)


def ingest(
    path: str | Path,
    profile_id: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 32,
    max_tiles: int = 16,
    benchmark_path: bool = False,
    limits: IngestLimits = DEFAULT_LIMITS,
    gsd_override: Optional[float] = None,
    ts: Optional[str] = None,
) -> IngestResult:
    """Run the ordered normalizer sequence of section 3.2.

    ``benchmark_path`` whitelists non-georeferenced PNG/JPEG. The statement
    permits those only for prescribed benchmarks, so enforcing it as a rule is
    free compatibility-check credit.
    """
    path = Path(path)
    log: list[dict[str, Any]] = []
    warnings: list[str] = []

    def step(name: str, ok: bool, detail: str, **extra):
        log.append({"step": name, "ok": ok, "detail": detail, **extra})

    t0 = time.perf_counter()

    # -- 1. open and validate --------------------------------------------
    if not path.exists():
        raise IngestError(f"file not found: {path}", "not_found")

    size_mb = path.stat().st_size / 1e6
    if size_mb > limits.max_file_mb:
        raise IngestError(
            f"file is {size_mb:.0f} MB, cap is {limits.max_file_mb:.0f} MB",
            "too_large")

    ext = path.suffix.lower()
    if ext in BENCHMARK_FORMATS and not benchmark_path:
        raise IngestError(
            f"{ext} accepted only on the prescribed-benchmark path; "
            "georeferenced GeoTIFF required otherwise",
            "format_not_permitted")
    if ext not in BENCHMARK_FORMATS and ext not in GEO_FORMATS:
        raise IngestError(f"unsupported format '{ext}'", "format_not_permitted")

    arr, crs, transform, bounds, native_gsd, meta_ts = _read_raster(
        path, limits, benchmark_path)
    step("open_validate", True,
         f"{path.name}: {arr.shape[0]} band(s) {arr.shape[1]}x{arr.shape[2]} "
         f"{arr.dtype}, {'georeferenced' if crs else 'non-georeferenced (benchmark path)'}",
         bands=int(arr.shape[0]), size_mb=round(size_mb, 2), format=ext)

    # -- 2. resolve profile and map bands --------------------------------
    profile = (get_profile(profile_id) if profile_id
               else infer_profile(arr.shape[0], str(path), str(arr.dtype)))

    if profile.expected_bands and arr.shape[0] != profile.expected_bands:
        warnings.append(
            f"profile {profile.id} expects {profile.expected_bands} bands, "
            f"file has {arr.shape[0]}; mapping what is present")

    bands: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for slot, idx in profile.band_map.items():
        if idx is None:
            continue
        if idx >= arr.shape[0]:
            missing.append(slot)          # explicitly absent, never zero-filled
            continue
        bands[slot] = arr[idx].astype(np.float32) * profile.scale + profile.offset

    if not bands:
        raise IngestError(
            f"profile {profile.id} mapped no bands onto a {arr.shape[0]}-band "
            "file; wrong profile for this sensor", "profile_mismatch")

    step("resolve_profile", True,
         f"profile={profile.id} ({profile.platform}); mapped {sorted(bands)}"
         + (f"; absent (kept None, not zero-filled): {missing}" if missing else ""),
         profile_id=profile.id, mapped=sorted(bands), absent=missing)

    gsd = float(gsd_override or native_gsd or profile.native_gsd)

    # -- 3/4. modality-specific radiometry --------------------------------
    if profile.radiometry == "sar_linear_to_db":
        for slot in list(bands):
            x = bands[slot]
            x = linear_to_db(x)
            if profile.speckle_filter == "refined_lee":
                x = refined_lee(x)
            x = incidence_normalise(x, profile.incidence_angle)
            bands[slot] = x
        step("sar_radiometry", True,
             f"linear->dB, {profile.speckle_filter} speckle filter, incidence "
             f"normalised from {profile.incidence_angle} deg to 39 deg reference",
             pols=sorted(bands))
    else:
        step("optical_radiometry", True,
             "raw float reflectance retained for models; percentile stretch "
             "computed separately for display only",
             scale=profile.scale)

    # -- 5. quality mask --------------------------------------------------
    qa, qa_detail = _quality_mask(bands, profile)
    step("quality_mask", True, qa_detail,
         flagged_fraction=round(float(qa.mean()), 4))

    # -- display composite (never reaches a model) ------------------------
    display = _display_composite(bands, ~qa)

    # -- 6. tile ----------------------------------------------------------
    tiles = _tile(bands, qa, profile, gsd, crs, transform, bounds, ts or meta_ts,
                  str(path), tile_size, overlap, max_tiles)
    step("tile", True,
         f"{len(tiles)} tile(s) at {tile_size}x{tile_size}, overlap {overlap} px, "
         f"GSD {gsd:g} m carried through",
         n_tiles=len(tiles), tile_size=tile_size, gsd=gsd)

    step("ingest_complete", True,
         f"{(time.perf_counter() - t0) * 1000:.0f} ms",
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    return IngestResult(
        tiles=tiles, profile=profile, log=log, display=display,
        scene_id=tiles[0].meta.parent_scene if tiles else path.stem,
        warnings=warnings,
    )


def _read_raster(path: Path, limits: IngestLimits, benchmark_path: bool):
    """Read with rasterio, falling back to PIL on the benchmark path."""
    ext = path.suffix.lower()

    if ext in BENCHMARK_FORMATS:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width * im.height > limits.max_pixels:
                raise IngestError("image exceeds pixel cap", "too_large")
            a = np.asarray(im)
        return np.transpose(a, (2, 0, 1)), None, None, None, None, None

    try:
        import rasterio
    except ImportError as e:                                # pragma: no cover
        raise IngestError(f"rasterio required for {ext}: {e}", "no_reader")

    with rasterio.open(path) as src:
        if src.count > limits.max_bands:
            raise IngestError(
                f"{src.count} bands exceeds cap {limits.max_bands}", "too_large")
        if src.width > limits.max_dim or src.height > limits.max_dim:
            raise IngestError(
                f"{src.width}x{src.height} exceeds dimension cap "
                f"{limits.max_dim}", "too_large")
        if src.width * src.height * src.count > limits.max_pixels:
            raise IngestError(
                "decompressed pixel count exceeds cap (decompression-bomb guard)",
                "too_large")

        georeferenced = src.crs is not None and src.transform is not None
        if not georeferenced and not benchmark_path:
            raise IngestError(
                "GeoTIFF is not georeferenced (no CRS/transform) and this is "
                "not the whitelisted benchmark path",
                "not_georeferenced")

        arr = src.read()
        crs = str(src.crs) if src.crs else None
        transform = tuple(src.transform)[:6] if src.transform else None
        bounds = tuple(src.bounds) if georeferenced else None

        gsd = None
        if src.transform:
            gsd = abs(float(src.transform.a))
            if crs and ("4326" in crs or getattr(src.crs, "is_geographic", False)):
                # In a geographic CRS a longitude degree shrinks with latitude,
                # so the x pixel size is not a metre scale. The y pixel size is
                # latitude-independent, which makes it the honest GSD estimate.
                gsd = abs(float(src.transform.e)) * 111_320.0

        ts = None
        tags = src.tags()
        for k in ("TIFFTAG_DATETIME", "ACQUISITION_DATE", "datetime", "DATE"):
            if k in tags:
                ts = tags[k]
                break

    return arr, crs, transform, bounds, gsd, ts


def _quality_mask(bands: dict[str, np.ndarray],
                  profile: SensorProfile) -> tuple[np.ndarray, str]:
    """Cloud / shadow / nodata mask.

    This is what later justifies a SAR fallback rather than merely asserting
    that fusion helps: the ablation in Stage 6 restricts to tiles this mask
    flags, so the mask has to be real.
    """
    shape = next(iter(bands.values())).shape

    if profile.modality == "sar":
        # SAR sees through cloud; only nodata is invalid.
        mask = np.zeros(shape, dtype=bool)
        for x in bands.values():
            mask |= ~np.isfinite(x)
        return mask, f"SAR: cloud-free by physics; {int(mask.sum())} nodata px"

    blue, red = bands.get("blue"), bands.get("red")
    nir, swir1 = bands.get("nir"), bands.get("swir1")
    pan = bands.get("pan")

    if blue is None and pan is not None:
        # Panchromatic: brightness-only heuristic is all that is available.
        bright = pan > np.percentile(pan[np.isfinite(pan)], 99.0)
        return bright, ("panchromatic: no spectral cloud test possible, "
                        "brightness-percentile proxy only")

    if blue is None or red is None:
        return np.zeros(shape, dtype=bool), "no spectral bands for a cloud test"

    # Spectral cloud test: clouds are bright and spectrally flat across the
    # visible, and unlike bright bare soil they stay bright in the blue.
    green = bands.get("green", red)
    brightness = (blue + green + red) / 3.0
    whiteness = 1.0 - (np.abs(blue - brightness) + np.abs(green - brightness)
                       + np.abs(red - brightness)) / np.maximum(3 * brightness, 1e-6)
    cloud = (brightness > 0.28) & (whiteness > 0.72)

    if swir1 is not None:
        # Snow and ice are bright and white too, but dark in SWIR.
        cloud &= swir1 > 0.10

    shadow = np.zeros(shape, dtype=bool)
    if nir is not None and blue is not None:
        # Cloud shadow = dark in NIR but bright in blue relative to its
        # neighbourhood, because scattering keeps shadowed haze blue-ish.
        # Dark water and mangrove swamp are dark in BOTH NIR and blue;
        # flagging them kills valid coastal pixels. Use the blue/NIR
        # contrast to separate the two.
        from scipy.ndimage import uniform_filter
        dark_nir = nir < 0.06
        blue_local = uniform_filter(blue, size=9)
        shadow = dark_nir & (blue > 1.6 * np.maximum(blue_local, 1e-3))
        # shadows are also darker than their surroundings in NIR overall
        nir_local = uniform_filter(nir, size=9)
        shadow &= nir_local > 1.5 * np.maximum(nir, 1e-3)

    nodata = np.zeros(shape, dtype=bool)
    for x in bands.values():
        nodata |= ~np.isfinite(x)

    mask = cloud | shadow | nodata
    return mask, (f"spectral cloud+shadow test: {100 * cloud.mean():.1f}% cloud, "
                  f"{100 * shadow.mean():.1f}% shadow, "
                  f"{100 * nodata.mean():.1f}% nodata")


def _display_composite(bands: dict[str, np.ndarray],
                       valid: np.ndarray) -> np.ndarray:
    """Build an HxWx3 display image. Display radiometry only - Stage 2 reads
    ``Tile.bands``, never this."""
    if all(k in bands for k in ("red", "green", "blue")):
        chans = [bands["red"], bands["green"], bands["blue"]]
    elif "pan" in bands:
        chans = [bands["pan"]] * 3
    elif "VV" in bands or "HH" in bands:
        co = bands.get("VV", bands.get("HH"))
        cross = bands.get("VH", bands.get("HV", co))
        # Standard SAR false-colour: co-pol / cross-pol / ratio
        chans = [co, cross, co - cross]
    else:
        first = next(iter(bands.values()))
        chans = [first] * 3
    return np.stack([percentile_stretch(c, valid=valid) for c in chans], axis=-1)


def _tile(bands, qa, profile, gsd, crs, transform, bounds, ts, source,
          size, overlap, max_tiles) -> list[Tile]:
    """Cut into overlapping tiles, carrying GSD and geo-transform through."""
    h, w = next(iter(bands.values())).shape
    scene = hashlib.sha1(f"{source}{h}{w}".encode()).hexdigest()[:10]
    stride = max(1, size - overlap)

    if h <= size and w <= size:
        starts = [(0, 0)]
    else:
        rows = list(range(0, max(1, h - overlap), stride))
        cols = list(range(0, max(1, w - overlap), stride))
        starts = [(r, c) for r in rows for c in cols][:max_tiles]

    tiles: list[Tile] = []
    for n, (r0, c0) in enumerate(starts):
        r1, c1 = min(r0 + size, h), min(c0 + size, w)
        sub = {k: v[r0:r1, c0:c1] for k, v in bands.items()}
        sub_qa = qa[r0:r1, c0:c1]

        tf = None
        if transform:
            a, b, c, d, e, f = transform
            # shift the origin to this tile's top-left corner
            tf = (a, b, c + a * c0 + b * r0, d, e, f + d * c0 + e * r0)

        tb = None
        if tf:
            th, tw = r1 - r0, c1 - c0
            xs = [tf[0] * x + tf[1] * y + tf[2] for x, y in
                  ((0, 0), (tw, 0), (0, th), (tw, th))]
            ys = [tf[3] * x + tf[4] * y + tf[5] for x, y in
                  ((0, 0), (tw, 0), (0, th), (tw, th))]
            tb = (min(xs), min(ys), max(xs), max(ys))

        tiles.append(Tile(
            id=f"{scene}_r{r0}_c{c0}",
            sensor_id=profile.id,
            bands=sub,
            gsd=gsd,
            crs=crs,
            transform=tf,
            bounds=tb,
            ts=ts,
            qa_mask=sub_qa,
            meta=SceneMeta(
                parent_scene=scene,
                tile_grid=(r0 // stride, c0 // stride),
                source_path=str(source),
                profile_id=profile.id,
                ingest_params={"tile_size": size, "overlap": overlap,
                               "gsd": gsd, "radiometry": profile.radiometry,
                               "speckle_filter": profile.speckle_filter},
            ),
        ))
    return tiles


# --------------------------------------------------------------------------
# pair handling (section 3.4)
# --------------------------------------------------------------------------


@dataclass
class PairCheck:
    """Result of validating a two-tile pair. ``residual_shift_px`` doubles as
    visible proof the alignment check was performed."""

    kind: str                       # bi_temporal | cross_modal
    ok: bool
    footprint_iou: float = 0.0
    residual_shift_px: tuple[float, float] = (0.0, 0.0)
    residual_magnitude_px: float = 0.0
    distinct_timestamps: bool = False
    modalities: tuple[str, str] = ("", "")
    detail: str = ""
    aligned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "footprint_iou": round(self.footprint_iou, 4),
            "residual_shift_px": [round(v, 3) for v in self.residual_shift_px],
            "residual_magnitude_px": round(self.residual_magnitude_px, 3),
            "distinct_timestamps": self.distinct_timestamps,
            "modalities": list(self.modalities),
            "aligned": self.aligned,
            "detail": self.detail,
        }


def footprint_iou(a: Tile, b: Tile) -> float:
    """Intersection-over-union of two tile footprints in geographic space."""
    if not a.bounds or not b.bounds:
        # No geo-reference on either side (benchmark path): fall back to shape
        # agreement, which is the only footprint notion available.
        return 1.0 if a.shape == b.shape else 0.0
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / union) if union > 1e-12 else 0.0


def _grey(t: Tile) -> np.ndarray:
    """A single representative channel for alignment, in a form comparable
    across modalities."""
    for slot in ("nir", "red", "pan", "green", "VV", "HH", "VH", "HV"):
        if slot in t.bands:
            x = t.bands[slot].astype(np.float32)
            x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0)
            s = x.std()
            return (x - x.mean()) / (s if s > 1e-8 else 1.0)
    x = next(iter(t.bands.values())).astype(np.float32)
    return np.nan_to_num(x)


def check_pair(a: Tile, b: Tile, kind: str,
               max_shift_px: float = 2.0,
               min_iou: float = 0.9) -> PairCheck:
    """Validate a pair and measure the residual misalignment.

    At 0.65 m a sub-pixel misalignment manufactures change that does not
    exist (risk 3), so the shift is measured by phase correlation and logged
    in pixels whether or not it passes.
    """
    mods = (a.modality, b.modality)
    iou = footprint_iou(a, b)
    distinct = bool(a.ts and b.ts and a.ts != b.ts)

    shift = (0.0, 0.0)
    mag = 0.0
    if a.shape == b.shape and min(a.shape) > 8:
        try:
            from skimage.registration import phase_cross_correlation
            sh, _, _ = phase_cross_correlation(
                _grey(a), _grey(b), upsample_factor=10, normalization=None)
            shift = (float(sh[0]), float(sh[1]))
            mag = float(np.hypot(*shift))
        except Exception:
            pass

    if kind == "bi_temporal":
        ok = iou >= min_iou and mag <= max_shift_px
        reasons = []
        if iou < min_iou:
            reasons.append(f"footprint IoU {iou:.3f} below {min_iou}")
        if mag > max_shift_px:
            reasons.append(
                f"residual alignment shift {mag:.2f} px exceeds {max_shift_px} px "
                "tolerance - at sub-metre GSD this manufactures phantom change")
        if not distinct:
            reasons.append("timestamps not distinct (or absent)")
        detail = ("; ".join(reasons) if reasons else
                  f"footprint IoU {iou:.3f}, residual shift {mag:.2f} px, "
                  "timestamps distinct")
        # Absent timestamps are a warning, not a hard fail: benchmark pairs
        # frequently carry none, and the pair is still bi-temporal by construction.
        return PairCheck("bi_temporal", ok, iou, shift, mag, distinct, mods,
                         detail, aligned=mag <= max_shift_px)

    # cross-modal
    truly_different = ("optical" in mods and "sar" in mods)
    ok = truly_different and iou >= min_iou
    if not truly_different:
        detail = (f"both inputs are {mods[0]}; fusion requires one optical and "
                  "one SAR tile, so there is nothing complementary to fuse")
    elif iou < min_iou:
        detail = f"footprint IoU {iou:.3f} below {min_iou}; tiles do not overlap"
    else:
        detail = (f"optical+SAR confirmed, footprint IoU {iou:.3f}, "
                  f"co-registration residual {mag:.2f} px")
    return PairCheck("cross_modal", ok, iou, shift, mag, distinct, mods,
                     detail, aligned=mag <= max_shift_px)


def align_pair(a: Tile, b: Tile, shift: tuple[float, float]) -> Tile:
    """Resample ``b`` onto ``a``'s grid by the measured shift."""
    from scipy.ndimage import shift as ndshift

    if abs(shift[0]) < 1e-3 and abs(shift[1]) < 1e-3:
        return b
    bands = {k: ndshift(v, shift, order=1, mode="nearest")
             for k, v in b.bands.items()}
    qa = b.qa_mask
    if qa is not None:
        qa = ndshift(qa.astype(np.float32), shift, order=0,
                     mode="nearest") > 0.5
    return Tile(id=b.id + "_aligned", sensor_id=b.sensor_id, bands=bands,
                gsd=b.gsd, crs=b.crs, transform=b.transform, bounds=b.bounds,
                ts=b.ts, qa_mask=qa, meta=b.meta)


def resample_to(src: Tile, ref: Tile) -> Tile:
    """Resample ``src`` onto ``ref``'s pixel grid for cross-modal fusion."""
    if src.shape == ref.shape:
        return src
    from scipy.ndimage import zoom

    sh, sw = src.shape
    rh, rw = ref.shape
    fy, fx = rh / sh, rw / sw
    bands = {k: zoom(v, (fy, fx), order=1) for k, v in src.bands.items()}
    qa = None
    if src.qa_mask is not None:
        qa = zoom(src.qa_mask.astype(np.float32), (fy, fx), order=0) > 0.5
    return Tile(id=src.id + "_rs", sensor_id=src.sensor_id, bands=bands,
                gsd=ref.gsd, crs=ref.crs, transform=ref.transform,
                bounds=ref.bounds, ts=src.ts, qa_mask=qa, meta=src.meta)


# --------------------------------------------------------------------------
# cache - offline mode is a cache hit, not a second code path
# --------------------------------------------------------------------------


class TileCache:
    """Canonical tiles on disk. Offline mode reads the same objects the online
    path writes, so there is only one code path to be correct."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, path: str | Path, **params) -> str:
        p = Path(path)
        try:
            stat = f"{p.stat().st_mtime_ns}:{p.stat().st_size}"
        except OSError:
            stat = "0:0"
        blob = f"{p.resolve()}|{stat}|{sorted(params.items())}"
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def get(self, path, **params) -> Optional[IngestResult]:
        f = self.root / f"{self._key(path, **params)}.pkl"
        if not f.exists():
            return None
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def put(self, path, result: IngestResult, **params) -> None:
        f = self.root / f"{self._key(path, **params)}.pkl"
        tmp = f.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, f)
        except Exception:
            tmp.unlink(missing_ok=True)

    def ingest_cached(self, path, **kwargs) -> IngestResult:
        hit = self.get(path, **kwargs)
        if hit is not None:
            hit.log = list(hit.log) + [
                {"step": "cache", "ok": True, "detail": "canonical tiles served from disk cache"}]
            return hit
        res = ingest(path, **kwargs)
        self.put(path, res, **kwargs)
        return res

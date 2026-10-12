"""Sensor profile registry (section 3.1).

Adding RISAT is one profile entry and a calibration check, not a code change.
Nothing downstream of Stage 0 branches on dataset or platform — that is the
whole point of this file.

The band maps below are the load-bearing detail. Cartosat-2S PAN is a single
band and MX is four; neither is RGB and neither is Sentinel-2's twelve. Any
model that assumes three or twelve channels breaks on evaluation day (risk 2),
so the mapping from physical index to semantic slot lives here and absent slots
stay ``None`` rather than being zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np


@dataclass
class SensorProfile:
    """Everything Stage 0 needs to turn a raw raster into a canonical Tile."""

    id: str
    platform: str
    native_gsd: float                       # metres
    band_map: dict[str, Optional[int]]      # semantic slot -> physical index
    dtype: str = "uint16"
    scale: float = 1.0
    offset: float = 0.0
    nodata: Optional[float] = 0.0
    radiometry: str = "optical_percentile_stretch"   # or sar_linear_to_db
    speckle_filter: str = "none"            # or refined_lee
    incidence_angle: Optional[float] = None  # degrees; SAR normalisation
    calibration_note: str = ""
    expected_bands: Optional[int] = None
    modality: str = "optical"

    def present_slots(self) -> list[str]:
        """Semantic slots this sensor actually carries."""
        return [k for k, v in self.band_map.items() if v is not None]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["present_slots"] = self.present_slots()
        return d


# --------------------------------------------------------------------------
# shipped profiles
# --------------------------------------------------------------------------
#
# BigEarthNet stacks Sentinel-2 as 12 bands in the order
#   B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
# (B10 cirrus is dropped), which is where these indices come from.

S2_MSI = SensorProfile(
    id="s2_msi",
    platform="Sentinel-2 MSI",
    native_gsd=10.0,
    band_map={
        "blue": 1, "green": 2, "red": 3, "nir": 7,
        "swir1": 10, "swir2": 11, "pan": None,
        "VV": None, "VH": None, "HH": None, "HV": None,
    },
    dtype="uint16",
    scale=1.0 / 10000.0,          # L2A reflectance scaling
    offset=0.0,
    nodata=0.0,
    radiometry="optical_percentile_stretch",
    expected_bands=12,
    modality="optical",
    calibration_note=(
        "BigEarthNet 12-band stack, B10 dropped. Scale 1/10000 to surface "
        "reflectance. Mandated adaptation corpus at 10 m GSD."
    ),
)

S1_GRD = SensorProfile(
    id="s1_grd",
    platform="Sentinel-1 GRD",
    native_gsd=10.0,
    band_map={
        "VV": 0, "VH": 1, "HH": None, "HV": None,
        "blue": None, "green": None, "red": None,
        "nir": None, "swir1": None, "swir2": None, "pan": None,
    },
    dtype="float32",
    scale=1.0,
    nodata=0.0,
    radiometry="sar_linear_to_db",
    speckle_filter="refined_lee",
    incidence_angle=39.0,          # S1 IW mid-swath nominal
    expected_bands=2,
    modality="sar",
    calibration_note=(
        "Sigma-nought linear power, dual-pol VV/VH. Converted to dB and "
        "refined-Lee filtered. Incidence-angle normalised to 39 deg."
    ),
)

CARTOSAT2S_PAN = SensorProfile(
    id="cartosat2s_pan",
    platform="Cartosat-2S panchromatic",
    native_gsd=0.65,
    band_map={
        "pan": 0,
        "blue": None, "green": None, "red": None, "nir": None,
        "swir1": None, "swir2": None,
        "VV": None, "VH": None, "HH": None, "HV": None,
    },
    dtype="uint16",
    scale=1.0 / 1023.0,            # 10-bit DN
    nodata=0.0,
    radiometry="optical_percentile_stretch",
    expected_bands=1,
    modality="optical",
    calibration_note=(
        "SINGLE BAND. Not RGB. Any model assuming three channels fails here "
        "(risk 2). Sub-metre GSD, 15-30x finer than the adaptation corpus."
    ),
)

CARTOSAT2S_MX = SensorProfile(
    id="cartosat2s_mx",
    platform="Cartosat-2S multispectral",
    native_gsd=1.6,
    band_map={
        "blue": 0, "green": 1, "red": 2, "nir": 3,
        "swir1": None, "swir2": None, "pan": None,
        "VV": None, "VH": None, "HH": None, "HV": None,
    },
    dtype="uint16",
    scale=1.0 / 1023.0,
    nodata=0.0,
    radiometry="optical_percentile_stretch",
    expected_bands=4,
    modality="optical",
    calibration_note=(
        "FOUR bands B/G/R/NIR. No SWIR, so any index needing SWIR (NDBI, "
        "MNDWI) must fall back to a NIR-only formulation on this sensor."
    ),
)

RISAT_SAR = SensorProfile(
    id="risat_sar",
    platform="RISAT C-band SAR",
    native_gsd=1.0,
    band_map={
        "HH": 0, "HV": 1, "VV": None, "VH": None,
        "blue": None, "green": None, "red": None,
        "nir": None, "swir1": None, "swir2": None, "pan": None,
    },
    dtype="float32",
    scale=1.0,
    nodata=0.0,
    radiometry="sar_linear_to_db",
    speckle_filter="refined_lee",
    incidence_angle=35.0,          # differs from S1 — normalisation must know
    expected_bands=2,
    modality="sar",
    calibration_note=(
        "C-band like Sentinel-1, which helps transfer, but polarisation "
        "differs (HH/HV vs VV/VH) and the incidence angle convention differs. "
        "The SAR branch must tolerate a missing second polarisation."
    ),
)

GENERIC_RGB = SensorProfile(
    id="generic_rgb",
    platform="generic RGB (benchmark path)",
    native_gsd=0.5,
    band_map={
        "red": 0, "green": 1, "blue": 2,
        "nir": None, "swir1": None, "swir2": None, "pan": None,
        "VV": None, "VH": None, "HH": None, "HV": None,
    },
    dtype="uint8",
    scale=1.0 / 255.0,
    nodata=None,
    radiometry="optical_percentile_stretch",
    expected_bands=3,
    modality="optical",
    calibration_note=(
        "PNG/JPEG benchmark path only (VRSBench, RSVQA-HR, CDVQA). The "
        "statement permits non-georeferenced input only for prescribed "
        "benchmarks, so this profile is gated to the whitelisted path."
    ),
)

RSVQA_LR = SensorProfile(
    id="rsvqa_lr",
    platform="RSVQA-LR (Sentinel-2 RGB crop)",
    native_gsd=10.0,
    band_map={
        "red": 0, "green": 1, "blue": 2,
        "nir": None, "swir1": None, "swir2": None, "pan": None,
        "VV": None, "VH": None, "HH": None, "HV": None,
    },
    dtype="uint8",
    scale=1.0 / 255.0,
    nodata=None,
    radiometry="optical_percentile_stretch",
    expected_bands=3,
    modality="optical",
    calibration_note="RSVQA low-resolution split, Sentinel-2 derived RGB at 10 m.",
)


PROFILES: dict[str, SensorProfile] = {
    p.id: p for p in (
        S2_MSI, S1_GRD, CARTOSAT2S_PAN, CARTOSAT2S_MX,
        RISAT_SAR, GENERIC_RGB, RSVQA_LR,
    )
}

#: Formats permitted without georeferencing, and only on the benchmark path.
BENCHMARK_FORMATS = frozenset({".png", ".jpg", ".jpeg"})
GEO_FORMATS = frozenset({".tif", ".tiff", ".geotiff", ".jp2"})


def get_profile(profile_id: str) -> SensorProfile:
    if profile_id not in PROFILES:
        raise UnknownSensorError(
            f"no sensor profile '{profile_id}'; known: {sorted(PROFILES)}"
        )
    return PROFILES[profile_id]


def register_profile(profile: SensorProfile) -> SensorProfile:
    """Adding a sensor is an entry here, not a code change downstream."""
    PROFILES[profile.id] = profile
    return profile


class UnknownSensorError(KeyError):
    """Raised rather than guessing a profile — guessing is how a Cartosat PAN
    scene silently becomes a very dark RGB."""


def infer_profile(n_bands: int, path: str = "", dtype: str = "",
                  hint: Optional[str] = None) -> SensorProfile:
    """Best-effort profile resolution for user uploads.

    Band count alone is ambiguous (4 could be Cartosat MX or RGBA), so the
    filename is consulted too. The resolved profile id is written to the trace
    so a wrong inference is visible rather than silent.
    """
    if hint:
        return get_profile(hint)

    name = str(path).lower()
    for token, pid in (
        ("cartosat", "cartosat2s_mx"), ("risat", "risat_sar"),
        ("s1", "s1_grd"), ("sentinel-1", "s1_grd"), ("sentinel1", "s1_grd"),
        ("s2", "s2_msi"), ("sentinel-2", "s2_msi"), ("sentinel2", "s2_msi"),
        ("bigearthnet", "s2_msi"), ("ben_", "s2_msi"),
    ):
        if token in name:
            prof = PROFILES[pid]
            # a filename saying "cartosat" with one band is PAN, not MX
            if pid == "cartosat2s_mx" and n_bands == 1:
                return CARTOSAT2S_PAN
            if prof.expected_bands in (None, n_bands) or n_bands >= 2:
                return prof

    if n_bands == 1:
        return CARTOSAT2S_PAN
    if n_bands == 2:
        return S1_GRD
    if n_bands in (3, 4) and dtype in ("uint8", "int8"):
        return GENERIC_RGB
    if n_bands == 4:
        return CARTOSAT2S_MX
    if n_bands >= 12:
        return S2_MSI
    return GENERIC_RGB


# --------------------------------------------------------------------------
# radiometry — the operations the profile names
# --------------------------------------------------------------------------


def linear_to_db(x: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """SAR sigma-nought linear power to decibels."""
    return 10.0 * np.log10(np.maximum(x.astype(np.float32), floor))


def incidence_normalise(db: np.ndarray, angle: float,
                        reference: float = 39.0) -> np.ndarray:
    """Cosine-law incidence-angle normalisation to a reference angle.

    Kept explicit because S1 and RISAT differ here; leaving it implicit is how
    a RISAT scene ends up systematically brighter than the training corpus.
    """
    if angle is None:
        return db
    a = np.deg2rad(float(angle))
    r = np.deg2rad(float(reference))
    cos_a, cos_r = np.cos(a), np.cos(r)
    if cos_a <= 1e-6:
        return db
    return db + 10.0 * np.log10(cos_r / cos_a)


def refined_lee(img: np.ndarray, size: int = 7, cu: float = 0.523) -> np.ndarray:
    """Refined-Lee speckle filter.

    Adaptive local-statistics filter: in homogeneous areas it averages hard, in
    heterogeneous areas it preserves the pixel. ``cu`` is the noise variation
    coefficient for single-look intensity.
    """
    from scipy.ndimage import uniform_filter

    x = img.astype(np.float32)
    mean = uniform_filter(x, size=size)
    sq = uniform_filter(x * x, size=size)
    var = np.maximum(sq - mean * mean, 0.0)

    denom = np.maximum(mean * mean, 1e-8)
    ci2 = var / denom
    cu2 = cu * cu
    # weight -> 0 in flat areas (take the mean), -> 1 at edges (keep the pixel)
    w = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-8), 0.0, 1.0)
    return mean + w * (x - mean)


def percentile_stretch(x: np.ndarray, lo: float = 2.0, hi: float = 98.0,
                       valid: Optional[np.ndarray] = None) -> np.ndarray:
    """2-98 percentile stretch to [0,1]. DISPLAY ONLY.

    Display radiometry must never reach a model (section 3.2): a per-scene
    stretch destroys the absolute reflectance relationships that spectral
    indices depend on.
    """
    a = x.astype(np.float32)
    sample = a[valid] if valid is not None and valid.any() else a
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros_like(a)
    p1, p2 = np.percentile(sample, [lo, hi])
    if p2 - p1 < 1e-8:
        p2 = p1 + 1e-8
    return np.clip((a - p1) / (p2 - p1), 0.0, 1.0)

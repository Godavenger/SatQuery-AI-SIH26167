"""Deterministic regressions using generated rasters, never private weights/data."""
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.contracts import Answer, Slots, TaskResult, Tile, TraceRecord, _jsonable
from satquery.pipeline import run_query
from satquery.stage0.ingest import (
    IngestError, align_pair, check_pair, footprint_iou, ingest, resample_to,
)
from satquery.stage1.controller import Controller
from satquery.stage1.gate import run_gate
from satquery.stage1.router import RuleRouter, extract_slots
from satquery.stage2.heads import ChangeHead, FuseHead, VQAHead
from satquery.stage3.integrate import calibrate, merge
from satquery.stage4.report import build_pdf


@pytest.fixture
def spectral_only(monkeypatch):
    from satquery import pipeline
    from satquery.stage2 import heads
    monkeypatch.setattr(pipeline, "_controller", Controller(RuleRouter()))
    for name in ("_neural_presence", "_neural_lulc", "_neural_answer",
                 "_neural_sar_lulc", "_neural_sar_presence"):
        monkeypatch.setattr(heads, name, lambda *args, **kwargs: None)


def write_raster(path, pixels, *, nodata=None, transform=None, crs="EPSG:32643"):
    with rasterio.open(path, "w", driver="GTiff", height=pixels.shape[1],
                       width=pixels.shape[2], count=pixels.shape[0],
                       dtype=pixels.dtype, crs=crs,
                       transform=transform or from_origin(500000, 2800000, 10, 10),
                       nodata=nodata) as ds:
        ds.write(pixels)
    return path


def tile(bands=None, **kwargs):
    defaults = dict(id="tile", sensor_id="s2_msi", gsd=10,
                    bands=bands or {"nir": np.ones((32, 32), np.float32) * .2,
                                    "green": np.ones((32, 32), np.float32) * .4,
                                    "red": np.ones((32, 32), np.float32) * .1})
    defaults.update(kwargs)
    return Tile(**defaults)


def test_pipeline_import_without_torch():
    proc = subprocess.run([sys.executable, "-c",
                           "import sys; sys.modules['torch'] = None; "
                           "import satquery.pipeline; import satquery.stage2.heads"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_json_boolean_values_remain_booleans():
    value = _jsonable({"ok": True, "missing": np.bool_(False), "count": np.int64(2)})
    assert value["ok"] is True
    assert value["missing"] is False
    assert json.loads(json.dumps(value))["ok"] is True


def test_hash_identifies_geometry_and_quality():
    t = tile()
    assert t.content_hash() != replace(t, qa_mask=np.ones(t.shape, bool)).content_hash()
    assert t.content_hash() != replace(t, ts="2026-10-25").content_hash()


def test_projected_grounding_returns_longitude_latitude():
    t = tile(crs="EPSG:32643", transform=tuple(from_origin(500000, 2800000, 10, 10))[:6])
    lon, lat = t.pixel_to_geo(0, 0)
    assert lon == pytest.approx(75, abs=.01)
    assert 25 < lat < 26
    box = t.box_to_geo(0, 0, 20, 10)
    assert box["west"] < box["east"] < 76
    assert 25 < box["south"] < box["north"] < 26


def test_rotated_bounds_include_all_corners():
    t = tile(transform=(1, -1, 0, 1, 1, 0))
    assert t.box_to_geo(0, 0, 2, 3) == dict(west=-3, east=2, south=0, north=5)


def test_source_nodata_is_excluded(tmp_path):
    values = np.full((12, 32, 32), 1800, dtype=np.uint16)
    values[:, :8] = 65535
    path = write_raster(tmp_path / "scene.tif", values, nodata=65535)
    t = ingest(path, profile_id="s2_msi", tile_size=32, overlap=0).tiles[0]
    assert t.qa_mask[:8].all()
    assert t.clear_fraction == .75


def test_sar_zero_power_stays_invalid_after_db_conversion(tmp_path):
    values = np.full((2, 32, 32), .03, dtype=np.float32)
    values[:, :8] = 0
    path = write_raster(tmp_path / "radar.tif", values)
    t = ingest(path, profile_id="s1_grd", tile_size=32, overlap=0).tiles[0]
    assert t.qa_mask[:8].all()


def test_all_nodata_pan_does_not_crash(tmp_path):
    path = write_raster(tmp_path / "pan.tif", np.full((1, 32, 32), np.nan, np.float32))
    t = ingest(path, profile_id="cartosat2s_pan", tile_size=32, overlap=0).tiles[0]
    assert t.clear_fraction == 0


@pytest.mark.parametrize("kwargs", [dict(tile_size=0), dict(overlap=-1),
                                     dict(max_tiles=0), dict(gsd_override=float("nan"))])
def test_invalid_ingest_parameters_fail_before_read(tmp_path, kwargs):
    with pytest.raises(IngestError, match="must be"):
        ingest(tmp_path / "missing.tif", **kwargs)


@pytest.mark.parametrize("options", [dict(profile_ids=[]), dict(timestamps=[None, None])])
def test_per_tile_metadata_cannot_silently_drop_inputs(options):
    with pytest.raises(ValueError, match="one entry per tile"):
        run_query("Describe this scene", ["unused.tif"], **options)


def test_spectral_pipeline_runs_from_generated_geotiff(tmp_path, spectral_only):
    values = np.full((12, 64, 64), 1000, np.uint16)
    values[7] = 4000
    path = write_raster(tmp_path / "sample.tif", values)
    trace = run_query("Describe this scene", [path], profile_ids=["s2_msi"], use_cache=False)
    assert trace.answer and not trace.answer.abstained
    assert "vegetation" in trace.answer.text
    assert trace.answer.contributing == ["satquery-caption"]
    assert {s.stage for s in trace.steps} == {0, 1, 2, 3}
    assert json.loads(trace.to_json())["answer"]["abstained"] is False


def test_explicit_benchmark_path_supports_png(tmp_path, spectral_only):
    from PIL import Image
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), (50, 80, 30)).save(path)
    with pytest.raises(IngestError, match="benchmark"):
        run_query("Describe this scene", [path], use_cache=False)
    trace = run_query("Describe this scene", [path], benchmark_path=True, use_cache=False)
    assert trace.answer is not None


def test_cloud_only_input_abstains():
    verdict, _ = run_gate("vqa", [tile(qa_mask=np.ones((32, 32), bool))])
    assert verdict.action == "abstain"
    assert "no usable pixels" in verdict.reason


def test_unsupported_discrete_object_question_abstains():
    verdict, _ = run_gate("vqa", [tile()], "Are there any ships here?")
    assert verdict.action == "abstain"


def test_supported_context_does_not_hide_unsupported_object():
    verdict, _ = run_gate("vqa", [tile()], "Are there ships on this water body?")
    assert verdict.action == "abstain"


@pytest.mark.parametrize("query", ["How many water bodies are here?",
                                    "What percentage is water?",
                                    "Which land cover is dominant?"])
def test_single_date_statistics_do_not_route_to_change_qa(query):
    assert RuleRouter().route(query)[0] == "vqa"


def test_rgb_does_not_claim_water_absent_without_an_estimator(spectral_only):
    t = tile(bands={k: np.ones((32, 32), np.float32) * .1 for k in ("red", "green", "blue")})
    result = VQAHead().run([t], Slots(raw_query="Is there water?", target_class="water"), {})
    assert result.abstained
    assert "cannot determine" in result.text


def test_cross_modal_pair_does_not_use_texture_for_alignment():
    rng = np.random.default_rng(42)
    optical = tile(bands={"nir": rng.random((32, 32), dtype=np.float32)})
    sar = tile(bands={"VV": rng.random((32, 32), dtype=np.float32)}, sensor_id="s1_grd")
    pair = check_pair(optical, sar, "cross_modal")
    assert pair.ok and pair.aligned
    assert pair.residual_shift_px == (0, 0)


def test_cross_modal_inputs_cannot_be_change_pair():
    a = tile(ts="2025-01-01")
    b = tile(bands={"VV": np.ones((32, 32))}, ts="2026-01-01")
    verdict, _ = run_gate("change", [a, b])
    assert verdict.action == "abstain"
    assert "matching modalities" in verdict.reason


def test_same_shape_different_grid_reprojects_pixels():
    a = tile(crs="EPSG:32643", transform=(10, 0, 500000, 0, -10, 2800000),
             bounds=(500000, 2799680, 500320, 2800000))
    b = replace(a, id="shifted", transform=(10, 0, 500010, 0, -10, 2800000),
                bounds=(500010, 2799680, 500330, 2800000))
    result = resample_to(b, a)
    assert result is not b
    assert result.qa_mask[:, 0].all()
    assert result.transform == a.transform
    assert result.valid_mask[:, 1:].all()


def test_footprint_comparison_transforms_crs():
    from rasterio.warp import transform_bounds
    a = tile(crs="EPSG:32643", bounds=(500000, 2799680, 500320, 2800000))
    b = replace(a, crs="EPSG:4326", bounds=transform_bounds(a.crs, "EPSG:4326", *a.bounds))
    assert footprint_iou(a, b) > .99


def test_alignment_marks_unobserved_edges_invalid():
    a = tile()
    result = align_pair(a, a, (3, -2))
    assert result.qa_mask[:3].all()
    assert result.qa_mask[:, -2:].all()


def test_change_mask_excludes_nodata_and_cloud():
    a = tile()
    b = replace(a, bands={k: x.copy() for k, x in a.bands.items()},
                qa_mask=np.zeros(a.shape, bool))
    b.qa_mask[:8] = True
    for arr in b.bands.values():
        arr[:8] = 20
    assert not ChangeHead._cva(a, b, 1.5).any()


def test_small_changed_region_is_not_erased_by_percentile_normalization():
    a = tile()
    b = replace(a, bands={k: x.copy() for k, x in a.bands.items()})
    for arr in b.bands.values():
        arr[10:14, 10:14] += .2
    assert ChangeHead._cva(a, b, 1.5).sum() == 16


def test_fusion_uses_sar_through_cloud_but_excludes_missing_radar(spectral_only):
    opt = tile(qa_mask=np.ones((32, 32), bool))
    sar = tile(bands={"VV": np.full((32, 32), -20, np.float32)}, sensor_id="s1_grd",
               qa_mask=np.zeros((32, 32), bool))
    sar.qa_mask[:8] = True
    result = FuseHead().run([opt, sar], Slots(), {})
    assert not result.visual.mask[:8].any()
    assert (result.visual.mask[8:] == 1).all()
    assert result.raw["sar_carried_water"] is True


def test_change_area_uses_common_clear_pixels():
    a = tile(ts="2025-01-01")
    b = replace(a, ts="2026-01-01", qa_mask=np.zeros(a.shape, bool))
    b.qa_mask[:8] = True
    result = ChangeHead().run([a, b], Slots(target_class="water"),
                              {"_effective_intent": "change-qa"})
    assert result.raw["deltas"]["water"]["delta_pct"] == 0
    assert result.raw["interval_days"] == 365


def test_unmeasurable_change_target_is_an_explicit_abstention():
    result = ChangeHead().run([tile(ts="2025-01-01"), tile(ts="2026-01-01")],
                              Slots(target_class="built_up"), {"_effective_intent": "change-qa"})
    assert result.abstained


def test_explicit_head_abstention_survives_calibration():
    result = TaskResult(text="No comparable observations", confidence=.99, abstained=True)
    answer = merge(result, "vqa", {}, [tile()], TraceRecord(run_id="test"))
    assert answer.abstained
    assert answer.abstain_reason == result.text


def test_nonfinite_confidence_cannot_become_an_answer():
    assert calibrate(float("nan"), "vqa") == 0
    for temperature in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            calibrate(.8, "vqa", {"vqa": temperature})


def test_temporal_slots_keep_full_years_and_relative_references():
    slots = extract_slots("Compare before 2024 and after 2026 monsoon", "change")
    assert slots.temporal_refs == ["before", "2024", "after", "2026", "monsoon"]


def test_pdf_accepts_literal_markup_in_user_text():
    trace = TraceRecord(run_id="test", query='Is <img src="bad"> & <b> literal?',
                        answer=Answer(text="Text <font is not markup & water>",
                                      confidence=.8, intent="vqa"))
    pdf = build_pdf(trace)
    assert pdf.startswith(b"%PDF-")

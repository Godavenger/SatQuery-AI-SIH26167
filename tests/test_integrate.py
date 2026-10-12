"""Unit tests for Stage 3 confidence calibration (satquery.stage3.integrate).

These use no model weights, so they pass on a fresh clone.
Run:  pip install pytest  &&  python -m pytest tests -q
"""
from satquery.stage3.integrate import (
    calibrate, DEFAULT_TEMPERATURES, DEFAULT_ABSTAIN_THRESH)


def test_calibrate_bounds():
    for p in (0.0, 0.01, 0.5, 0.87, 0.999, 1.0):
        for task in ("vqa", "change-qa", "describe", "unknown-task"):
            c = calibrate(p, task)
            assert 0.0 <= c <= 1.0


def test_calibrate_monotonic():
    # higher raw confidence -> higher (or equal) calibrated confidence
    xs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    cs = [calibrate(x, "vqa") for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(cs, cs[1:]))


def test_calibrate_identity_at_T1():
    # a task with no entry falls back to T=1.0, which is (near) identity
    for p in (0.3, 0.5, 0.73):
        assert abs(calibrate(p, "no-such-task") - p) < 1e-3


def test_calibrate_T_gt_1_softens():
    # vqa uses a fitted T=2.11 > 1, which pulls values toward 0.5
    assert calibrate(0.95, "vqa") < 0.95
    assert calibrate(0.05, "vqa") > 0.05


def test_temperature_and_threshold_tables_sane():
    for task in ("vqa", "describe", "ground", "change", "change-qa", "fuse"):
        assert DEFAULT_TEMPERATURES[task] > 0
        assert 0.0 < DEFAULT_ABSTAIN_THRESH[task] < 1.0

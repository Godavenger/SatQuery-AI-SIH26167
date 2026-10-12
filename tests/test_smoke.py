"""Import smoke tests for the pure-logic layer (no model weights required)."""
import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "satquery.contracts",
    "satquery.stage3.integrate",
])
def test_import(mod):
    assert importlib.import_module(mod) is not None


def test_new_run_id_shape():
    from satquery.contracts import new_run_id
    rid = new_run_id()
    # api._valid_run_id accepts [A-Za-z0-9_-]{1,80}; ids must stay in that set
    assert rid.startswith("run_")
    assert all(c.isalnum() or c in "_-" for c in rid)
    assert len(rid) <= 80

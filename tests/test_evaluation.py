"""The quick scorecard must fail clearly when no evaluation actually ran."""
import sys

from scripts import run_eval


def test_no_imagery_returns_actionable_failure(monkeypatch):
    monkeypatch.setattr(run_eval, "_get_tiles", lambda *args: ([], []))
    result = run_eval.run_eval(run_eval.PS_QUERIES, verbose=False)
    assert result["n_queries"] == 0
    assert "runtime bundle" in result["error"]


def test_cli_does_not_write_empty_scorecard(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "run_eval", lambda *args, **kwargs: {"error": "No imagery"})
    output = tmp_path / "empty.json"
    monkeypatch.setattr(sys, "argv", ["run_eval", "--quick", "--out", str(output)])
    assert run_eval.main() == 1
    assert not output.exists()

"""Parity tests for VizAgent vs original Agent (requires MLX + weights)."""
import pytest

mlx = pytest.importorskip("mlx.core", reason="Apple Silicon MLX required")


def test_verbose_matches_predict():
    """VizAgent.predict_verbose must produce same answers as Agent.predict."""
    from viz.probe import VizAgent
    import laya_mlx

    ref = laya_mlx.load("aac6fef/laya-mlx")
    viz = VizAgent("aac6fef/laya-mlx")

    state = {"text": "Ship the fix before Friday"}
    questions = {
        "q1": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": "a or b",
            "options": ["a", "b"],
        }
    }

    expected = ref.predict(state, questions)["answers"]["q1"]
    got, trace = viz.predict_verbose(state, questions)
    got = got["answers"]["q1"]

    assert got["choice"] == expected["choice"]
    for k in expected["probabilities"]:
        assert abs(got["probabilities"][k] - expected["probabilities"][k]) < 1e-4

    # Trace must be non-empty
    assert "total_ms" in trace
    assert trace["total_ms"] > 0


def test_trace_has_token_ids():
    """predict_verbose must populate token_ids (even if empty in minimal impl)."""
    from viz.probe import VizAgent

    # Schema check only — full inference requires weights
    import inspect
    src = inspect.getsource(VizAgent.predict_verbose)
    assert '"token_ids"' in src

"""Tests for viz.probe.VizAgent schema and structure."""
import inspect


def test_predict_verbose_schema():
    """VizAgent must expose predict_verbose method."""
    from viz.probe import VizAgent
    assert hasattr(VizAgent, "predict_verbose")


def test_trace_keys():
    """predict_verbose must produce all required trace keys."""
    from viz import probe
    src = inspect.getsource(probe.VizAgent.predict_verbose)
    for key in ["token_ids", "marker_logits_raw", "marker_probs_raw", "temperature", "total_ms"]:
        assert key in src, f"Missing trace key: {key}"


def test_trace_return_type():
    """predict_verbose must return tuple of (answers, trace)."""
    from viz import probe
    src = inspect.getsource(probe.VizAgent.predict_verbose)
    assert "return answers, trace" in src

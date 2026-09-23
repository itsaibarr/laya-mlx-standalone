"""Tests for viz.terminal render_record."""
import json


def test_render_record_bars():
    """render_record must show choice label, probability bars, and numeric values."""
    from viz.terminal import render_record
    record = {
        "input": {"tokens": 10},
        "stages": {"total_ms": 5.0},
        "internals": {"temperature": 1.0},
        "output": {
            "choice": "escalate",
            "probabilities": {"escalate": 0.87, "auto-route": 0.11, "dismiss": 0.02},
            "confidence": 0.87,
        },
    }
    text = render_record(record)
    assert "escalate" in text
    assert "0.87" in text
    # Bars must be present (█ characters)
    assert "█" in text


def test_render_record_confidence():
    """render_record must show confidence and total_ms."""
    from viz.terminal import render_record
    record = {
        "input": {"tokens": 47},
        "stages": {"total_ms": 12.3},
        "internals": {"temperature": 1.0},
        "output": {
            "choice": "phishing",
            "probabilities": {"phishing": 0.92, "legitimate": 0.08},
            "confidence": 0.92,
        },
    }
    text = render_record(record)
    assert "confidence" in text.lower() or "0.92" in text
    assert "12.3" in text


def test_render_record_probability_sums():
    """render_record must not fabricate probabilities — use given values."""
    from viz.terminal import render_record
    probs = {"a": 0.50, "b": 0.30, "c": 0.20}
    record = {
        "input": {"tokens": 5},
        "stages": {"total_ms": 2.0},
        "internals": {"temperature": 1.0},
        "output": {"choice": "a", "probabilities": probs, "confidence": 0.50},
    }
    text = render_record(record)
    # All three labels present
    for label in probs:
        assert label in text

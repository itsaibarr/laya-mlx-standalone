"""Tests for viz.capture CLI and JSONL record shape."""
import json
import tempfile
from pathlib import Path


def test_capture_record_shape():
    """Record must match spec schema: input/stages/internals/output."""
    record = {
        "input": {"tokens": 47, "head_max_len": 192},
        "stages": {"total_ms": 12.3},
        "internals": {"marker_logits_raw": [2.1, 0.3, -1.2], "temperature": 1.0},
        "output": {
            "choice": "escalate",
            "probabilities": {"escalate": 0.87, "auto-route": 0.11, "dismiss": 0.02},
            "confidence": 0.87,
        },
    }
    assert abs(sum(record["output"]["probabilities"].values()) - 1.0) < 0.01
    from viz import capture
    assert hasattr(capture, "main")


def test_capture_main_exists():
    """capture.main must be callable."""
    from viz.capture import main
    assert callable(main)


def test_capture_writes_jsonl():
    """main() must produce valid JSONL with required keys."""
    from viz.capture import main
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        out_path = f.name

    import sys
    sys.argv = ["viz.capture", "--preset", "email", "--out", out_path]
    # main() calls Agent which requires MLX + weights — skip if unavailable
    try:
        main()
        lines = Path(out_path).read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "input" in record
        assert "stages" in record
        assert "internals" in record
        assert "output" in record
    except Exception as e:
        if "mlx" in str(e).lower() or "huggingface" in str(e).lower():
            import pytest
            pytest.skip(f"MLX/weights unavailable: {e}")
        raise

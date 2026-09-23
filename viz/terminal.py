"""Rich/plain terminal rendering for one pipeline record.

Extends the pattern from laya_mlx/snake/ui.py:70 Canvas + bars + SHIELD.
Renders probability bars, confidence gate, latency, and temperature.
"""


def render_record(record: dict) -> str:
    """Render a single pipeline JSONL record as plain-text bars.

    Args:
        record: dict with keys input/stages/internals/output per spec.

    Returns:
        str: multi-line plain-text rendering with probability bars.
    """
    out = record.get("output", {})
    probs = out.get("probabilities", {})
    stages = record.get("stages", {})
    internals = record.get("internals", {})
    inp = record.get("input", {})

    lines = []

    # Header: choice + confidence
    lines.append(
        f"choice: {out.get('choice', '?')}  "
        f"confidence: {out.get('confidence', '?')}"
    )

    # Tokens + latency
    lines.append(
        f"tokens: {inp.get('tokens', '?')}  "
        f"total_ms: {stages.get('total_ms', '?')}"
    )

    # Temperature
    temp = internals.get("temperature", {})
    if temp:
        lines.append(f"temperature: {temp}")

    # Probability bars (sorted descending)
    for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        p_float = float(p)
        bar_len = round(p_float * 20)
        bars = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {label:12} {bars} {p_float:.2f}")

    # Confidence gate annotation
    conf = float(out.get("confidence", 0))
    if conf >= 0.85:
        lines.append(f"  confidence gate: {conf:.2f} >= 0.85 → AUTO")
    else:
        lines.append(f"  confidence gate: {conf:.2f} < 0.85 → HUMAN REVIEW")

    return "\n".join(lines) + "\n"

"""Capture real laya-mlx inference to JSONL for the pipeline explorer."""
import argparse
import json
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture laya-mlx inference to JSONL")
    ap.add_argument("--preset", default="email", choices=["email"])
    ap.add_argument("--model", default="aac6fef/laya-mlx")
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    args = ap.parse_args()

    PRESETS = {
        "email": (
            {"subject": "Your account will be locked in 24h", "body": "Click here to verify"},
            {"q1": {
                "type": "choice",
                "instructions": "Triage this email",
                "criteria": {
                    "phishing": "fraudulent attempt to steal credentials or money",
                    "legitimate": "genuine message from a real sender",
                    "needs-review": "cannot decide, needs a human",
                },
            }},
        ),
    }

    state, questions = PRESETS[args.preset]

    from viz.probe import VizAgent

    t0 = time.perf_counter()
    agent = VizAgent(args.model)
    load_ms = (time.perf_counter() - t0) * 1000.0

    answers, trace = agent.predict_verbose(state, questions)
    qid = next(iter(questions))

    record = {
        "input": {
            "tokens": int(answers.get("usage", {}).get("input_tokens", 0)),
            "head_max_len": 192,
        },
        "stages": {
            "load_ms": round(load_ms, 2),
            "total_ms": trace["total_ms"],
        },
        "internals": {
            "marker_logits_raw": trace["marker_logits_raw"],
            "temperature": trace["temperature"],
        },
        "output": answers["answers"][qid],
    }

    with open(args.out, "w") as f:
        f.write(json.dumps(record) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

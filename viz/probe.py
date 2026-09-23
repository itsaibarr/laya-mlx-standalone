"""VizAgent: subclass of laya_mlx.agent.Agent with predict_verbose() for pipeline visualization."""
from laya_mlx.agent import Agent


class VizAgent(Agent):
    """Agent that exposes intermediate trace data alongside predictions."""

    def predict_verbose(self, state, questions):
        """Run prediction and return both answers and trace data.

        Returns:
            tuple[dict, dict]: (answers, trace) where trace contains:
                token_ids, marker_logits_raw, marker_probs_raw,
                temperature, encode_ms, total_ms
        """
        import time

        t0 = time.perf_counter()
        answers = self.predict(state, questions)
        total_ms = (time.perf_counter() - t0) * 1000.0

        first_qid = next(iter(questions))
        temp = getattr(self, "temperature", 1.0)
        by_opts = getattr(self, "temperature_by_options", {})
        trace = {
            "token_ids": [],
            "marker_logits_raw": [],
            "marker_probs_raw": [],
            "temperature": list(temp) if isinstance(temp, (list, tuple)) else temp,
            "temperature_by_options": dict(by_opts) if isinstance(by_opts, dict) else {},
            "encode_ms": 0.0,
            "total_ms": round(total_ms, 2),
        }
        return answers, trace

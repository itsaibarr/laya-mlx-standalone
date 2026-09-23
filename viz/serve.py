"""Live playground backend: serves viz-web/ and runs real laya-mlx inference.

Usage:
    python3 viz/serve.py --model aac6fef/laya-mlx --port 8137

Endpoints:
    GET  /                    -> viz-web/index.html
    GET  /runs/<file>         -> bundled JSONL captures
    GET  /api/health          -> {"ok": true, "model": ...}
    GET  /api/presets         -> upstream question packs (triage/email/guard/...)
    POST /api/predict         -> {state: {...}, questions: {...}}
                                 returns {answers, trace, stages}

Stdlib only. Model loads once at startup and stays resident.
"""

import argparse
import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from viz.probe import VizAgent

AGENT: VizAgent | None = None
MODEL_ID = ""
WEB_DIR = Path(__file__).resolve().parent.parent / "viz-web"
LOAD_MS = 0.0
REQ_SEQ = 0  # increments per /api/predict call — proof each Run is a fresh inference

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "laya-viz/1.0"

    def log_message(self, format, *args):
        pass  # quiet; see /tmp/viz-serve.log for startup only

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "model": MODEL_ID, "load_ms": LOAD_MS})
            return
        if parsed.path == "/api/presets":
            try:
                from laya_mlx import presets as P

                packs = {}
                for name in ["triage", "email", "guard", "moderation", "router"]:
                    fn = getattr(P, f"{name}_questions", None)
                    if fn is None:
                        continue
                    try:
                        packs[name] = fn() if name != "email" else fn()
                    except Exception:
                        pass
                self._send_json({"presets": packs})
            except Exception as e:  # pragma: no cover
                self._send_json({"error": str(e)}, status=500)
            return
        rel = parsed.path.lstrip("/") or "index.html"
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR)) or not target.is_file():
            self.send_error(404, "not found")
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            state = payload.get("state")
            questions = payload.get("questions")
            if not isinstance(state, dict) or not isinstance(questions, dict) or not questions:
                raise ValueError("body must be {state: {...}, questions: {<id>: {...}}}")
            assert AGENT is not None, "model not loaded"
            global REQ_SEQ
            REQ_SEQ += 1
            answers, trace = AGENT.predict_verbose(state, questions)
            self._send_json({
                "request_id": REQ_SEQ,
                "answers": answers["answers"],
                "usage": answers.get("usage", {}),
                "model": answers.get("model", MODEL_ID),
                "stages": {"load_ms": LOAD_MS, "total_ms": trace.get("total_ms")},
                "internals": {
                    "temperature": trace.get("temperature"),
                    "temperature_by_options": trace.get("temperature_by_options", {}),
                },
            })
        except Exception as e:  # invalid question shape -> 400 with message
            self._send_json({"error": f"{type(e).__name__}: {e}"}, status=400)


def main() -> None:
    global AGENT, MODEL_ID, LOAD_MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aac6fef/laya-mlx")
    ap.add_argument("--port", type=int, default=8137)
    args = ap.parse_args()
    MODEL_ID = args.model
    t0 = time.perf_counter()
    AGENT = VizAgent(args.model)
    LOAD_MS = round((time.perf_counter() - t0) * 1000.0, 2)
    # Warm up Metal kernels so first user query is fast.
    try:
        AGENT.predict(
            {"text": "warmup"},
            {"w": {"type": "noul", "instructions": "Is this a warmup?"}},
        )
    except Exception:
        pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"laya-viz live on http://localhost:{args.port} (model {args.model}, load {LOAD_MS}ms)",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

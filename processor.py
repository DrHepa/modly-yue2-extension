"""One Modly JSON request, structured progress, exactly one terminal response."""
from __future__ import annotations
import contextlib
import json
import os
import signal
import sys
import threading
import traceback


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    output = sys.stdout
    def emit(message):
        output.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()
    cancelled = threading.Event()
    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: cancelled.set())
    try:
        raw = sys.stdin.readline(16 * 1024 * 1024 + 1)
        if not raw or len(raw) > 16 * 1024 * 1024:
            raise ValueError("Missing or oversized Modly JSON payload")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Modly payload must be an object")
        # No inference path may turn a missing local checkpoint into a Hub request.
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        emit({"type": "progress", "percent": 1, "label": "Validating YuE2 request"})
        with contextlib.redirect_stdout(sys.stderr):
            from yue2_modly.runtime import execute
            result = execute(payload, emit, cancelled.is_set)
        emit({"type": "done", "result": result})
        return 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        # Do not echo user text or credentials from a failed HTTP request.
        message = f"{type(exc).__name__}: {exc}"
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            token = os.environ.get(key)
            if token:
                message = message.replace(token, "[redacted]")
        emit({"type": "error", "message": message[:4000]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

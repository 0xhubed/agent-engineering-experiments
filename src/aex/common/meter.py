"""A local HTTP proxy that meters chat completions for code we don't control (e.g. the PageIndex SDK).

Third-party agents call the model through their own clients and report no usage. Pointing
them at this proxy instead of the vLLM endpoint puts every call — tokens and wall time —
into the active Ledger, so their cost is accounted exactly like our own calls.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from aex.common.accounting import Ledger
from aex.common.llm import Completion

_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


class MeteringProxy:
    def __init__(self, upstream_base_url: str, *, model: str, local: bool, defaults: dict | None = None,
                 timeout_s: float = 600.0) -> None:
        self.upstream = upstream_base_url.rstrip("/")
        self.model, self.local = model, local
        # Request fields filled in when the caller leaves them out — e.g. temperature 0 and a seed for an SDK
        # that sends neither, so its runs are as repeatable as ours. Fields the caller sets are kept.
        self.defaults = dict(defaults or {})
        self._http = httpx.Client(timeout=timeout_s)
        self._lock = threading.Lock()
        self._ledger: Ledger | None = None
        self._server: ThreadingHTTPServer | None = None

    # ── lifecycle ──

    def __enter__(self) -> MeteringProxy:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                proxy._forward(self, None)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                proxy._forward(self, body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("proxy not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @contextmanager
    def recording(self, ledger: Ledger):
        """Charge every metered call made inside the block to `ledger`."""
        with self._lock:
            self._ledger = ledger
        try:
            yield ledger
        finally:
            with self._lock:
                self._ledger = None

    # ── forwarding ──

    def _record(self, usage: dict | None, latency: float, finish: str | None) -> None:
        if not usage:
            return
        with self._lock:
            if self._ledger is not None:
                self._ledger.record(Completion("", int(usage.get("prompt_tokens", 0)),
                                               int(usage.get("completion_tokens", 0)), latency, self.model,
                                               finish_reason=finish or "stop"), local=self.local)

    def _forward(self, handler: BaseHTTPRequestHandler, body: bytes | None) -> None:
        path = handler.path[3:] if handler.path.startswith("/v1") else handler.path
        url = self.upstream + path
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in _HOP_HEADERS}
        is_chat = body is not None and path.startswith("/chat/completions")
        stream = False
        if is_chat:
            payload = json.loads(body)
            for key, value in self.defaults.items():
                payload.setdefault(key, value)
            stream = bool(payload.get("stream"))
            if stream:
                payload["stream_options"] = {**payload.get("stream_options", {}), "include_usage": True}
            body = json.dumps(payload).encode()
        started = time.perf_counter()
        request = self._http.build_request(handler.command, url, headers=headers, content=body)
        response = self._http.send(request, stream=True)
        try:
            handler.send_response(response.status_code)
            for k, v in response.headers.items():
                if k.lower() not in _HOP_HEADERS:
                    handler.send_header(k, v)
            if stream:
                handler.send_header("Transfer-Encoding", "chunked")
                handler.end_headers()
                usage, finish, buffer = None, None, b""
                for chunk in response.iter_raw():
                    handler.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    handler.wfile.flush()
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    for line in lines:
                        usage, finish = _scan_sse(line, usage, finish)
                usage, finish = _scan_sse(buffer, usage, finish)
                handler.wfile.write(b"0\r\n\r\n")
                if is_chat and response.status_code < 400:
                    self._record(usage, time.perf_counter() - started, finish)
            else:
                data = response.read()
                handler.send_header("Content-Length", str(len(data)))
                handler.end_headers()
                handler.wfile.write(data)
                if is_chat and response.status_code < 400:
                    parsed = json.loads(data)
                    finish = (parsed.get("choices") or [{}])[0].get("finish_reason")
                    self._record(parsed.get("usage"), time.perf_counter() - started, finish)
        finally:
            response.close()


def _scan_sse(line: bytes, usage: dict | None, finish: str | None) -> tuple[dict | None, str | None]:
    line = line.strip()
    if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
        return usage, finish
    try:
        event = json.loads(line[5:])
    except json.JSONDecodeError:
        return usage, finish
    for choice in event.get("choices") or []:
        finish = choice.get("finish_reason") or finish
    return event.get("usage") or usage, finish

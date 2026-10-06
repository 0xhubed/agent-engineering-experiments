import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from aex.common.accounting import Ledger
from aex.common.meter import MeteringProxy


class _Upstream(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append(req)
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"choices": [{"delta": {"content": "hi"}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
            if req.get("stream_options", {}).get("include_usage"):
                chunks.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def upstream():
    _Upstream.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_proxy_records_non_streamed_calls_into_the_active_ledger(upstream):
    with MeteringProxy(upstream, model="m", local=True) as proxy:
        ledger = Ledger()
        with proxy.recording(ledger):
            for _ in range(2):
                r = httpx.post(f"{proxy.base_url}/chat/completions", json={"model": "m", "messages": []})
                assert r.json()["choices"][0]["message"]["content"] == "ok"
        httpx.post(f"{proxy.base_url}/chat/completions", json={"model": "m", "messages": []})   # not recording
    assert ledger.input_tokens == 20 and ledger.output_tokens == 4
    assert ledger.sequential_calls == 2 and ledger.gpu_s is not None


def test_proxy_forces_usage_on_streams_and_passes_the_stream_through(upstream):
    with MeteringProxy(upstream, model="m", local=False) as proxy:
        ledger = Ledger()
        with proxy.recording(ledger):
            with httpx.stream("POST", f"{proxy.base_url}/chat/completions",
                              json={"model": "m", "messages": [], "stream": True}) as r:
                text = "".join(r.iter_text())
    assert '"hi"' in text and "[DONE]" in text
    assert _Upstream.seen[-1]["stream_options"]["include_usage"] is True
    assert ledger.input_tokens == 7 and ledger.output_tokens == 3 and ledger.gpu_s is None


def test_proxy_forwards_other_paths_untouched(upstream):
    with MeteringProxy(upstream, model="m", local=True) as proxy:
        assert httpx.get(f"{proxy.base_url}/models").json() == {"data": [{"id": "m"}]}


def test_proxy_fills_missing_fields_but_keeps_the_callers(upstream):
    defaults = {"temperature": 0.0, "seed": 0, "chat_template_kwargs": {"reasoning_effort": "medium"}}
    with MeteringProxy(upstream, model="m", local=True, defaults=defaults) as proxy:
        httpx.post(f"{proxy.base_url}/chat/completions", json={"model": "m", "messages": []})
        httpx.post(f"{proxy.base_url}/chat/completions", json={"model": "m", "messages": [], "temperature": 0.7,
                                                                "chat_template_kwargs": {"enable_thinking": False}})
    first, second = _Upstream.seen[-2], _Upstream.seen[-1]
    assert first["temperature"] == 0.0 and first["seed"] == 0
    assert first["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert second["temperature"] == 0.7 and second["seed"] == 0
    assert second["chat_template_kwargs"] == {"enable_thinking": False}

import json
import httpx
import pytest

from aex.common.llm import LLMError, MockClient, OpenAICompatClient


def _ok_response(text="42", prompt_tokens=10, completion_tokens=2):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


def test_mock_client_uses_responder_and_counts_words():
    client = MockClient(lambda msgs: "the answer is 7")
    out = client.complete([{"role": "user", "content": "what is it"}])
    assert out.text == "the answer is 7"
    assert out.input_tokens == 3
    assert out.output_tokens == 4
    assert out.model == "mock"


def test_compat_client_parses_usage_and_sends_determinism_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return _ok_response()

    client = OpenAICompatClient("http://spark:8000/v1", "qwen", transport=httpx.MockTransport(handler))
    out = client.complete([{"role": "user", "content": "hi"}], seed=5)
    assert out.text == "42" and out.input_tokens == 10 and out.output_tokens == 2
    assert b'"temperature": 0.0' in seen["body"] or b'"temperature":0.0' in seen["body"]
    assert b'"seed"' in seen["body"]


def test_compat_client_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else _ok_response()

    client = OpenAICompatClient("http://x/v1", "m", transport=httpx.MockTransport(handler), retries=3, backoff_s=0)
    assert client.complete([{"role": "user", "content": "q"}]).text == "42"
    assert calls["n"] == 3


def test_compat_client_raises_llmerror_after_retries_exhausted():
    client = OpenAICompatClient(
        "http://x/v1", "m", transport=httpx.MockTransport(lambda r: httpx.Response(500)), retries=2, backoff_s=0
    )
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "q"}])


def test_compat_client_does_not_retry_4xx():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad"})

    client = OpenAICompatClient("http://x/v1", "m", transport=httpx.MockTransport(handler), retries=3)
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "q"}])
    assert calls["n"] == 1


def test_compat_client_merges_extra_body_and_reports_finish_reason():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    client = OpenAICompatClient("http://x/v1", "m", transport=httpx.MockTransport(handler),
                                extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}})
    out = client.complete([{"role": "user", "content": "q"}])
    assert seen["body"]["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert seen["body"]["temperature"] == 0.0
    assert out.finish_reason == "length"


def test_finish_reason_defaults_to_stop():
    assert _ok_response().json()["choices"][0].get("finish_reason") is None
    client = OpenAICompatClient("http://x/v1", "m", transport=httpx.MockTransport(lambda r: _ok_response()))
    assert client.complete([{"role": "user", "content": "q"}]).finish_reason == "stop"
    assert MockClient(lambda m: "a").complete([{"role": "user", "content": "q"}]).finish_reason == "stop"

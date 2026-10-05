"""One chat interface for local vLLM endpoints, frontier APIs, and tests.

Every experiment talks to models through ChatClient so accounting and
determinism settings are identical across backends.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    finish_reason: str = "stop"   # "length" = hit max_tokens; the text may be incomplete


class LLMError(RuntimeError):
    """Raised when a model call fails after all retries."""


class ChatClient(Protocol):
    model: str
    local: bool

    def complete(self, messages: list[dict], *, max_tokens: int = 1024,
                 temperature: float = 0.0, seed: int = 0) -> Completion: ...


class OpenAICompatClient:
    """Speaks the OpenAI /chat/completions protocol (vLLM, SGLang, most APIs)."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "EMPTY", local: bool = True,
                 retries: int = 3, backoff_s: float = 1.0, timeout_s: float = 300.0,
                 transport: httpx.BaseTransport | None = None, extra_body: dict | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Server-specific request fields, e.g. {"chat_template_kwargs": {"reasoning_effort": "medium"}}.
        self.extra_body = dict(extra_body or {})
        self.local = local
        self._retries = retries
        self._backoff_s = backoff_s
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    def complete(self, messages: list[dict], *, max_tokens: int = 1024,
                 temperature: float = 0.0, seed: int = 0) -> Completion:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed, **self.extra_body}
        last_error = "no attempt made"
        for attempt in range(self._retries):
            started = time.perf_counter()
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                self._wait(attempt)
                continue
            elapsed = time.perf_counter() - started
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                self._wait(attempt)
                continue
            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
            data = response.json()
            usage = data.get("usage", {})
            choice = data["choices"][0]
            return Completion(
                text=choice["message"]["content"] or "",
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=elapsed,
                model=self.model,
                finish_reason=choice.get("finish_reason") or "stop",
            )
        raise LLMError(f"{self.model}: failed after {self._retries} attempts ({last_error})")

    def _wait(self, attempt: int) -> None:
        if self._backoff_s > 0:
            time.sleep(min(self._backoff_s * 2 ** attempt, 30))


class MockClient:
    """Deterministic stand-in for tests: a responder function maps messages to text."""

    def __init__(self, responder: Callable[[list[dict]], str], model: str = "mock",
                 local: bool = True) -> None:
        self.model = model
        self.local = local
        self._responder = responder

    def complete(self, messages: list[dict], *, max_tokens: int = 1024,
                 temperature: float = 0.0, seed: int = 0) -> Completion:
        text = self._responder(messages)
        prompt = " ".join(m["content"] for m in messages)
        return Completion(text=text, input_tokens=len(prompt.split()),
                          output_tokens=len(text.split()), latency_s=0.0, model=self.model)

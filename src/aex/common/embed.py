"""Embedding and reranking clients (vLLM's OpenAI-compatible /embeddings and /rerank), plus mocks.

Vectors are L2-normalised on arrival, so a dot product is the cosine similarity.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from aex.common.llm import LLMError

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    input_tokens: int
    latency_s: float


@dataclass(frozen=True)
class RerankResult:
    scores: list[float]          # one per document, in input order
    input_tokens: int
    latency_s: float


class Embedder(Protocol):
    model: str
    local: bool

    def embed(self, texts: list[str]) -> EmbedResult: ...

    def embed_query(self, text: str) -> EmbedResult: ...


class Reranker(Protocol):
    model: str
    local: bool

    def rerank(self, query: str, documents: list[str]) -> RerankResult: ...


def normalise(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class _Http:
    def __init__(self, base_url: str, *, api_key: str, retries: int, backoff_s: float, timeout_s: float,
                 transport: httpx.BaseTransport | None) -> None:
        self._retries, self._backoff_s = retries, backoff_s
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"},
                                  timeout=timeout_s, transport=transport)

    def post(self, path: str, body: dict) -> tuple[dict, float]:
        last = "no attempt made"
        for attempt in range(self._retries):
            started = time.perf_counter()
            try:
                response = self._http.post(path, json=body)
            except httpx.HTTPError as exc:
                last = f"transport error: {exc}"
            else:
                if response.status_code < 400:
                    return response.json(), time.perf_counter() - started
                if response.status_code < 500:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                last = f"HTTP {response.status_code}"
            if self._backoff_s > 0:
                time.sleep(min(self._backoff_s * 2 ** attempt, 30))
        raise LLMError(f"{path}: failed after {self._retries} attempts ({last})")


class OpenAICompatEmbedder:
    def __init__(self, base_url: str, model: str, *, query_prefix: str = "", batch_size: int = 64,
                 api_key: str = "EMPTY", local: bool = True, retries: int = 3, backoff_s: float = 1.0,
                 timeout_s: float = 300.0, transport: httpx.BaseTransport | None = None) -> None:
        self.model, self.local = model, local
        self._query_prefix, self._batch_size = query_prefix, batch_size
        self._http = _Http(base_url, api_key=api_key, retries=retries, backoff_s=backoff_s,
                           timeout_s=timeout_s, transport=transport)

    def embed(self, texts: list[str]) -> EmbedResult:
        vectors: list[list[float]] = []
        tokens, latency = 0, 0.0
        for i in range(0, len(texts), self._batch_size):
            data, elapsed = self._http.post("/embeddings", {"model": self.model, "input": texts[i:i + self._batch_size]})
            vectors += [normalise(d["embedding"]) for d in sorted(data["data"], key=lambda d: d["index"])]
            tokens += int(data.get("usage", {}).get("prompt_tokens", 0))
            latency += elapsed
        return EmbedResult(vectors, tokens, latency)

    def embed_query(self, text: str) -> EmbedResult:
        return self.embed([self._query_prefix + text])


class OpenAICompatReranker:
    """/rerank client. Templates wrap the query and each document before sending: Qwen3-Reranker's
    yes/no prompt format is not applied server-side by our vLLM build (serving/gb10/README.md)."""

    def __init__(self, base_url: str, model: str, *, query_template: str = "{query}",
                 document_template: str = "{document}", api_key: str = "EMPTY", local: bool = True,
                 retries: int = 3, backoff_s: float = 1.0, timeout_s: float = 300.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.model, self.local = model, local
        self._query_template, self._document_template = query_template, document_template
        self._http = _Http(base_url, api_key=api_key, retries=retries, backoff_s=backoff_s,
                           timeout_s=timeout_s, transport=transport)

    def rerank(self, query: str, documents: list[str]) -> RerankResult:
        if not documents:
            return RerankResult([], 0, 0.0)
        # str.replace, not str.format: document text may contain braces.
        body = {"model": self.model, "query": self._query_template.replace("{query}", query),
                "documents": [self._document_template.replace("{document}", d) for d in documents]}
        data, elapsed = self._http.post("/rerank", body)
        scores = [0.0] * len(documents)
        for r in data["results"]:
            scores[r["index"]] = float(r["relevance_score"])
        return RerankResult(scores, int(data.get("usage", {}).get("total_tokens", 0)), elapsed)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and offline runs."""

    def __init__(self, dim: int = 256, model: str = "hash", local: bool = True) -> None:
        self.dim, self.model, self.local = dim, model, local

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in _WORD.findall(text.casefold()):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return normalise(v)

    def embed(self, texts: list[str]) -> EmbedResult:
        return EmbedResult([self._one(t) for t in texts], sum(len(t.split()) for t in texts), 0.0)

    def embed_query(self, text: str) -> EmbedResult:
        return self.embed([text])


class OverlapReranker:
    """Mock reranker: share of query words present in the document."""

    def __init__(self, model: str = "overlap", local: bool = True) -> None:
        self.model, self.local = model, local

    def rerank(self, query: str, documents: list[str]) -> RerankResult:
        q = set(_WORD.findall(query.casefold()))
        scores = [len(q & set(_WORD.findall(d.casefold()))) / max(len(q), 1) for d in documents]
        return RerankResult(scores, sum(len(d.split()) for d in documents), 0.0)

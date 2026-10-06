"""Retrieval primitives shared by the vector arms: chunking, BM25, RRF, a cached chunk index.

Chunks never cross a page boundary, so every retrieved chunk is attributable to
exactly one page and evidence recall is measured the same way for every arm.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aex.common.embed import Embedder
from aex.experiments.tree_rag.arms.base import IndexStats, Store

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    page: int
    text: str


def chunk_doc(doc, *, max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    if not 0 <= overlap < max_words:
        raise ValueError(f"need 0 <= overlap < max_words, got overlap={overlap}, max_words={max_words}")
    chunks: list[Chunk] = []
    step = max_words - overlap
    for page in doc.pages:
        words = page.text.split()
        if not words:
            continue
        start = 0
        while True:
            chunks.append(Chunk(f"{doc.doc_id}:{page.number}:{start}", doc.doc_id, page.number,
                                " ".join(words[start:start + max_words])))
            if start + max_words >= len(words):
                break
            start += step
    return chunks


class BM25:
    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = [Counter(tokenize(t)) for t in texts]
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df = Counter(term for d in self._docs for term in d)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._k1, self._b = k1, b

    def scores(self, query: str, subset: list[int] | None = None) -> list[float]:
        terms = tokenize(query)
        out = []
        for i in (subset if subset is not None else range(len(self._docs))):
            d, length, s = self._docs[i], self._len[i], 0.0
            for t in terms:
                tf = d.get(t, 0)
                if tf:
                    s += self._idf[t] * tf * (self._k1 + 1) / (tf + self._k1 * (1 - self._b + self._b * length / (self._avg or 1)))
            out.append(s)
        return out


def top_k(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda i: (-fused[i], i))


class ChunkIndex:
    """All chunks of a store, with dense vectors (disk-cached per document) and BM25."""

    def __init__(self, store: Store, embedder: Embedder, *, max_words: int = 200, overlap: int = 40,
                 cache_dir: str | Path | None = None) -> None:
        self.store, self.embedder = store, embedder
        self._max_words, self._overlap = max_words, overlap
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._bm25: BM25 | None = None
        self._by_doc: dict[str, list[int]] = {}

    def _cache_file(self, doc) -> Path | None:
        if self._cache_dir is None:
            return None
        key = hashlib.sha256(f"{doc.sha256}|{self.embedder.model}|{self._max_words}|{self._overlap}".encode()).hexdigest()[:24]
        return self._cache_dir / f"{doc.doc_id}-{key}.json"

    def build(self) -> IndexStats:
        tokens, latency, vectors = 0, 0.0, []
        for doc_id in sorted(self.store.docs):
            doc = self.store.get(doc_id)
            chunks = chunk_doc(doc, max_words=self._max_words, overlap=self._overlap)
            cache = self._cache_file(doc)
            if cache is not None and cache.exists():
                cached = json.loads(cache.read_text())
                doc_vectors = cached["vectors"]
            else:
                result = self.embedder.embed([c.text for c in chunks]) if chunks else None
                cached = {"vectors": result.vectors if result else [],
                          "input_tokens": result.input_tokens if result else 0,
                          "latency_s": result.latency_s if result else 0.0}
                doc_vectors = cached["vectors"]
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(cached))
            # An index costs what it took to build, whether or not this build read it from the cache:
            # arms sharing a cache must each carry the cost.
            tokens, latency = tokens + cached["input_tokens"], latency + cached["latency_s"]
            start = len(self.chunks)
            self.chunks += chunks
            vectors += doc_vectors
            self._by_doc[doc_id] = list(range(start, len(self.chunks)))
        self._vectors = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), np.float32)
        self._bm25 = BM25([c.text for c in self.chunks])
        return IndexStats(input_tokens=tokens, gpu_s=latency if self.embedder.local and tokens else None)

    def candidates(self, doc_ids: list[str]) -> list[int]:
        return [i for d in doc_ids for i in self._by_doc.get(d, [])]

    def dense(self, query_vector: list[float], doc_ids: list[str], k: int) -> list[tuple[int, float]]:
        cand = self.candidates(doc_ids)
        if not cand:
            return []
        scores = (self._vectors[cand] @ np.asarray(query_vector, dtype=np.float32)).tolist()
        return [(cand[i], scores[i]) for i in top_k(scores, k)]

    def lexical(self, query: str, doc_ids: list[str], k: int) -> list[tuple[int, float]]:
        cand = self.candidates(doc_ids)
        scores = self._bm25.scores(query, cand)
        return [(cand[i], scores[i]) for i in top_k(scores, k) if scores[i] > 0]

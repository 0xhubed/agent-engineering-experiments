"""Vector arms: naive dense top-k, and the honest baseline (BM25 + dense, RRF, cross-encoder rerank)."""
from __future__ import annotations

from aex.common.accounting import Ledger
from aex.common.embed import Embedder, Reranker
from aex.common.llm import Completion
from aex.experiments.tree_rag.arms.base import IndexStats, Store, register
from aex.experiments.tree_rag.retrieval import ChunkIndex, rrf
from aex.experiments.tree_rag.types import EvidencePage, Question, Retrieval


def record_call(ledger: Ledger, model: str, local: bool, tokens: int, latency: float,
                group: str | None = None) -> None:
    """Embedding and rerank calls count toward query cost like any model call."""
    ledger.record(Completion("", tokens, 0, latency, model), local=local, parallel_group=group)


class _ChunkArm:
    uses_navigator = False

    def __init__(self, *, navigator=None, embedder: Embedder | None = None, reranker: Reranker | None = None,
                 scopes: dict | None = None, k: int = 8, max_words: int = 200, overlap: int = 40,
                 cache_dir: str | None = None, **options) -> None:
        self.embedder, self.reranker, self.scopes, self.k = embedder, reranker, scopes or {}, k
        self._chunking = dict(max_words=max_words, overlap=overlap, cache_dir=cache_dir)
        self.chunk_index: ChunkIndex | None = None

    def index(self, store: Store) -> IndexStats:
        if self.embedder is None:
            raise ValueError(f"arm {self.name!r} needs an embedder (services.embedder in the run config)")
        self.chunk_index = ChunkIndex(store, self.embedder, **self._chunking)
        return self.chunk_index.build()

    def _scope(self, q: Question, store: Store) -> list[str]:
        return store.scope(q, self.scopes.get(q.dataset, "corpus"))

    def _dense(self, q: Question, scope: list[str], k: int, ledger: Ledger) -> list[tuple[int, float]]:
        qe = self.embedder.embed_query(q.question)
        record_call(ledger, self.embedder.model, self.embedder.local, qe.input_tokens, qe.latency_s)
        return self.chunk_index.dense(qe.vectors[0], scope, k)

    def _result(self, hits: list[tuple[int, float]], ledger: Ledger, extra: dict | None = None) -> Retrieval:
        chunks = [self.chunk_index.chunks[i] for i, _ in hits]
        evidence = [EvidencePage(c.doc_id, c.page, c.text, kind="chunk") for c in chunks]
        trace = {"pages": [[c.doc_id, c.page] for c in chunks], "chunks": [c.id for c in chunks],
                 "scores": [round(s, 4) for _, s in hits], **(extra or {})}
        return Retrieval(evidence, trace, ledger)


@register("chunk_embed")
class ChunkEmbedArm(_ChunkArm):
    family = "vector"

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        ledger = Ledger()
        return self._result(self._dense(q, self._scope(q, store), self.k, ledger), ledger)


@register("hybrid_rerank")
class HybridRerankArm(_ChunkArm):
    family = "vector"

    def __init__(self, *, candidates: int = 50, rerank_top: int = 30, rrf_k: int = 60, **options) -> None:
        super().__init__(**options)
        self.candidates, self.rerank_top, self.rrf_k = candidates, rerank_top, rrf_k

    def index(self, store: Store) -> IndexStats:
        if self.reranker is None:
            raise ValueError("arm 'hybrid_rerank' needs a reranker (services.reranker in the run config)")
        return super().index(store)

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        ledger = Ledger()
        scope = self._scope(q, store)
        dense = self._dense(q, scope, self.candidates, ledger)
        lexical = self.chunk_index.lexical(q.question, scope, self.candidates)
        fused = rrf([[i for i, _ in dense], [i for i, _ in lexical]], k=self.rrf_k)[:self.rerank_top]
        rr = self.reranker.rerank(q.question, [self.chunk_index.chunks[i].text for i in fused])
        record_call(ledger, self.reranker.model, self.reranker.local, rr.input_tokens, rr.latency_s)
        ranked = sorted(zip(fused, rr.scores), key=lambda p: (-p[1], fused.index(p[0])))[:self.k]
        return self._result(ranked, ledger, {"fused": len(fused)})

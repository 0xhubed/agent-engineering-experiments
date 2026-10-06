"""Vectors across documents, tree within a document (VecTree-style, spec §6.1).

Dense search over one vector per document picks the top `doc_k` documents of the corpus;
PageIndex then navigates only inside those. The navigation step is exactly the pageindex
arm's, so the two arms differ only in how the documents are chosen.
"""
from __future__ import annotations

import numpy as np

from aex.common.accounting import Ledger
from aex.common.embed import Embedder
from aex.experiments.tree_rag.arms.base import IndexStats, Store, register
from aex.experiments.tree_rag.arms.pageindex_arm import PageIndexArm
from aex.experiments.tree_rag.arms.vector import record_call
from aex.experiments.tree_rag.retrieval import top_k
from aex.experiments.tree_rag.types import Question, Retrieval


def doc_card(doc, *, max_words: int = 1500) -> str:
    """What a document is, for choosing between documents: title, section titles, first page."""
    parts = [doc.title, " / ".join(n.title for n in doc.tree), doc.pages[0].text if doc.pages else ""]
    return " ".join(" ".join(parts).split()[:max_words])


@register("vec_tree")
class VecTreeArm(PageIndexArm):
    family = "hybrid"
    uses_navigator = True

    def __init__(self, *, embedder: Embedder | None = None, doc_k: int = 3, **options) -> None:
        super().__init__(**options)
        self.embedder, self.doc_k = embedder, doc_k
        self._doc_ids: list[str] = []
        self._doc_vectors: np.ndarray | None = None

    def index(self, store: Store) -> IndexStats:
        if self.embedder is None:
            raise ValueError("arm 'vec_tree' needs an embedder (services.embedder in the run config)")
        tree_stats = super().index(store)
        self._doc_ids = sorted(store.docs)
        cards = self.embedder.embed([doc_card(store.get(d)) for d in self._doc_ids])
        self._doc_vectors = np.asarray(cards.vectors, dtype=np.float32)
        local_s = cards.latency_s if self.embedder.local else None
        return IndexStats(input_tokens=tree_stats.input_tokens + cards.input_tokens,
                          output_tokens=tree_stats.output_tokens,
                          gpu_s=None if tree_stats.gpu_s is None and local_s is None
                          else (tree_stats.gpu_s or 0.0) + (local_s or 0.0))

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        self._ensure_client()
        ledger = Ledger()
        scope = set(store.scope(q, self.scopes.get(q.dataset, "corpus")))
        qe = self.embedder.embed_query(q.question)
        record_call(ledger, self.embedder.model, self.embedder.local, qe.input_tokens, qe.latency_s)
        candidates = [i for i, d in enumerate(self._doc_ids) if d in scope]
        scores = (self._doc_vectors[candidates] @ np.asarray(qe.vectors[0], dtype=np.float32)).tolist()
        chosen = [self._doc_ids[candidates[i]] for i in top_k(scores, self.doc_k)]
        retrieval = self._navigate(q, store, chosen, ledger)
        # The SDK's own answer is scored only for the faithful pageindex arm; keep it here for the explorer.
        retrieval.trace["sdk_answer"] = retrieval.trace.pop("native_answer", None)
        retrieval.trace["doc_candidates"] = chosen
        return retrieval

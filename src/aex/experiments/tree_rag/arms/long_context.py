"""Whole document in context, no retrieval. N/A when it does not fit — never truncated silently.

The arm is handed the question's own document(s): in regime A it measures reading
with document selection solved, like the by-ISIN reference (spec §5.1).
"""
from __future__ import annotations

from aex.common.accounting import Ledger
from aex.experiments.tree_rag.arms.base import IndexStats, Store, register
from aex.experiments.tree_rag.types import EvidencePage, Question, Retrieval


@register("long_context")
class LongContextArm:
    family = "long_context"
    uses_navigator = False

    def __init__(self, *, navigator=None, max_context_words: int = 150_000, **options) -> None:
        # ~1.4 tokens per word for English financial text; 150k words ≈ 210k tokens, inside a 262k window
        # with room for the prompt, the reasoning and the answer.
        self.max_context_words = max_context_words

    def index(self, store: Store) -> IndexStats:
        return IndexStats()

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        docs = [store.get(d) for d in q.doc_ids]
        words = sum(len(p.text.split()) for d in docs for p in d.pages)
        trace = {"doc_given": list(q.doc_ids), "words": words, "limit": self.max_context_words}
        if words > self.max_context_words:
            return Retrieval([], trace, Ledger(), not_applicable=True)
        evidence = [EvidencePage(d.doc_id, p.number, p.text) for d in docs for p in d.pages if p.text.strip()]
        trace["pages"] = [[e.doc_id, e.page] for e in evidence]
        return Retrieval(evidence, trace, Ledger(), max_evidence_words=self.max_context_words)

"""Gold evidence, handed straight to the answerer: the ceiling for every other arm."""
from __future__ import annotations

from aex.common.accounting import Ledger
from aex.common.llm import ChatClient
from aex.experiments.tree_rag.arms.base import IndexStats, Store, register
from aex.experiments.tree_rag.types import EvidencePage, Question, Retrieval


@register("oracle")
class OracleArm:
    family = "oracle"
    uses_navigator = False

    def __init__(self, *, navigator: ChatClient | None = None, **options) -> None:
        pass

    def index(self, store: Store) -> IndexStats:
        return IndexStats()

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        evidence = [EvidencePage(doc_id, page, store.get(doc_id).page(page).text)
                    for doc_id, page in q.evidence]
        return Retrieval(evidence, {"pages": [[doc_id, page] for doc_id, page in q.evidence]}, Ledger())

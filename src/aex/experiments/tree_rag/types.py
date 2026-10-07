"""Data shared by every Tree RAG arm: parsed documents, questions, retrievals."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from aex.common.accounting import Ledger
from aex.common.llm import Completion
from aex.common.scoring import GoldAnswer

Regime = Literal["A", "B", "claim"]
QType = Literal["lookup", "table", "computation", "cross_doc", "disambiguation", "unanswerable"]


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class TreeNode:
    id: str
    parent: str | None
    title: str
    start_page: int
    end_page: int
    level: int


@dataclass(frozen=True)
class ParsedDoc:
    doc_id: str
    title: str
    sha256: str
    pages: tuple[Page, ...]
    tree: tuple[TreeNode, ...]

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def page(self, n: int) -> Page:
        return self.pages[n - 1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ParsedDoc:
        return cls(
            doc_id=d["doc_id"], title=d["title"], sha256=d["sha256"],
            pages=tuple(Page(**p) for p in d["pages"]),
            tree=tuple(TreeNode(**n) for n in d["tree"]),
        )


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: str
    regime: Regime
    qtype: QType
    question: str
    doc_ids: tuple[str, ...]
    gold: GoldAnswer
    evidence: tuple[tuple[str, int], ...]
    split: Literal["dev", "test"]
    pair_id: str | None = None                          # regime A: ISIN and description forms share it
    form: Literal["isin", "description"] | None = None


@dataclass(frozen=True)
class EvidencePage:
    doc_id: str
    page: int
    text: str
    kind: str = "page"            # "page" | "chunk" (text from this page) | "summary" (generated; pages page..end_page)
    end_page: int | None = None


@dataclass
class Retrieval:
    evidence: list[EvidencePage]
    trace: dict
    ledger: Ledger
    not_applicable: bool = False
    max_evidence_words: int | None = None   # overrides the shared budget (only long_context uses this)


@dataclass(frozen=True)
class Answer:
    text: str
    completion: Completion
    truncated: bool

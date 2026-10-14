"""The contract every retrieval method implements. Answering is NOT per-arm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Protocol

from aex.common.llm import ChatClient
from aex.experiments.tree_rag.types import ParsedDoc, Question, Retrieval

ArmFamily = Literal["oracle", "vector", "tree", "hybrid", "long_context"]


@dataclass
class IndexStats:
    input_tokens: int = 0
    output_tokens: int = 0
    gpu_s: float | None = None
    cost_usd: float | None = None


class Store:
    def __init__(self, docs: Iterable[ParsedDoc], groups: dict[str, str] | None = None) -> None:
        self.docs: dict[str, ParsedDoc] = {d.doc_id: d for d in docs}
        # doc_id -> corpus name (from corpus manifests); retrieval never crosses corpora in phase 1
        self.groups: dict[str, str] = dict(groups or {})

    def scope(self, q: Question, mode: str = "corpus") -> list[str]:
        """Documents an arm may search for q: its corpus (default) or only the question's own documents."""
        if mode == "question_docs":
            return list(q.doc_ids)
        if mode != "corpus":
            raise ValueError(f"unknown scope {mode!r}")
        if not self.groups:
            return sorted(self.docs)
        wanted = {self.groups.get(d) for d in q.doc_ids}
        return sorted(d for d in self.docs if self.groups.get(d) in wanted)

    def reachable(self, questions: Iterable[Question], scopes: dict[str, str]) -> Store:
        """The documents some question can search, and nothing else: arms index only what a run can use
        (a dev run then skips the FinanceBench filings only test questions ask about)."""
        keep = {d for q in questions for d in self.scope(q, scopes.get(q.dataset, "corpus"))}
        return Store((self.docs[d] for d in sorted(keep)), groups=self.groups)

    def get(self, doc_id: str) -> ParsedDoc:
        if doc_id not in self.docs:
            raise KeyError(f"document {doc_id!r} is not in the store")
        return self.docs[doc_id]

    def __len__(self) -> int:
        return len(self.docs)


class Arm(Protocol):
    name: str
    family: ArmFamily
    uses_navigator: bool

    def index(self, store: Store) -> IndexStats: ...

    def retrieve(self, q: Question, store: Store) -> Retrieval: ...


ARMS: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def decorate(cls: type) -> type:
        cls.name = name
        ARMS[name] = cls
        return cls
    return decorate


def make_arm(name: str, *, navigator: ChatClient | None = None, **options) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known arms: {sorted(ARMS)}")
    return ARMS[name](navigator=navigator, **options)

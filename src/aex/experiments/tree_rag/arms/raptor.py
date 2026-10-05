"""RAPTOR-style collapsed tree (Sarthi et al., ICLR 2024) — our implementation.

Per document: page-bounded chunks are the leaves; leaf vectors are clustered, each cluster
is summarised by the navigator model, summaries are embedded and clustered again, up to
`max_levels`. Retrieval ranks every node of every level together ("collapsed tree") and
takes nodes in order until the word budget is spent.

Simplifications vs the paper, to state in the article: seeded k-means instead of UMAP +
Gaussian mixtures (hard assignment, no soft membership); cluster count = ceil(n / branching);
a word budget instead of a token budget.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from aex.common.accounting import Ledger
from aex.common.embed import Embedder
from aex.common.llm import ChatClient, LLMError
from aex.experiments.tree_rag.arms.base import IndexStats, Store, register
from aex.experiments.tree_rag.arms.vector import record_call
from aex.experiments.tree_rag.retrieval import chunk_doc, top_k
from aex.experiments.tree_rag.types import EvidencePage, Question, Retrieval

SUMMARY_PROMPT = """Summarise these excerpts from "{title}" in at most {words} words.
Keep every number, date, percentage and named party exactly as written.

Excerpts:
{excerpts}"""


@dataclass
class RNode:
    id: str
    doc_id: str
    level: int                  # 0 = leaf chunk
    text: str
    start: int                  # first page covered
    end: int                    # last page covered
    children: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)


def kmeans(x: np.ndarray, k: int, *, seed: int, iterations: int = 25) -> np.ndarray:
    """Deterministic k-means with k-means++ seeding. Returns a cluster label per row."""
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centres], axis=0)
        total = d2.sum()
        centres.append(x[rng.choice(n, p=d2 / total)] if total > 0 else x[rng.integers(n)])
    centres = np.asarray(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        new = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        if (new == labels).all() and _ > 0:
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    return labels


@register("raptor")
class RaptorArm:
    family = "tree"
    uses_navigator = True

    def __init__(self, *, navigator: ChatClient | None = None, embedder: Embedder | None = None,
                 scopes: dict | None = None, branching: int = 6, max_levels: int = 3, k_words: int = 2000,
                 summary_words: int = 120, max_words: int = 200, overlap: int = 40, seed: int = 17,
                 summary_max_tokens: int = 4096, cache_dir: str | None = None, **options) -> None:
        self.navigator, self.embedder, self.scopes = navigator, embedder, scopes or {}
        self.branching, self.max_levels, self.k_words = branching, max_levels, k_words
        self.summary_words, self.max_words, self.overlap, self.seed = summary_words, max_words, overlap, seed
        self.summary_max_tokens = summary_max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.nodes: dict[str, list[RNode]] = {}
        self._matrix: dict[str, np.ndarray] = {}

    # ── index ──

    def _cache_file(self, doc) -> Path | None:
        if self.cache_dir is None:
            return None
        key = "|".join(map(str, (doc.sha256, self.embedder.model, self.navigator.model, self.branching,
                                 self.max_levels, self.summary_words, self.max_words, self.overlap, self.seed,
                                 json.dumps(getattr(self.navigator, "extra_body", {}) or {}, sort_keys=True))))
        return self.cache_dir / f"raptor-{doc.doc_id}-{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    def index(self, store: Store) -> IndexStats:
        if self.embedder is None:
            raise ValueError("arm 'raptor' needs an embedder (services.embedder in the run config)")
        if self.navigator is None:
            raise ValueError("arm 'raptor' needs a navigator to write summaries (navigators in the run config)")
        ledger = Ledger()
        embed_tokens, embed_s = 0, 0.0
        for doc_id in sorted(store.docs):
            doc = store.get(doc_id)
            cache = self._cache_file(doc)
            if cache is not None and cache.exists():
                nodes = [RNode(**n) for n in json.loads(cache.read_text())]
            else:
                nodes, tokens, seconds = self._build(doc, ledger)
                embed_tokens, embed_s = embed_tokens + tokens, embed_s + seconds
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps([asdict(n) for n in nodes]))
            self.nodes[doc_id] = nodes
            self._matrix[doc_id] = np.asarray([n.vector for n in nodes], dtype=np.float32)
        local_embed = embed_s if self.embedder.local and embed_tokens else None
        gpu = None if ledger.gpu_s is None and local_embed is None else (ledger.gpu_s or 0.0) + (local_embed or 0.0)
        return IndexStats(input_tokens=ledger.input_tokens + embed_tokens, output_tokens=ledger.output_tokens,
                          gpu_s=gpu)

    def _build(self, doc, ledger: Ledger) -> tuple[list[RNode], int, float]:
        chunks = chunk_doc(doc, max_words=self.max_words, overlap=self.overlap)
        if not chunks:
            return [], 0, 0.0
        leaves_embed = self.embedder.embed([c.text for c in chunks])
        tokens, seconds = leaves_embed.input_tokens, leaves_embed.latency_s
        level_nodes = [RNode(c.id, doc.doc_id, 0, c.text, c.page, c.page, [], v)
                       for c, v in zip(chunks, leaves_embed.vectors)]
        nodes = list(level_nodes)
        level = 0
        while len(level_nodes) > self.branching and level < self.max_levels:
            level += 1
            labels = kmeans(np.asarray([n.vector for n in level_nodes], dtype=np.float32),
                            math.ceil(len(level_nodes) / self.branching), seed=self.seed)
            parents: list[RNode] = []
            for j in sorted(set(labels.tolist())):
                members = [n for n, lab in zip(level_nodes, labels) if lab == j]
                summary = self._summarise(doc.title, members, ledger)
                parents.append(RNode(f"{doc.doc_id}:L{level}:{j}", doc.doc_id, level, summary,
                                     min(m.start for m in members), max(m.end for m in members),
                                     [m.id for m in members]))
            embedded = self.embedder.embed([p.text for p in parents])
            tokens, seconds = tokens + embedded.input_tokens, seconds + embedded.latency_s
            for p, v in zip(parents, embedded.vectors):
                p.vector = v
            nodes += parents
            level_nodes = parents
        return nodes, tokens, seconds

    def _summarise(self, title: str, members: list[RNode], ledger: Ledger) -> str:
        excerpts = "\n\n".join(f"[p.{m.start}" + (f"-{m.end}" if m.end != m.start else "") + f"] {m.text}"
                               for m in members)
        prompt = SUMMARY_PROMPT.format(title=title, words=self.summary_words, excerpts=excerpts)
        completion = self.navigator.complete([{"role": "user", "content": prompt}],
                                             max_tokens=self.summary_max_tokens, seed=self.seed)
        ledger.record(completion, local=self.navigator.local)
        if completion.finish_reason == "length":
            raise LLMError(f"raptor: summary truncated at max_tokens={self.summary_max_tokens}")
        return completion.text.strip()

    # ── query ──

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        ledger = Ledger()
        qe = self.embedder.embed_query(q.question)
        record_call(ledger, self.embedder.model, self.embedder.local, qe.input_tokens, qe.latency_s)
        qv = np.asarray(qe.vectors[0], dtype=np.float32)
        candidates: list[tuple[RNode, float]] = []
        for doc_id in store.scope(q, self.scopes.get(q.dataset, "corpus")):
            if doc_id in self.nodes and len(self.nodes[doc_id]):
                scores = (self._matrix[doc_id] @ qv).tolist()
                candidates += list(zip(self.nodes[doc_id], scores))
        ranked = [candidates[i] for i in top_k([s for _, s in candidates], len(candidates))]
        chosen, used = [], 0
        for node, score in ranked:
            words = len(node.text.split())
            if used + words > self.k_words:
                continue
            chosen.append((node, score))
            used += words
            if used >= self.k_words:
                break
        evidence = [EvidencePage(n.doc_id, n.start, n.text, kind="chunk") if n.level == 0 else
                    EvidencePage(n.doc_id, n.start, n.text, kind="summary", end_page=n.end) for n, _ in chosen]
        trace = {"pages": [[n.doc_id, n.start] for n, _ in chosen if n.level == 0],
                 "nodes": [{"id": n.id, "level": n.level, "score": round(s, 4)} for n, s in chosen],
                 "summaries": sum(1 for n, _ in chosen if n.level > 0)}
        return Retrieval(evidence, trace, ledger)

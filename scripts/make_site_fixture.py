"""Synthetic Tree RAG results for building the site before real runs exist.

EVERYTHING HERE IS FAKE: documents, questions, hit rates, latencies. The
data_version 2000-01-01.0 tells the site to label the page SYNTHETIC.
"""
from __future__ import annotations

import hashlib
import random
import re
import sys
import tempfile
from pathlib import Path

from aex.common.accounting import Ledger
from aex.common.checkpoint import Checkpoint
from aex.common.llm import Completion, MockClient
from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.arms import ARMS, IndexStats, Store
from aex.experiments.tree_rag.export import export
from aex.experiments.tree_rag.gold import assign_split
from aex.experiments.tree_rag.parse import build_tree
from aex.experiments.tree_rag.run import RunConfig, run_experiment
from aex.experiments.tree_rag.types import EvidencePage, Page, ParsedDoc, Question, Retrieval

FIXTURE_VERSION = "2000-01-01.0"
LOCAL, FRONTIER = "qwen3.6-35b-a3b", "frontier-ref"
QTYPES = ["lookup", "table", "computation", "cross_doc", "disambiguation", "unanswerable"]
HIT_RATE = {  # synthetic probability of retrieving the gold page, by regime
    "chunk_embed":   {"A": 0.55, "B": 0.62, "claim": 0.60},
    "hybrid_rerank": {"A": 0.70, "B": 0.78, "claim": 0.80},
    "raptor":        {"A": 0.60, "B": 0.72, "claim": 0.70},
    "pageindex":     {"A": 0.62, "B": 0.86, "claim": 0.86},
    "vec_tree":      {"A": 0.83, "B": 0.84, "claim": 0.84},
    "long_context":  {"A": 0.90, "B": 0.00, "claim": 0.88},
}
FAMILY = {"chunk_embed": "vector", "hybrid_rerank": "vector", "raptor": "tree", "pageindex": "tree",
          "vec_tree": "hybrid", "long_context": "long_context"}
NAVIGATING = {"pageindex", "vec_tree"}
NAV_BONUS = {FRONTIER: 0.06}
LONG_CONTEXT_MAX_PAGES = 80
FACT = re.compile(r"Field (F\d+) of ([\w-]+) is ([\d.]+%)")
ASK = re.compile(r"field (F\d+) of ([\w-]+)\?")


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


# ---------- synthetic corpus ----------

def _corpus_layout() -> dict[str, tuple[str, int, list[tuple[str, int, int]]]]:
    layout = {}
    for i in range(1, 9):  # regime A: near-identical termsheets
        layout[f"sp-{i:03d}"] = (f"Barrier reverse convertible P{i} (synthetic)", 6,
                                 [("Product terms", 1, 1), ("Barrier and coupon", 1, 2),
                                  ("Observation schedule", 1, 4), ("Risk factors", 1, 6)])
    for prefix, count, pages, title in [("bp", 3, 120, "Base prospectus"), ("fb", 2, 60, "Annual report")]:
        chapters, per = (10, 12) if pages == 120 else (6, 10)
        for i in range(1, count + 1):
            headings = []
            for ch in range(chapters):
                start = 1 + ch * per
                headings.append((f"Chapter {ch + 1}", 1, start))
                headings += [(f"{ch + 1}.{s + 1} Section", 2, start + 1 + s * 3) for s in range(3)]
            layout[f"{prefix}-{i:03d}"] = (f"{title} {prefix.upper()}{i} (synthetic)", pages, headings)
    return layout


def build_corpus_and_questions(seed: int = 17) -> tuple[Store, list[Question]]:
    layout = _corpus_layout()
    texts = {doc_id: [[f"{title}, page {n}. Standard provisions apply."] for n in range(1, n_pages + 1)]
             for doc_id, (title, n_pages, _) in layout.items()}
    rng = _rng("questions", seed)
    questions: list[Question] = []
    for regime, prefix, n in [("A", "sp", 60), ("B", "bp", 45), ("claim", "fb", 30)]:
        doc_ids = sorted(d for d in layout if d.startswith(prefix))
        for k in range(n):
            qtype, doc_id = QTYPES[k % 6], doc_ids[k % len(doc_ids)]
            qid, field = f"{regime.lower()}-{k:03d}", f"F{k}"
            text = f"What is field {field} of {doc_id}?"
            if qtype == "unanswerable":
                gold, evidence = GoldAnswer("unanswerable", None), ()
            else:
                page = rng.randint(1, layout[doc_id][1])
                value = f"{rng.randint(50, 95)}%"
                texts[doc_id][page - 1].append(f"Field {field} of {doc_id} is {value}.")
                gold, evidence = GoldAnswer("numeric", value), ((doc_id, page),)
            questions.append(Question(qid, "synthetic-" + {"A": "termsheets", "B": "prospectuses",
                                                            "claim": "financebench"}[regime],
                                      regime, qtype, text, (doc_id,), gold, evidence,
                                      assign_split(qid, seed)))
    docs = [ParsedDoc(doc_id, title, f"sha256:synthetic-{doc_id}",
                      tuple(Page(n + 1, " ".join(lines)) for n, lines in enumerate(texts[doc_id])),
                      build_tree(headings, n_pages, title))
            for doc_id, (title, n_pages, headings) in layout.items()]
    return Store(docs), questions


# ---------- simulated arms ----------

def _call(ledger: Ledger, model: str, local: bool, tokens: int, latency: float, group: str | None = None):
    ledger.record(Completion("", tokens, tokens // 25, round(latency, 3), model), local=local, parallel_group=group)


def _vector_like(name, q, store, rng, ledger, hit):
    doc = store.get(q.doc_ids[0])
    others = [d for d in sorted(store.docs) if d != doc.doc_id and d[:2] == doc.doc_id[:2]]
    wrong_doc = q.regime == "A" and not hit and name != "raptor" and others
    pool_doc = store.get(rng.choice(others)) if wrong_doc else doc
    pages = [q.evidence[0]] if hit and q.evidence else []
    candidates = [(pool_doc.doc_id, n) for n in range(1, pool_doc.n_pages + 1) if (pool_doc.doc_id, n) not in q.evidence]
    pages += rng.sample(candidates, min(5 - len(pages), len(candidates)))
    _call(ledger, "embedder", True, 40, 0.12 + rng.random() * 0.08)
    if name == "hybrid_rerank":
        _call(ledger, "reranker", True, 2000, 0.35 + rng.random() * 0.15)
    if name == "raptor":
        _call(ledger, "embedder", True, 40, 0.10)
    scores = [round(0.92 - i * 0.07 - rng.random() * 0.03, 3) for i in range(len(pages))]
    return pages, {"pages": [list(p) for p in pages], "scores": scores}


def _ancestors(tree, node):
    by_id = {n.id: n for n in tree}
    chain = [node]
    while chain[-1].parent:
        chain.append(by_id[chain[-1].parent])
    return list(reversed(chain))


def _tree_like(name, navigator, q, store, rng, ledger, hit):
    nav_id, local = (navigator.model, navigator.local) if navigator else (LOCAL, True)
    trace: dict = {}
    doc = store.get(q.doc_ids[0])
    if name == "vec_tree":
        others = [d for d in sorted(store.docs) if d != doc.doc_id and d[:2] == doc.doc_id[:2]]
        if q.regime == "A" and not hit and others and rng.random() < 0.5:
            doc = store.get(rng.choice(others))
        trace["doc_candidates"] = [doc.doc_id] + others[:2]
        _call(ledger, "embedder", True, 40, 0.15)
    leaves = [n for n in doc.tree if not any(m.parent == n.id for m in doc.tree)]
    gold_page = q.evidence[0][1] if q.evidence and q.evidence[0][0] == doc.doc_id else None
    containing = sorted((n for n in doc.tree if gold_page and n.start_page <= gold_page <= n.end_page),
                        key=lambda n: n.level)
    target = containing[-1] if hit and containing else rng.choice(leaves)
    roots = [n for n in doc.tree if n.parent is None]
    visited = [n.id for n in rng.sample(roots, min(len(roots), rng.randint(1, 3)))]
    backtracks = []
    if not hit or rng.random() < 0.25:
        detour = rng.choice(leaves)
        visited.append(detour.id)
        backtracks.append(detour.id)
    visited += [n.id for n in _ancestors(doc.tree, target) if n.id not in visited]
    per_call = (0.8 + rng.random() * 0.7) if not local else (1.2 + rng.random() * 1.3)
    for _ in range(len(visited) + 1):
        _call(ledger, nav_id, local, 1500, per_call)
    if hit and containing:   # read a window around the evidence, inside the chosen section
        first, last = max(target.start_page, gold_page - 1), min(target.end_page, gold_page + 2)
    else:
        first, last = target.start_page, min(target.end_page, target.start_page + 2)
    pages = [(doc.doc_id, p) for p in range(first, last + 1)]
    trace |= {"doc_id": doc.doc_id, "visited": visited, "backtracks": backtracks,
              "selected": target.id, "pages": [list(p) for p in pages]}
    return pages, trace


def _simulated_arm(name: str) -> type:
    class SimulatedArm:
        family = FAMILY[name]
        uses_navigator = name in NAVIGATING

        def __init__(self, *, navigator=None, **options):
            self.navigator = navigator

        def index(self, store: Store) -> IndexStats:
            tokens = {"raptor": 900_000, "pageindex": 400_000, "vec_tree": 420_000}.get(name, 20_000)
            return IndexStats(input_tokens=tokens, output_tokens=tokens // 10, gpu_s=round(tokens / 2500, 1))

        def retrieve(self, q: Question, store: Store) -> Retrieval:
            nav_id = self.navigator.model if self.navigator else "none"
            rng = _rng(name, nav_id, q.qid)
            ledger = Ledger()
            doc = store.get(q.doc_ids[0])
            if name == "long_context":
                if doc.n_pages > LONG_CONTEXT_MAX_PAGES:
                    return Retrieval([], {}, ledger, not_applicable=True)
                keep_gold = rng.random() < HIT_RATE[name][q.regime]
                pages = [(doc.doc_id, n) for n in range(1, doc.n_pages + 1)
                         if keep_gold or (doc.doc_id, n) not in q.evidence]
                trace = {"pages": [list(p) for p in pages]}
            else:
                hit = rng.random() < HIT_RATE[name][q.regime] + NAV_BONUS.get(nav_id, 0.0)
                simulate = _tree_like if name in NAVIGATING else _vector_like
                args = (name, self.navigator, q, store, rng, ledger, hit) if name in NAVIGATING else (name, q, store, rng, ledger, hit)
                pages, trace = simulate(*args)
            evidence = [EvidencePage(d, p, store.get(d).page(p).text) for d, p in pages]
            return Retrieval(evidence, trace, ledger)

    SimulatedArm.name = name
    return SimulatedArm


# Not registered globally: the real arms share these names. build_fixture swaps them in temporarily.
SIMULATED = {name: _simulated_arm(name) for name in HIT_RATE}


# ---------- simulated answerer ----------

def _answerer(model_id: str):
    def respond(messages: list[dict]) -> str:
        prompt = messages[-1]["content"]
        rng = _rng(model_id, prompt)
        asked = ASK.search(prompt)
        facts = {(f, d): v for f, d, v in FACT.findall(prompt)}
        if asked and (asked.group(1), asked.group(2)) in facts:
            return "ANSWER: 12%" if rng.random() < 0.06 else f"ANSWER: {facts[(asked.group(1), asked.group(2))]}"
        return "ANSWER: 70%" if rng.random() < 0.15 else "ANSWER: NOT_STATED"
    return respond


class _TimedClient:
    """MockClient with plausible token counts and latency, so cost charts have shape."""

    def __init__(self, model: str, local: bool, sec_per_ktok: float):
        self.model, self.local = model, local
        self._inner = MockClient(_answerer(model), model=model, local=local)
        self._sec_per_ktok = sec_per_ktok

    def complete(self, messages, **kw):
        c = self._inner.complete(messages, **kw)
        tokens_in, tokens_out = int(c.input_tokens * 1.3), int(c.output_tokens * 1.3) + 40
        return Completion(c.text, tokens_in, tokens_out, round(0.4 + self._sec_per_ktok * tokens_in / 1000, 3), self.model)


def build_fixture(out_dir: str | Path) -> Path:
    store, questions = build_corpus_and_questions()
    clients = {LOCAL: _TimedClient(LOCAL, True, 0.35), FRONTIER: _TimedClient(FRONTIER, False, 0.12)}
    cfg = RunConfig(seed=17, gold="", parsed_dir="", checkpoint="", arms=["oracle", *HIT_RATE],
                    navigators=[LOCAL, FRONTIER], answerers=[LOCAL], judge=LOCAL, models={},
                    splits=("test",), prices={FRONTIER: [3.0, 15.0]})
    saved = dict(ARMS)
    ARMS.update(SIMULATED)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Checkpoint(Path(tmp) / "fixture.sqlite")
            run_experiment(cfg, clients=clients, store=store, questions=questions, checkpoint=checkpoint)
            rows = checkpoint.rows()
            checkpoint.close()
        return export(rows, questions=questions, store=store,
                      models={LOCAL: {"kind": "openai", "local": True}, FRONTIER: {"kind": "openai", "local": False}},
                      judge=LOCAL, manifest_hash="sha256:" + "f" * 64, data_version=FIXTURE_VERSION,
                      out_dir=out_dir)
    finally:
        ARMS.clear()
        ARMS.update(saved)


if __name__ == "__main__":
    print(f"wrote {build_fixture(sys.argv[1])}")

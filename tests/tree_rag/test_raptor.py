import numpy as np
import pytest

from aex.common.embed import HashEmbedder
from aex.common.llm import MockClient
from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.arms import ARMS, Store, make_arm
from aex.experiments.tree_rag.arms.raptor import kmeans
from aex.experiments.tree_rag.run import evidence_metrics
from aex.experiments.tree_rag.types import Page, ParsedDoc, Question

TOPICS = ["barrier level observation", "coupon payment schedule", "redemption physical delivery",
          "risk factors liquidity", "selling restrictions jurisdiction", "issuer rating guarantor"]


def _doc(doc_id="d1"):
    pages = tuple(Page(i + 1, f"{TOPICS[i % 6]} clause {i} " + " ".join(f"w{i}x{j}" for j in range(30)))
                  for i in range(24))
    return ParsedDoc(doc_id, "Doc", f"sha256:{doc_id}", pages, ())


def _summariser(calls):
    def respond(messages):
        calls.append(messages[-1]["content"])
        body = messages[-1]["content"].split("Excerpts:\n", 1)[1]
        return "SUMMARY " + " ".join(body.split()[:10])
    return MockClient(respond, model="nav")


def _arm(calls, tmp_path=None, **kw):
    return make_arm("raptor", navigator=_summariser(calls), embedder=HashEmbedder(dim=64), max_words=20, overlap=0,
                    branching=4, cache_dir=str(tmp_path) if tmp_path else None, **kw)


def _q(text="barrier level observation"):
    return Question("q1", "d", "B", "lookup", text, ("d1",), GoldAnswer("numeric", "1"), (("d1", 1),), "test")


def test_registered():
    assert ARMS["raptor"].family == "tree" and ARMS["raptor"].uses_navigator


def test_kmeans_is_deterministic_and_assigns_everything():
    x = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    a, b = kmeans(x, 5, seed=17), kmeans(x, 5, seed=17)
    assert (a == b).all() and len(a) == 40 and set(a.tolist()) <= set(range(5))


def test_tree_levels_spans_and_index_cost():
    calls = []
    arm = _arm(calls, max_levels=2)
    stats = arm.index(Store([_doc()]))
    nodes = arm.nodes["d1"]
    levels = sorted({n.level for n in nodes})
    assert levels == [0, 1, 2]
    leaves = {n.id: n for n in nodes if n.level == 0}
    for s in (n for n in nodes if n.level > 0):
        covered = [p for c in s.children for p in ((leaves[c].start, leaves[c].end) if c in leaves else
                   (next(m for m in nodes if m.id == c).start, next(m for m in nodes if m.id == c).end))]
        assert s.start == min(covered) and s.end == max(covered)
    assert stats.input_tokens > 0 and stats.output_tokens > 0 and len(calls) == sum(1 for n in nodes if n.level > 0)


def test_collapsed_retrieval_respects_budget_and_marks_summaries():
    arm = _arm([], k_words=60)
    arm.index(Store([_doc()]))
    r = arm.retrieve(_q(), Store([_doc()]))
    assert sum(len(e.text.split()) for e in r.evidence) <= 60
    kinds = {e.kind for e in r.evidence}
    assert kinds <= {"chunk", "summary"}
    for e in r.evidence:
        if e.kind == "summary":
            assert e.end_page is not None and e.end_page >= e.page
    assert r.ledger.sequential_calls == 1                       # the question embedding


def test_a_summary_only_hit_has_zero_recall():
    arm = _arm([], k_words=400)
    store = Store([_doc()])
    arm.index(store)
    summary = next(n for n in arm.nodes["d1"] if n.level > 0)
    r = arm.retrieve(_q(summary.text), store)
    top = r.evidence[0]
    assert top.kind == "summary"
    only_summaries = type(r)([e for e in r.evidence if e.kind == "summary"], r.trace, r.ledger)
    assert evidence_metrics(_q(), only_summaries)[0] == 0.0


def test_cache_skips_summaries_on_rebuild(tmp_path):
    calls = []
    _arm(calls, tmp_path).index(Store([_doc()]))
    first = len(calls)
    stats = _arm(calls, tmp_path).index(Store([_doc()]))
    assert len(calls) == first and stats.input_tokens == 0


def test_needs_navigator_and_embedder():
    with pytest.raises(ValueError, match="embedder"):
        make_arm("raptor", navigator=_summariser([])).index(Store([_doc()]))
    with pytest.raises(ValueError, match="navigator"):
        make_arm("raptor", embedder=HashEmbedder()).index(Store([_doc()]))

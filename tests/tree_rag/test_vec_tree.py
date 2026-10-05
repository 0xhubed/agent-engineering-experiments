import pytest

import aex.experiments.tree_rag.arms.pageindex_arm as pia
from aex.common.embed import HashEmbedder
from aex.common.llm import OpenAICompatClient
from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.arms import ARMS, Store, make_arm
from aex.experiments.tree_rag.run import evidence_metrics
from aex.experiments.tree_rag.types import Page, ParsedDoc, Question, TreeNode

from tests.tree_rag.test_pageindex_arm import FakeClient


def _doc(doc_id, title, barrier):
    pages = (Page(1, f"{title} product terms"), Page(2, f"Barrier level {barrier}"), Page(3, "Coupon dates"))
    return ParsedDoc(doc_id, title, f"sha256:{doc_id}", pages, (TreeNode("n1", None, "Barrier", 2, 2, 1),))


def _store():
    return Store([_doc("alpha", "Alpha Nestle reverse convertible", "60%"),
                  _doc("beta", "Beta Roche autocallable", "65%"),
                  _doc("gamma", "Gamma Novartis tracker", "70%")],
                 groups={"alpha": "ts", "beta": "ts", "gamma": "ts"})


def _q(text="What is the barrier of the Alpha Nestle reverse convertible?", doc="alpha"):
    return Question("q1", "ts", "A", "disambiguation", text, (doc,), GoldAnswer("numeric", "60%"), ((doc, 2),), "test")


@pytest.fixture
def arm_factory(tmp_path, monkeypatch):
    for d in ("alpha", "beta", "gamma"):
        (tmp_path / f"{d}.pdf").write_bytes(b"%PDF " + d.encode())

    def make(doc_k=2, reads=(("2", "alpha.pdf"),)):
        FakeClient.instances = []
        monkeypatch.setattr(pia, "_client_factory", lambda **config: FakeClient(reads=reads, shown=(), **config))
        nav = OpenAICompatClient("http://upstream/v1", "nav")
        return make_arm("vec_tree", navigator=nav, embedder=HashEmbedder(dim=256), doc_k=doc_k,
                        pdf_dir=str(tmp_path), storage_dir=str(tmp_path / "pi"))
    return make


def test_registered_as_hybrid_navigating_arm():
    assert ARMS["vec_tree"].family == "hybrid" and ARMS["vec_tree"].uses_navigator


def test_vectors_pick_documents_then_pageindex_navigates_only_those(arm_factory):
    arm = arm_factory(doc_k=2)
    stats = arm.index(_store())
    assert stats.input_tokens > 0                                  # document vectors (+ PageIndex index)
    r = arm.retrieve(_q(), _store())
    assert r.trace["doc_candidates"][0] == "alpha" and len(r.trace["doc_candidates"]) == 2
    question, scope, _ = FakeClient.instances[-1].chats[-1]
    assert sorted(scope) == sorted(f"pi-{d}" for d in r.trace["doc_candidates"])
    assert [(e.doc_id, e.page) for e in r.evidence] == [("alpha", 2)]
    assert r.ledger.sequential_calls >= 1                          # the question embedding
    assert "native_answer" not in r.trace and r.trace["sdk_answer"]


def test_doc_k_one_sends_a_single_document_id(arm_factory):
    arm = arm_factory(doc_k=1)
    arm.index(_store())
    arm.retrieve(_q(), _store())
    assert FakeClient.instances[-1].chats[-1][1] == "pi-alpha"


def test_wrong_document_choice_shows_up_as_wrong_doc(arm_factory):
    arm = arm_factory(doc_k=1, reads=(("2", "gamma.pdf"),))
    arm.index(_store())
    q = _q("What is the barrier of the Gamma Novartis tracker?", doc="alpha")   # gold says alpha
    r = arm.retrieve(q, _store())
    assert r.trace["doc_candidates"] == ["gamma"]
    _, _, wrong_doc = evidence_metrics(q, r)
    assert wrong_doc is True


def test_needs_an_embedder():
    with pytest.raises(ValueError, match="embedder"):
        make_arm("vec_tree", navigator=OpenAICompatClient("http://x/v1", "n"), pdf_dir="/tmp").index(_store())

import pytest

from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.arms import ARMS, Store, make_arm
from aex.experiments.tree_rag.types import Page, ParsedDoc, Question


def _doc():
    return ParsedDoc("d1", "Doc", "sha256:x", (Page(1, "intro"), Page(2, "barrier 60%"), Page(3, "dates")), ())


def _q(evidence=(("d1", 2),), kind="numeric"):
    return Question("q1", "fixture", "A", "lookup", "What barrier?", ("d1",),
                    GoldAnswer(kind, "60%"), evidence, "test")


def test_oracle_is_registered_and_returns_gold_pages():
    assert "oracle" in ARMS
    arm = make_arm("oracle")
    store = Store([_doc()])
    assert arm.index(store).input_tokens == 0
    r = arm.retrieve(_q(), store)
    assert [(e.doc_id, e.page, e.text) for e in r.evidence] == [("d1", 2, "barrier 60%")]
    assert r.trace == {"pages": [["d1", 2]]}
    assert r.ledger.sequential_calls == 0 and not r.not_applicable


def test_oracle_on_unanswerable_returns_nothing():
    r = make_arm("oracle").retrieve(_q(evidence=(), kind="unanswerable"), Store([_doc()]))
    assert r.evidence == []


def test_unknown_arm_lists_known_names():
    with pytest.raises(KeyError, match="oracle"):
        make_arm("nope")


def test_store_get_unknown_doc_names_it():
    with pytest.raises(KeyError, match="missing-doc"):
        Store([_doc()]).get("missing-doc")

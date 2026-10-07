from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.types import Question
from aex.gold.report import quota_report, unique


def _q(qid, regime="A", qtype="lookup", split="test", pair=None, form=None):
    kind = "unanswerable" if qtype == "unanswerable" else "numeric"
    return Question(qid, "d", regime, qtype, "q?", ("d",), GoldAnswer(kind, None), (), split, pair, form)


def test_pairs_count_once():
    qs = [_q("p1-isin", pair="p1", form="isin"), _q("p1-desc", pair="p1", form="description"), _q("b1", "B")]
    assert [q.qid for q in unique(qs)] == ["p1-isin", "b1"]


def test_quotas_pass_and_fail():
    full = []
    for r in ("A", "B"):
        for t in ("lookup", "table", "computation", "cross_doc", "disambiguation", "unanswerable"):
            full += [_q(f"{r}-{t}-{i}", r, t) for i in range(30)]
    table, failures = quota_report(full)
    assert failures == [] and "| lookup | 30 | 30 | 60 |" in table
    small = [q for q in full if q.regime == "A" and q.qtype != "unanswerable"][:100]
    _, failures = quota_report(small)
    assert any("regime A: 100 test questions" in f for f in failures)
    assert any("regime A: 0% unanswerable" in f for f in failures)
    assert any(f.startswith("unanswerable: 0") for f in failures)
    assert any("regime B" in f for f in failures)

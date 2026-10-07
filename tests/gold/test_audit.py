import json
import threading
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from aex.gold.audit import Audit, latest, make_handler, model_accepts, report, sample, wilson


def _rec(qid, qtype="lookup", regime="A", verified="claude-opus-5.5", pair=None, form=None):
    r = {"qid": qid, "dataset": "termsheets", "regime": regime, "qtype": qtype, "question": f"Q {qid}?",
         "doc_ids": ["d"], "gold": {"kind": "numeric", "value": "60%"}, "evidence": [{"doc_id": "d", "page": 1}],
         "verified_by": verified}
    return r | ({"pair_id": pair, "form": form} if pair else {})


def test_model_accepts_one_per_pair_preferring_the_description_form(tmp_path):
    p = tmp_path / "g.jsonl"
    rows = [_rec("p1-isin", pair="p1", form="isin"), _rec("p1-desc", pair="p1", form="description"),
            _rec("b1", regime="B"), _rec("h1", verified="daniel")]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [r["qid"] for r in model_accepts([p])] == ["b1", "p1-desc"]


def test_sample_is_stratified_seeded_and_capped():
    recs = [_rec(f"l{i}") for i in range(100)] + [_rec(f"u{i}", "unanswerable") for i in range(3)]
    s = sample(recs, n=20, seed=1)
    assert s == sample(recs, n=20, seed=1)
    assert sum(r["qtype"] == "unanswerable" for r in s) == 3 and 19 <= len(s) <= 21


def test_wilson_and_report():
    lo, hi = wilson(3, 60)
    assert 0.01 < lo < 0.05 < hi < 0.15
    queue = [_rec(f"q{i}") for i in range(4)]
    r = report(queue, {"q0": "correct", "q1": "wrong", "q2": "unsure"})
    assert (r["decided"], r["wrong"], r["unsure"], r["error_rate"]) == (2, 1, 1, 0.5)
    assert r["wrong_qids"] == ["q1"] and r["by_type"]["lookup"] == {"wrong": 1, "n": 2}


def test_audit_page_flow(tmp_path):
    queue = [_rec("q0"), _rec("q1")]
    a = Audit(queue, tmp_path / "grades.jsonl", lambda d, p: "The barrier is 60% of the initial level.")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(a, lambda d: tmp_path / "x.pdf"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "Q q0?" in urllib.request.urlopen(base + "/").read().decode()
        urllib.request.urlopen(base + "/grade", data=urllib.parse.urlencode(
            {"qid": "q0", "action": "wrong", "note": "barrier is 65%"}).encode())
        assert latest(tmp_path / "grades.jsonl") == {"q0": "wrong"}
        assert "Q q1?" in urllib.request.urlopen(base + "/").read().decode()
        urllib.request.urlopen(base + "/undo", data=b"")
        assert latest(tmp_path / "grades.jsonl") == {}
    finally:
        server.shutdown()

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from aex.corpus.manifest import DocEntry
from aex.experiments.tree_rag.gold import load_questions
from aex.experiments.tree_rag.types import Page, ParsedDoc
from aex.gold.review_app import (Review, ReviewError, check_bind_address, gold_records, make_handler,
                                 parse_evidence, render_question)


def _entry(doc_id, regime, corpus, isin=""):
    return DocEntry(doc_id, corpus, regime, "Example Bank AG", isin, "brc", "en", "", "https://x.example/a.pdf",
                    "", None, "")


def _draft(draft_id, qtype="lookup", regime="A", corpus="termsheets", doc="ts-1", **over):
    base = {"draft_id": draft_id, "task": "t", "status": "draft", "corpus": corpus, "regime": regime,
            "qtype": qtype, "doc_ids": [doc], "question": "What is the barrier level of {product}?",
            "answer": "60%", "kind": "numeric", "evidence": [{"doc_id": doc, "page": 1}],
            "quotes": [{"doc_id": doc, "page": 1, "text": "The barrier level is 60%", "ok": True}]}
    return base | over


@pytest.fixture
def review(tmp_path):
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    lines = [_draft("w-1"), _draft("w-2", qtype="table"), _draft("u-1", qtype="unanswerable", answer=None,
                                                              kind="unanswerable", evidence=[], quotes=[]),
             _draft("x-rejected", status="auto_rejected")]
    (drafts / "termsheets.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (drafts / "prospectuses.jsonl").write_text(json.dumps(_draft(
        "b-1", regime="B", corpus="prospectuses", doc="sn-1",
        question="In the Example securities note, what is the minimum denomination?")) + "\n")
    (drafts / "descriptions.json").write_text(json.dumps(
        {"ts-1": {"description": "the Example barrier reverse convertible on Nestle SA", "attrs": [], "siblings": []}}))
    entries = {"ts-1": _entry("ts-1", "A", "termsheets", "CH0000000001"), "sn-1": _entry("sn-1", "B", "prospectuses")}
    docs = {d: ParsedDoc(d, d, "x", (Page(1, "The barrier level is 60% of the initial level."),), ()) for d in entries}
    return Review(drafts, tmp_path / "gold", entries, docs)


def _accept(review, draft_id, **over):
    d = review.drafts[draft_id]
    edits = {"question": d["question"], "answer": "60%", "kind": d["kind"], "qtype": d["qtype"],
             "evidence": d["evidence"]} | over
    review.decide(draft_id, "accept", edits)


def _gold(review, corpus="termsheets"):
    path = review.gold_dir / f"{corpus}.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("host,ok", [("100.84.235.48", True), ("0.0.0.0", False), ("127.0.0.1", False),
                                     ("192.168.1.5", False), ("localhost", False)])
def test_binds_only_to_tailscale(host, ok):
    if ok:
        check_bind_address(host)
    else:
        with pytest.raises(ReviewError):
            check_bind_address(host)


def test_auto_rejected_drafts_are_not_offered(review):
    assert set(review.drafts) == {"w-1", "w-2", "u-1", "b-1"}


def test_accept_writes_both_forms_of_a_regime_a_pair_as_valid_gold(review):
    _accept(review, "w-1")
    records = _gold(review)
    assert [r["form"] for r in records] == ["isin", "description"]
    assert records[0]["question"] == "What is the barrier level of the product with ISIN CH0000000001?"
    assert records[1]["question"].endswith("of the Example barrier reverse convertible on Nestle SA?")
    assert {r["pair_id"] for r in records} == {"w-1"} and all(r["verified_by"] == "daniel" for r in records)
    qs = load_questions(review.gold_dir / "termsheets.jsonl", seed=17)
    assert len({q.split for q in qs}) == 1


def test_regime_b_and_unanswerable(review):
    _accept(review, "b-1")
    (b,) = _gold(review, "prospectuses")
    assert b["qid"] == "b-1" and "pair_id" not in b
    _accept(review, "u-1", answer="", evidence=[])
    u = [r for r in _gold(review) if r["qtype"] == "unanswerable"]
    assert u and u[0]["gold"] == {"kind": "unanswerable", "value": None} and u[0]["evidence"] == []


def test_reject_never_writes_and_undo_reverts(review):
    review.decide("w-1", "reject")
    assert _gold(review) == []
    _accept(review, "w-2")
    assert len(_gold(review)) == 2
    review.decide("", "undo")
    assert _gold(review) == [] and "w-2" not in review.decisions() and "w-1" in review.decisions()


def test_invalid_accept_is_refused_and_not_recorded(review):
    with pytest.raises(ReviewError):
        _accept(review, "w-1", evidence=[])          # answerable without evidence
    with pytest.raises(ReviewError):
        _accept(review, "w-1", question="What is the barrier level?")   # lost the placeholder
    assert review.decisions() == {}


def test_next_item_balances_types_and_skip_moves_to_the_end(review):
    first = review.next_item()
    _accept(review, first["draft_id"]) if first["kind"] != "unanswerable" else _accept(
        review, first["draft_id"], evidence=[])
    second = review.next_item()
    assert (second["regime"], second["qtype"]) != (first["regime"], first["qtype"])
    review.decide(second["draft_id"], "skip")
    assert review.next_item()["draft_id"] != second["draft_id"]


def test_helpers():
    assert render_question("{product} pays what coupon?", "the X note") == "The X note pays what coupon?"
    assert parse_evidence("ts-1:3, sn-1:41") == [{"doc_id": "ts-1", "page": 3}, {"doc_id": "sn-1", "page": 41}]
    with pytest.raises(ReviewError):
        parse_evidence("ts-1")
    d = _draft("w-9", question="List the dates of {product}.", kind="date_list")
    (isin, desc) = gold_records(d, {"question": d["question"], "answer": "2027-01-01, 2027-07-01", "kind": "date_list",
                                    "evidence": d["evidence"]}, isin="CH1", description="the Y")
    assert isin["gold"]["value"] == ["2027-01-01", "2027-07-01"] and desc["question"] == "List the dates of the Y."


def test_http_flow(review, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(review, tmp_path, tmp_path / "cache"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert 'name="draft_id"' in page and review.next_item()["draft_id"] in page
        body = urllib.parse.urlencode({"draft_id": "b-1", "action": "accept", "question": review.drafts["b-1"]["question"],
                                       "answer": "CHF 1,000", "kind": "numeric", "qtype": "lookup",
                                       "evidence": "sn-1:1"}).encode()
        urllib.request.urlopen(base + "/decide", data=body)
        assert _gold(review, "prospectuses")[0]["gold"]["value"] == "CHF 1,000"
        bad = urllib.parse.urlencode({"draft_id": "w-1", "action": "accept", "question": "x {product}",
                                      "answer": "1", "kind": "numeric", "evidence": ""}).encode()
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/decide", data=bad)
        assert err.value.code == 400
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/undo")   # GET never undoes
        urllib.request.urlopen(base + "/undo", data=b"")
        assert _gold(review, "prospectuses") == []
    finally:
        server.shutdown()


def test_new_drafts_are_picked_up_without_a_restart(review):
    path = review.drafts_dir / "termsheets.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(_draft("late-1", qtype="computation")) + "\n")
    import os
    os.utime(path, ns=(1, 2))   # force a different mtime even on coarse clocks
    review.next_item()
    assert "late-1" in review.drafts

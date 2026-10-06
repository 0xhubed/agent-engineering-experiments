import json

import pytest

from aex.common.llm import MockClient
from aex.corpus.manifest import DocEntry
from aex.experiments.tree_rag.arms.base import Store
from aex.experiments.tree_rag.types import Page, ParsedDoc
from aex.gold.draft import (Drafter, DraftError, describe_all, disambiguation_drafts, parse_json,
                            prospectus_windows, quote_on_page)

TS_TEXT = ("Final Terms dated 29 September 2026 in relation to the Securities Note II dated 28 May 2026. "
           "7.00% p.a. JB Barrier Reverse Convertible on Swiss Re AG. Barrier Level 60.00% of the Initial Level. "
           "Redemption Date 6 October 2027. Market Disruption Events are set out in the Securities Note. ")


def _doc(doc_id, texts):
    return ParsedDoc(doc_id, doc_id, "sha256:x", tuple(Page(i + 1, t) for i, t in enumerate(texts)), ())


def _entry(doc_id, regime="A", corpus="termsheets", product_type="barrier_reverse_convertible", isin="CH0000000001",
           issue_date="2026-09-29", issuer="Bank Julius Baer & Co. Ltd."):
    return DocEntry(doc_id, corpus, regime, issuer, isin, product_type, "en", issue_date, "https://x.example/a.pdf",
                    "", None, "")


def _facts(barrier="60.00%", maturity="2027-10-06", under=("Swiss Re AG",), coupon="7.00%"):
    def f(value, page=1, quote="Barrier Level 60.00% of the Initial Level"):
        return {"value": value, "page": page, "quote": quote, "quote_ok": True}
    return {"underlyings": f(list(under)), "barrier_level": f(barrier), "maturity_date": f(maturity),
            "coupon_rate": f(coupon), "currency": f("CHF"), "strike_level": f(None) | {"quote_ok": False}}


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('Sure:\n```json\n{"a": [1, {"b": 2}]}\n```\nthanks') == {"a": [1, {"b": 2}]}
    with pytest.raises(DraftError):
        parse_json("no json here")


def test_quote_on_page_allows_typography_but_not_inventions():
    page = "The Barrier Level is 60.00% of the Ini-\ntial Level of the Underlying (“Swiss Re AG”)."
    assert quote_on_page("Barrier Level is 60.00% of the Initial Level", page)
    assert quote_on_page('of the Underlying ("Swiss Re AG")', page)
    assert not quote_on_page("Barrier Level is 65.00% of the Strike Level of each Underlying", page)
    assert not quote_on_page("60%", page)   # too short to prove anything


def test_descriptions_are_unique_and_disambiguation_targets_differing_fields():
    entries = {d: _entry(d, isin=f"CH{i:010d}") for i, d in enumerate(["a", "b", "c"])}
    facts = {"a": _facts(barrier="60.00%", maturity="2027-10-06"),
             "b": _facts(barrier="65.00%", maturity="2028-04-06"),
             "c": _facts(under=("Nestle SA",))}
    desc = describe_all(entries, facts)
    assert desc["c"]["description"] == "the Julius Baer barrier reverse convertible on Nestle SA"
    assert desc["c"]["siblings"] == []
    assert desc["a"]["description"].endswith("maturing in October 2027") and desc["a"]["siblings"] == ["b"]
    assert len({d["description"] for d in desc.values()}) == 3
    drafts = disambiguation_drafts(entries, facts, desc)
    assert {d["doc_ids"][0] for d in drafts} == {"a", "b"}
    a = [d for d in drafts if d["doc_ids"] == ["a"]]
    assert a[0]["field"] == "barrier_level" and a[0]["answer"] == "60.00%" and "{product}" in a[0]["question"]
    assert all(d["field"] not in desc[d["doc_ids"][0]]["attrs"] for d in drafts)   # no give-aways


def test_identical_products_get_no_description():
    entries = {d: _entry(d) for d in ("a", "b")}
    desc = describe_all(entries, {"a": _facts(), "b": _facts()})
    assert desc["a"]["description"] is None and "not unique" in desc["a"]["reason"]
    assert disambiguation_drafts(entries, {"a": _facts(), "b": _facts()}, desc) == []


def test_prospectus_windows_do_not_overlap_and_are_seeded():
    texts = [("| a | b |\n|---|---|\n| 1 | 2 |\n" if i % 7 == 0 else "") + "word " * 200 for i in range(60)]
    doc = _doc("p", texts)
    w1 = prospectus_windows(doc, 8, seed=1)
    assert w1 == prospectus_windows(doc, 8, seed=1) and len(w1) == 8
    pages = [p for w in w1 for p in w]
    assert len(pages) == len(set(pages))


def _drafter(tmp_path, responder):
    ts = _doc("ts-1", [TS_TEXT, "Other terms " * 30])
    sn = _doc("sn-1", ["Securities Note II dated 28 May 2026 " + "intro " * 50,
                       "Market Disruption Event means a suspension of trading on the exchange " + "x " * 50])
    entries = {"ts-1": _entry("ts-1"), "sn-1": _entry("sn-1", regime="B", corpus="prospectuses",
                                                      product_type="securities_note", isin="",
                                                      issue_date="2026-05-28")}
    return Drafter(MockClient(responder), Store([ts, sn]), entries, tmp_path / "drafts", prospectus_windows=1)


def _reply_for(messages):
    text = messages[0]["content"]
    if "Write up to 1 question(s) of each" in text:
        return json.dumps({"questions": [
            {"qtype": "lookup", "question": "What is the barrier level of {product}?", "answer": "60.00%",
             "kind": "numeric", "evidence": [1], "quote": "Barrier Level 60.00% of the Initial Level"},
            {"qtype": "lookup", "question": "What is the cap of {product}?", "answer": "120%", "kind": "numeric",
             "evidence": [1], "quote": "The Cap is 120% of the Initial Level"}]})
    if "does NOT answer" in text:
        return json.dumps({"questions": [{"question": "What is the cap level of {product}?", "keywords": ["cap"]}]})
    if "Does the document text below answer" in text:
        return json.dumps({"answered": False, "page": None, "quote": None})
    if "List up to 4 provisions" in text:
        return json.dumps({"provisions": [{"wording": "Market Disruption Events", "keywords": ["disruption"], "page": 1}]})
    if "needs BOTH documents" in text:
        return json.dumps({"questions": [{
            "question": "What counts as a market disruption for {product}?", "answer": "A trading suspension",
            "kind": "free", "evidence": [{"doc_id": "ts-1", "page": 1}, {"doc_id": "sn-1", "page": 2}],
            "quotes": [{"doc_id": "ts-1", "page": 1, "text": "Market Disruption Events are set out in the Securities Note"},
                       {"doc_id": "sn-1", "page": 2, "text": "Market Disruption Event means a suspension of trading"}]}]})
    return "{}"


def _lines(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_window_unanswerable_and_cross_doc_stages_write_checked_drafts_and_resume(tmp_path):
    d = _drafter(tmp_path, _reply_for)
    tasks = d.window_tasks() + d.unanswerable_tasks() + d.cross_doc_tasks()
    assert d.run(tasks, workers=2, log=lambda s: None) == 0
    ts = _lines(tmp_path / "drafts" / "termsheets.jsonl")
    by = {(r["qtype"], r["status"]): r for r in ts}
    assert by[("lookup", "draft")]["evidence"] == [{"doc_id": "ts-1", "page": 1}]
    assert "quote not found" in by[("lookup", "auto_rejected")]["reason"]   # the invented cap
    assert by[("unanswerable", "draft")]["evidence"] == [] and by[("unanswerable", "draft")]["kind"] == "unanswerable"
    cross = by[("cross_doc", "draft")]
    assert cross["doc_ids"] == ["ts-1", "sn-1"] and all(q["ok"] for q in cross["quotes"])
    assert d.linked_notes("ts-1") == ["sn-1"]
    # A second run finds every task done and writes nothing.
    before = (tmp_path / "drafts" / "termsheets.jsonl").read_text()
    assert d.run(d.window_tasks() + d.unanswerable_tasks() + d.cross_doc_tasks(), log=lambda s: None) == 0
    assert (tmp_path / "drafts" / "termsheets.jsonl").read_text() == before


def test_unanswerable_found_by_the_checker_is_auto_rejected(tmp_path):
    def reply(messages):
        if "Does the document text below answer" in messages[0]["content"]:
            return json.dumps({"answered": True, "page": 1, "quote": "Barrier Level 60.00%"})
        return _reply_for(messages)
    d = _drafter(tmp_path, reply)
    d.run(d.unanswerable_tasks(), log=lambda s: None)
    (r,) = [r for r in _lines(tmp_path / "drafts" / "termsheets.jsonl") if r["qtype"] == "unanswerable"]
    assert r["status"] == "auto_rejected" and "checker found an answer" in r["reason"]


def test_failed_tasks_are_not_marked_done(tmp_path):
    d = _drafter(tmp_path, lambda m: "not json")
    assert d.run(d.window_tasks(), log=lambda s: None) == len(d.window_tasks())
    assert d.done() == set()


@pytest.mark.parametrize("kind,answer,want", [
    ("exact", "65.00% of the initial reference price", ("numeric", "65.00%")),
    ("numeric", "CHF 1,152.00", ("numeric", "CHF 1,152.00")),
    ("free", "2027-03-30, 2027-06-29", ("date_list", ["2027-03-30", "2027-06-29"])),
    ("date_list", ["2027-03-30", "2027-06-29"], ("date_list", ["2027-03-30", "2027-06-29"])),
    ("date", "30 March 2027", ("free", "30 March 2027")),
    ("numeric", "about a third", ("free", "about a third")),
    ("exact", "Bank Julius Baer & Co. Ltd.", ("exact", "Bank Julius Baer & Co. Ltd.")),
    ("bogus", "yes", ("free", "yes")),
])
def test_normalise_answer(kind, answer, want):
    from aex.gold.draft import normalise_answer
    assert normalise_answer(kind, answer) == want


def test_focus_is_seeded_per_document():
    from aex.gold.draft import focus_for
    assert focus_for("a", 1) == focus_for("a", 1)
    assert len({focus_for(f"d{i}", 1) for i in range(10)}) > 1

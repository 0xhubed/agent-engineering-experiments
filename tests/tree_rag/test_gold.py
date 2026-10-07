import json
from pathlib import Path

import pytest

from aex.experiments.tree_rag.gold import GoldError, assign_split, load_questions
from aex.experiments.tree_rag.types import Page, ParsedDoc, TreeNode

FIXTURE = Path(__file__).parent / "fixtures" / "gold_fixture.jsonl"


def test_loads_fixture_with_typed_fields():
    qs = load_questions(FIXTURE, seed=17)
    assert [q.qid for q in qs] == ["fx-0001", "fx-0002", "fx-0003"]
    assert qs[0].evidence == (("fx-doc-1", 2),)
    assert qs[1].gold.kind == "date_list" and qs[1].gold.value == ["2026-06-15", "2026-12-15"]
    assert qs[2].gold.kind == "unanswerable" and qs[2].evidence == ()
    assert all(q.split in ("dev", "test") for q in qs)


def test_split_is_deterministic_and_roughly_proportional():
    splits = [assign_split(f"q-{i}", seed=17) for i in range(2000)]
    assert splits == [assign_split(f"q-{i}", seed=17) for i in range(2000)]
    dev_share = splits.count("dev") / len(splits)
    assert 0.17 < dev_share < 0.23
    assert splits != [assign_split(f"q-{i}", seed=18) for i in range(2000)]


def _write(tmp_path, records):
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def _base():
    return json.loads(FIXTURE.read_text().splitlines()[0])


def test_rejects_schema_violation_with_line_number(tmp_path):
    bad = _base() | {"regime": "Z"}
    with pytest.raises(GoldError, match=r"g\.jsonl:1"):
        load_questions(_write(tmp_path, [bad]), seed=1)


def test_rejects_duplicate_qid(tmp_path):
    with pytest.raises(GoldError, match="duplicate"):
        load_questions(_write(tmp_path, [_base(), _base()]), seed=1)


def test_rejects_unverified(tmp_path):
    with pytest.raises(GoldError):
        load_questions(_write(tmp_path, [_base() | {"verified_by": ""}]), seed=1)


def test_rejects_answerable_without_evidence_and_unanswerable_with_evidence(tmp_path):
    with pytest.raises(GoldError, match="evidence"):
        load_questions(_write(tmp_path, [_base() | {"evidence": []}]), seed=1)
    unans = json.loads(FIXTURE.read_text().splitlines()[2]) | {"evidence": [{"doc_id": "fx-doc-1", "page": 1}]}
    with pytest.raises(GoldError, match="evidence"):
        load_questions(_write(tmp_path, [unans]), seed=1)


def test_blank_lines_are_ignored(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps(_base()) + "\n\n")
    assert len(load_questions(p, seed=1)) == 1


def test_parsed_doc_roundtrip():
    doc = ParsedDoc("d1", "Title", "sha256:x", (Page(1, "a"), Page(2, "b")),
                    (TreeNode("n1", None, "Root", 1, 2, 1),))
    again = ParsedDoc.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert again == doc and again.n_pages == 2 and again.page(2).text == "b"


def _pair_line(qid, form, pair_id="p-0001"):
    return json.dumps({"qid": qid, "dataset": "termsheets", "regime": "A", "qtype": "lookup",
                       "question": f"What is the barrier of {qid}?", "doc_ids": ["d"],
                       "gold": {"kind": "numeric", "value": "60%"}, "evidence": [{"doc_id": "d", "page": 1}],
                       "verified_by": "daniel", "pair_id": pair_id, "form": form})


def test_pair_forms_share_a_split_and_are_typed(tmp_path):
    p = tmp_path / "g.jsonl"
    lines = [_pair_line(f"q-{i}-isin", "isin", f"p-{i}") + "\n" + _pair_line(f"q-{i}-desc", "description", f"p-{i}")
             for i in range(40)]
    p.write_text("\n".join(lines) + "\n")
    qs = load_questions(p, seed=17)
    by_pair = {}
    for q in qs:
        by_pair.setdefault(q.pair_id, set()).add(q.split)
    assert all(len(s) == 1 for s in by_pair.values())
    assert {q.form for q in qs} == {"isin", "description"}
    assert len({next(iter(s)) for s in by_pair.values()}) == 2   # both splits occur


def test_pair_id_requires_form(tmp_path):
    p = tmp_path / "g.jsonl"
    record = json.loads(_pair_line("q-1", "isin"))
    del record["form"]
    p.write_text(json.dumps(record) + "\n")
    with pytest.raises(GoldError):
        load_questions(p, seed=17)

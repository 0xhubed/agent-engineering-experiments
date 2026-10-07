import json

import jsonschema
import pytest

from aex.corpus.financebench import (COMMIT, FinanceBenchError, build_manifest, doc_id_for, gold_answer,
                                     qid_for, qtype_for, to_gold, verify_pages)
from aex.corpus.manifest import ExclusionList, load_manifest, write_manifest
from aex.experiments.tree_rag.gold import _SCHEMA, load_questions


def _q(fid="financebench_id_03029", doc="3M_2018_10K", qtype="metrics-generated",
       reasoning="Information extraction", answer="$1577.00", pages=(59,), page_text="Cash flow statement"):
    return {"financebench_id": fid, "company": "3M", "doc_name": doc, "question_type": qtype,
            "question_reasoning": reasoning, "question": "What is the FY2018 capital expenditure for 3M?",
            "answer": answer, "evidence": [{"doc_name": doc, "evidence_page_num": p, "evidence_text": "x",
                                            "evidence_text_full_page": page_text} for p in pages]}


DOC_INFO = [
    {"doc_name": "3M_2018_10K", "company": "3M", "doc_type": "10k", "doc_period": 2018,
     "doc_link": "https://investors.3m.com/x.pdf"},
    {"doc_name": "PEPSICO_2023_8K_dated-2023-05-30", "company": "PepsiCo", "doc_type": "8k", "doc_period": 2023,
     "doc_link": "https://pepsico.com/x.pdf"},
]


def test_ids_follow_the_patterns():
    assert doc_id_for("3M_2018_10K") == "fb-3m-2018-10k"
    assert doc_id_for("FOOTLOCKER_2022_8K_dated_2022-08-19") == "fb-footlocker-2022-8k-dated-2022-08-19"
    assert qid_for("financebench_id_03029") == "financebench-03029"
    with pytest.raises(FinanceBenchError):
        qid_for("fb-1")


@pytest.mark.parametrize("qtype,reasoning,want", [
    ("metrics-generated", "Information extraction", "table"),
    ("metrics-generated", "Numerical reasoning", "computation"),
    ("domain-relevant", "Logical reasoning (based on numerical reasoning) OR Logical reasoning", "computation"),
    ("domain-relevant", "Information extraction", "lookup"),
    ("novel-generated", None, "lookup"),
])
def test_qtype_mapping(qtype, reasoning, want):
    assert qtype_for(_q(qtype=qtype, reasoning=reasoning)) == want


@pytest.mark.parametrize("answer,kind", [
    ("$1577.00", "numeric"), ("4.2%", "numeric"), ("-0.02", "numeric"), ("0", "numeric"),
    ("$400,000,000 increase.", "free"), ("Yes", "free"), ("They could receive $66.56 per share.", "free"),
])
def test_numeric_only_when_the_whole_answer_is_one_number(answer, kind):
    assert gold_answer(answer)["kind"] == kind


def test_manifest_pins_the_commit_keeps_existing_checksums_and_validates(tmp_path):
    questions = [_q(), _q(fid="financebench_id_00001", doc="PEPSICO_2023_8K_dated-2023-05-30")]
    first = build_manifest(questions, DOC_INFO)
    assert [e.doc_id for e in first] == ["fb-3m-2018-10k", "fb-pepsico-2023-8k-dated-2023-05-30"]
    assert all(COMMIT in e.url and e.regime == "claim" for e in first)
    assert first[1].issue_date == "2023-05-30" and first[0].issue_date == ""
    from dataclasses import replace
    pinned = [replace(first[0], sha256="sha256:" + "a" * 64, pages=120)]
    again = build_manifest(questions, DOC_INFO, pinned)
    assert again[0].sha256 == "sha256:" + "a" * 64 and again[0].pages == 120 and again[1].sha256 == ""
    path = tmp_path / "manifest.csv"
    write_manifest(path, again)
    excl = tmp_path / "x.txt"
    excl.write_text("Forbidden Securities\n")
    assert len(load_manifest(path, excluded=ExclusionList.load(excl))) == 2


def test_unknown_document_fails():
    with pytest.raises(FinanceBenchError):
        build_manifest([_q(doc="NOPE_2020_10K")], DOC_INFO)


def test_gold_is_valid_v1_with_one_based_pages_and_drops_unavailable_docs(tmp_path):
    questions = [_q(pages=(59, 59, 60)), _q(fid="financebench_id_00001", doc="PEPSICO_2023_8K_dated-2023-05-30",
                                             qtype="novel-generated", reasoning=None, answer="Yes")]
    records = to_gold(questions, available={"fb-3m-2018-10k"})
    assert len(records) == 1
    (r,) = records
    jsonschema.validate(r, _SCHEMA)
    assert r["evidence"] == [{"doc_id": "fb-3m-2018-10k", "page": 60}, {"doc_id": "fb-3m-2018-10k", "page": 61}]
    assert r["gold"] == {"kind": "numeric", "value": "$1577.00"} and r["verified_by"] == "financebench"
    path = tmp_path / "g.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in to_gold(questions)))
    assert {q.qid for q in load_questions(path, seed=0)} == {"financebench-03029", "financebench-00001"}


def test_verify_pages_finds_the_matching_page():
    pages = ["cover", "table of contents", "Consolidated Statement of Cash Flows 2018 capex 1,577", "notes"]
    good = _q(pages=(2,), page_text="Consolidated Statement of Cash Flows 2018 capex 1,577")   # 0-based 2 -> p3
    off = _q(fid="financebench_id_00002", pages=(1,), page_text="Consolidated Statement of Cash Flows 2018 capex 1,577")
    report = verify_pages([good, off], {"fb-3m-2018-10k": pages})
    assert [(r["mapped"], r["best"]) for r in report] == [(3, 3), (2, 3)]

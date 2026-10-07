import json

import pytest

from aex.corpus.manifest import DocEntry
from aex.gold import ai_review
from aex.gold.review_app import ReviewError


@pytest.fixture
def repo(tmp_path, monkeypatch):
    drafts = tmp_path / "gold" / "drafts"
    drafts.mkdir(parents=True)
    d = {"draft_id": "w-1", "task": "t", "status": "draft", "corpus": "termsheets", "regime": "A", "qtype": "lookup",
         "doc_ids": ["ts-1"], "question": "What is the barrier of {product}?", "answer": "60%", "kind": "numeric",
         "evidence": [{"doc_id": "ts-1", "page": 1}], "quotes": []}
    (drafts / "termsheets.jsonl").write_text(json.dumps(d) + "\n")
    (drafts / "descriptions.json").write_text(json.dumps({"ts-1": {"description": "the X note", "attrs": [], "siblings": []}}))
    monkeypatch.setattr(ai_review, "DRAFTS", drafts)
    monkeypatch.setattr(ai_review, "AI_DIR", tmp_path / "gold" / "review" / "ai")
    monkeypatch.setattr(ai_review, "DECISIONS", tmp_path / "gold" / "review" / "decisions.jsonl")
    entry = DocEntry("ts-1", "termsheets", "A", "Ex AG", "CH0000000001", "brc", "en", "", "https://x/a.pdf", "", None, "")
    monkeypatch.setattr(ai_review, "entries", lambda: {"ts-1": entry})
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_decisions_need_reasons_and_merge_marks_the_model_as_verifier(repo):
    draft = ai_review.load_drafts()["w-1"]
    with pytest.raises(ReviewError):
        ai_review.decide(draft, "accept", by="r0", reason=" ")
    with pytest.raises(ReviewError):
        ai_review.decide(draft, "accept", by="r0", reason="ok", evidence="")   # answerable without evidence
    ai_review.decide(draft, "accept", by="r0", reason="barrier stated on p.1", answer="60.00%")
    assert ai_review.decided_ids() == {"w-1"}
    print(ai_review.merge())
    gold = [json.loads(ln) for ln in (repo / "gold" / "termsheets.jsonl").read_text().splitlines()]
    assert [g["form"] for g in gold] == ["isin", "description"]
    assert all(g["verified_by"] == "claude-opus-5.5" and g["gold"]["value"] == "60.00%" for g in gold)
    decision = json.loads((repo / "gold" / "review" / "decisions.jsonl").read_text())
    assert decision["verifier"] == "claude-opus-5.5" and decision["reason"]
    assert "merged 0" in ai_review.merge()   # idempotent

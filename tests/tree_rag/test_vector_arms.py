import json
from pathlib import Path

import pytest

from aex.common.accounting import Ledger
from aex.common.checkpoint import Checkpoint
from aex.common.embed import HashEmbedder, OverlapReranker
from aex.common.llm import MockClient
from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.answer import answer
from aex.experiments.tree_rag.arms import ARMS, Store, make_arm
from aex.experiments.tree_rag.run import RunConfig, evidence_metrics, run_experiment
from aex.experiments.tree_rag.types import EvidencePage, Page, ParsedDoc, Question, Retrieval

FIX = Path(__file__).parent / "fixtures"


def _doc(doc_id, pages):
    return ParsedDoc(doc_id, doc_id, f"sha256:{doc_id}", tuple(Page(i + 1, t) for i, t in enumerate(pages)), ())


def _store():
    return Store([_doc("p1", ["Product terms for the Alpha note.", "Barrier level: 60% of initial fixing.",
                              "Coupon payment dates: 15.06.2026 and 15.12.2026."]),
                  _doc("p2", ["Product terms for the Beta note.", "Barrier level: 65% of initial fixing."]),
                  _doc("bp", ["Base prospectus general conditions."])],
                 groups={"p1": "termsheets", "p2": "termsheets", "bp": "prospectuses"})


def _q(text="What is the barrier level?", docs=("p1",), evidence=(("p1", 2),), dataset="termsheets"):
    return Question("q1", dataset, "A", "lookup", text, docs, GoldAnswer("numeric", "60%"), evidence, "test")


SERVICES = {"embedder": HashEmbedder(dim=128), "reranker": OverlapReranker()}


def test_real_arms_are_registered_with_families():
    assert ARMS["chunk_embed"].family == "vector" and ARMS["hybrid_rerank"].family == "vector"
    assert ARMS["long_context"].family == "long_context"


def test_chunk_embed_searches_the_corpus_not_other_corpora_and_records_query_cost():
    arm = make_arm("chunk_embed", k=3, **SERVICES)
    stats = arm.index(_store())
    assert stats.input_tokens > 0
    r = arm.retrieve(_q(), _store())
    assert {e.doc_id for e in r.evidence} <= {"p1", "p2"}
    assert all(e.kind == "chunk" for e in r.evidence) and len(r.evidence) == 3
    assert ["p1", 2] in r.trace["pages"]          # the hashing mock is lexical, not semantic: no rank claim
    assert r.ledger.sequential_calls == 1


def test_question_docs_scope_restricts_to_the_questions_document():
    arm = make_arm("chunk_embed", k=5, scopes={"termsheets": "question_docs"}, **SERVICES)
    arm.index(_store())
    assert {e.doc_id for e in arm.retrieve(_q(), _store()).evidence} == {"p1"}


def test_hybrid_reranks_fused_candidates():
    arm = make_arm("hybrid_rerank", k=2, **SERVICES)
    arm.index(_store())
    r = arm.retrieve(_q("coupon payment dates"), _store())
    assert r.trace["pages"][0] == ["p1", 3]
    assert r.ledger.sequential_calls == 2          # query embedding + rerank


def test_arms_without_services_fail_at_index_time_with_a_clear_message():
    with pytest.raises(ValueError, match="embedder"):
        make_arm("chunk_embed").index(_store())
    with pytest.raises(ValueError, match="reranker"):
        make_arm("hybrid_rerank", embedder=HashEmbedder()).index(_store())


def test_long_context_reads_whole_doc_with_its_own_budget_or_is_not_applicable():
    arm = make_arm("long_context", max_context_words=50)
    r = arm.retrieve(_q(), _store())
    assert [e.page for e in r.evidence] == [1, 2, 3] and r.max_evidence_words == 50
    assert not r.not_applicable
    assert make_arm("long_context", max_context_words=5).retrieve(_q(), _store()).not_applicable

    seen = {}
    def respond(messages):
        seen["prompt"] = messages[-1]["content"]
        return "ANSWER: 60%"
    long_text = Retrieval([EvidencePage("p1", 1, "word " * 10_000)], {}, Ledger(), max_evidence_words=12_000)
    assert not answer(_q(), long_text, MockClient(respond), max_evidence_words=100).truncated


def test_summaries_never_count_as_retrieved_pages():
    r = Retrieval([EvidencePage("p1", 1, "summary of the terms", kind="summary", end_page=3),
                   EvidencePage("p1", 3, "coupon")], {}, Ledger())
    recall, precision, _ = evidence_metrics(_q(), r)
    assert recall == 0.0 and precision == 0.0


def test_resumed_run_keeps_the_first_index_cost(tmp_path):
    cfg = RunConfig(seed=17, gold="", parsed_dir="", checkpoint="", arms=["chunk_embed"], navigators=["m"],
                    answerers=["m"], judge="m", models={}, splits=("test",),
                    arm_options={"chunk_embed": {"cache_dir": str(tmp_path / "cache")}})
    q1, q2 = _q(), Question("q2", "termsheets", "A", "lookup", "Coupon dates?", ("p1",),
                            GoldAnswer("numeric", "1"), (("p1", 3),), "test")
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(cfg, clients={"m": MockClient(lambda m: "ANSWER: 60%", model="m")}, store=_store(),
                   questions=[q1], checkpoint=cp, services=SERVICES)
    first = cp.rows()[0]["tokens_index_amortised"]
    run_experiment(cfg, clients={"m": MockClient(lambda m: "ANSWER: 60%", model="m")}, store=_store(),
                   questions=[q1, q2], checkpoint=cp, services=SERVICES)
    assert first > 0
    assert cp.meta_get("index|chunk_embed|none")["input_tokens"] > 0
    assert [r["qid"] for r in cp.rows()] == ["q1", "q2"]


def test_detail_evidence_carries_kind_and_short_snippet(tmp_path):
    cfg = RunConfig(seed=17, gold="", parsed_dir="", checkpoint="", arms=["chunk_embed"], navigators=["m"],
                    answerers=["m"], judge="m", models={}, splits=("test",))
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(cfg, clients={"m": MockClient(lambda m: "ANSWER: 60%", model="m")}, store=_store(),
                   questions=[_q()], checkpoint=cp, services=SERVICES)
    ev = cp.rows()[0]["detail"]["evidence"][0]
    assert ev["kind"] == "chunk" and len(ev["snippet"]) <= 300


def test_cli_builds_services_and_corpus_groups(tmp_path):
    import subprocess, sys, yaml
    repo = Path(__file__).resolve().parents[2]
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for d in _store().docs.values():
        (parsed / f"{d.doc_id}.json").write_text(json.dumps(d.to_dict()))
    (parsed / "groups.json").write_text(json.dumps(_store().groups))
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"qid": "t-1", "dataset": "termsheets", "regime": "A", "qtype": "lookup",
                                "question": "What is the barrier level of the Alpha note?", "doc_ids": ["p1"],
                                "gold": {"kind": "numeric", "value": "60%"},
                                "evidence": [{"doc_id": "p1", "page": 2}], "verified_by": "test"}) + "\n")
    cfg = {"seed": 17, "gold": str(gold), "parsed_dir": str(parsed), "checkpoint": str(tmp_path / "cp.sqlite"),
           "splits": ["dev", "test"], "arms": ["hybrid_rerank"], "navigators": ["mock"], "answerers": ["mock"],
           "judge": "mock", "models": {"mock": {"kind": "mock"}},
           "services": {"embedder": {"kind": "hash", "dim": 64}, "reranker": {"kind": "overlap"}},
           "arm_options": {"hybrid_rerank": {"k": 3}}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "-m", "aex.experiments.tree_rag.run", str(tmp_path / "c.yaml")],
                   cwd=repo, check=True, capture_output=True, text=True)
    (row,) = Checkpoint(tmp_path / "cp.sqlite").rows()
    assert row["failure"] != "error" and row["arm"] == "hybrid_rerank"
    assert {e["doc_id"] for e in row["detail"]["evidence"]} <= {"p1", "p2"}   # groups loaded: bp excluded


def test_build_service_passes_reranker_templates():
    from aex.experiments.tree_rag.run import build_service
    rr = build_service("reranker", {"kind": "openai_rerank", "base_url": "http://x/v1", "model": "r",
                                    "query_template": "Q:{query}", "document_template": "D:{document}"})
    assert rr._query_template == "Q:{query}" and rr._document_template == "D:{document}"

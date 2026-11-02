import json
import subprocess
import sys
from pathlib import Path

import pytest

from aex.common.checkpoint import Checkpoint
from aex.common.llm import LLMError, MockClient
from aex.experiments.tree_rag.arms import Store
from aex.experiments.tree_rag.gold import load_questions
from aex.experiments.tree_rag.run import RunConfig, classify_failure, run_experiment
from aex.experiments.tree_rag.types import ParsedDoc

FIX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[2]


def _store():
    return Store([ParsedDoc.from_dict(json.loads((FIX / "parsed" / "fx-doc-1.json").read_text()))])


def _cfg(tmp_path, **over):
    base = dict(seed=17, gold=str(FIX / "gold_fixture.jsonl"), parsed_dir=str(FIX / "parsed"),
                checkpoint=str(tmp_path / "cp.sqlite"), arms=["oracle"], navigators=["mock"],
                answerers=["smart"], judge="smart", models={}, splits=("dev", "test"))
    return RunConfig(**(base | over))


def smart_responder(calls):
    def respond(messages):
        calls.append(1)
        prompt = messages[-1]["content"]
        if "60%" in prompt:
            return "ANSWER: 60%"
        if "15.06.2026" in prompt:
            return "ANSWER: 15.06.2026; 15.12.2026"
        return "ANSWER: NOT_STATED"
    return respond


def test_oracle_run_scores_all_fixture_questions_correct(tmp_path):
    calls = []
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(_cfg(tmp_path), clients={"smart": MockClient(smart_responder(calls), model="smart")},
                   store=_store(), questions=load_questions(FIX / "gold_fixture.jsonl", seed=17), checkpoint=cp)
    rows = {r["qid"]: r for r in cp.rows()}
    assert set(rows) == {"fx-0001", "fx-0002", "fx-0003"}
    assert all(r["correct"] is True for r in rows.values())
    assert rows["fx-0001"]["evidence_recall"] == 1.0 and rows["fx-0001"]["wrong_doc"] is False
    assert rows["fx-0003"]["evidence_recall"] is None and rows["fx-0003"]["evidence_precision"] is None
    assert rows["fx-0001"]["navigator"] == "none" and rows["fx-0001"]["answerer"] == "smart"
    assert rows["fx-0001"]["llm_calls_sequential"] == 1
    assert rows["fx-0001"]["detail"]["answer"] == "60%"
    assert len(calls) == 3


def test_rerun_skips_finished_rows(tmp_path):
    calls = []
    clients = {"smart": MockClient(smart_responder(calls), model="smart")}
    qs = load_questions(FIX / "gold_fixture.jsonl", seed=17)
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(_cfg(tmp_path), clients=clients, store=_store(), questions=qs, checkpoint=cp)
    run_experiment(_cfg(tmp_path), clients=clients, store=_store(), questions=qs, checkpoint=cp)
    assert len(calls) == 3 and len(cp) == 3


def test_endpoint_failure_records_error_row_then_retries_on_resume(tmp_path):
    def broken(messages):
        raise LLMError("HTTP 503 after retries")

    qs = load_questions(FIX / "gold_fixture.jsonl", seed=17)
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(_cfg(tmp_path), clients={"smart": MockClient(broken, model="smart")},
                   store=_store(), questions=qs, checkpoint=cp)
    rows = cp.rows()
    assert len(rows) == 3
    assert all(r["failure"] == "error" and r["correct"] is None for r in rows)
    assert "503" in rows[0]["detail"]["error"]

    calls = []
    run_experiment(_cfg(tmp_path), clients={"smart": MockClient(smart_responder(calls), model="smart")},
                   store=_store(), questions=qs, checkpoint=cp)
    assert len(cp) == 3 and all(r["correct"] is True for r in cp.rows())


def test_confident_answer_to_unanswerable_is_flagged(tmp_path):
    qs = [q for q in load_questions(FIX / "gold_fixture.jsonl", seed=17) if q.qid == "fx-0003"]
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(_cfg(tmp_path), clients={"smart": MockClient(lambda m: "ANSWER: AA-", model="smart")},
                   store=_store(), questions=qs, checkpoint=cp)
    (row,) = cp.rows()
    assert row["correct"] is False and row["failure"] == "answered_unanswerable"


def test_split_filter(tmp_path):
    qs = load_questions(FIX / "gold_fixture.jsonl", seed=17)
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(_cfg(tmp_path, splits=("test",)), clients={"smart": MockClient(lambda m: "x", model="smart")},
                   store=_store(), questions=qs, checkpoint=cp)
    assert {r["qid"] for r in cp.rows()} == {q.qid for q in qs if q.split == "test"}


@pytest.mark.parametrize("kwargs,expected", [
    (dict(correct=True, prior=None, recall=1.0, wrong_doc=False, parse_failure=False), None),
    (dict(correct=False, prior="answered_unanswerable", recall=None, wrong_doc=None, parse_failure=False), "answered_unanswerable"),
    (dict(correct=False, prior=None, recall=0.0, wrong_doc=True, parse_failure=False), "wrong_doc"),
    (dict(correct=False, prior=None, recall=0.0, wrong_doc=False, parse_failure=True), "parse_failure"),
    (dict(correct=False, prior=None, recall=0.0, wrong_doc=False, parse_failure=False), "wrong_page"),
    (dict(correct=False, prior=None, recall=None, wrong_doc=None, parse_failure=False), "wrong_page"),
    (dict(correct=False, prior=None, recall=0.5, wrong_doc=False, parse_failure=False), "right_evidence_wrong_answer"),
])
def test_classify_failure(kwargs, expected):
    assert classify_failure(**kwargs) == expected


def test_cli_smoke_run(tmp_path):
    cfg = (REPO / "configs" / "tree_rag" / "phase0-mock.yaml").read_text()
    cfg = cfg.replace("runs/phase0-mock.sqlite", str(tmp_path / "smoke.sqlite"))
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg)
    out = subprocess.run([sys.executable, "-m", "aex.experiments.tree_rag.run", str(cfg_path)],
                         cwd=REPO, capture_output=True, text=True, check=True)
    assert "oracle" in out.stdout
    assert (tmp_path / "smoke.sqlite.manifest.json").exists()


def test_cli_arms_subset_must_come_from_the_config(tmp_path):
    cfg = (REPO / "configs" / "tree_rag" / "phase0-mock.yaml").read_text()
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg.replace("runs/phase0-mock.sqlite", str(tmp_path / "smoke.sqlite")))
    run = [sys.executable, "-m", "aex.experiments.tree_rag.run", str(cfg_path), "--arms"]
    assert "oracle" in subprocess.run(run + ["oracle"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    bad = subprocess.run(run + ["raptor"], cwd=REPO, capture_output=True, text=True)
    assert bad.returncode != 0 and "arms not in the config: raptor" in bad.stderr


def test_build_client_passes_extra_body():
    from aex.experiments.tree_rag.run import build_client
    client = build_client("q", {"kind": "openai", "base_url": "http://x/v1", "model": "q",
                                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_dgx_smoke_config_loads():
    cfg = RunConfig.from_yaml(REPO / "configs" / "tree_rag" / "smoke-dgx.yaml")
    assert cfg.max_answer_tokens >= 8192
    assert cfg.models[cfg.answerers[0]]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert cfg.models[cfg.judge]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_shards_split_the_questions_without_overlap():
    from aex.experiments.tree_rag.run import in_shard
    qids = [f"q{n}" for n in range(200)]
    parts = [{q for q in qids if in_shard(q, (i, 4))} for i in range(4)]
    assert set().union(*parts) == set(qids) and sum(map(len, parts)) == 200 and all(parts)


def test_rescore_changes_only_deterministic_rows_and_keeps_the_old_verdict():
    from aex.common.scoring import GoldAnswer
    from aex.experiments.tree_rag.arms import Store
    from aex.experiments.tree_rag.rescore import rescore
    from aex.experiments.tree_rag.types import Page, ParsedDoc, Question
    store = Store([ParsedDoc("d1", "Doc", "sha", (Page(1, "text"),), ())])
    qs = {"q1": Question("q1", "ts", "A", "lookup", "Final fixing?", ("d1",), GoldAnswer("date", "2028-02-11"),
                         (("d1", 1),), "dev"),
          "q2": Question("q2", "ts", "A", "lookup", "Why?", ("d1",), GoldAnswer("free", "because"), (("d1", 1),), "dev")}
    base = {"arm": "a", "navigator": "none", "answerer": "x", "evidence_recall": 1.0, "wrong_doc": False}
    rows = [base | {"qid": "q1", "correct": False, "judge": "date", "failure": "right_evidence_wrong_answer",
                    "detail": {"raw": "ANSWER: 11. Februar 2028"}},
            base | {"qid": "q2", "correct": False, "judge": "llm", "failure": "x", "detail": {"raw": "ANSWER: no"}},
            base | {"qid": "q1", "correct": None, "judge": None, "failure": "error", "detail": {"raw": None}}]
    changed = rescore(rows, qs, store)
    assert len(changed) == 1
    row = changed[0]
    assert row["correct"] is True and row["failure"] is None
    assert row["detail"]["rescored"][0]["correct"] is False

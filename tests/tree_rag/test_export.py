import json
from pathlib import Path

import jsonschema
import pytest

from aex.common.accounting import Ledger
from aex.common.checkpoint import Checkpoint
from aex.common.llm import MockClient
from aex.experiments.tree_rag.arms import Store, register
from aex.experiments.tree_rag.export import RUN_FIELDS, ExportError, export
from aex.experiments.tree_rag.gold import load_questions
from aex.experiments.tree_rag.run import RunConfig, run_experiment
from aex.experiments.tree_rag.types import Page, ParsedDoc, Retrieval

FIX = Path(__file__).parent / "fixtures"
SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "src/aex/experiments/tree_rag/schema/tree-rag-runs.v1.schema.json").read_text())
HASH = "sha256:" + "0" * 64


@register("always_na")
class AlwaysNotApplicable:
    family = "long_context"
    uses_navigator = False

    def __init__(self, *, navigator=None, **options):
        pass

    def index(self, store):
        from aex.experiments.tree_rag.arms import IndexStats
        return IndexStats()

    def retrieve(self, q, store):
        return Retrieval([], {}, Ledger(), not_applicable=True)


def _store(long_page=False):
    doc = json.loads((FIX / "parsed" / "fx-doc-1.json").read_text())
    if long_page:
        doc["pages"][1]["text"] = "The barrier level is 60%. " + "filler " * 500
    return Store([ParsedDoc.from_dict(doc)])


def _rows(tmp_path, arms=("oracle",), responder=lambda m: "ANSWER: 60%", store=None):
    cfg = RunConfig(seed=17, gold=str(FIX / "gold_fixture.jsonl"), parsed_dir="", checkpoint="",
                    arms=list(arms), navigators=["mock"], answerers=["m"], judge="m", models={},
                    splits=("dev", "test"))
    cp = Checkpoint(tmp_path / "cp.sqlite")
    qs = load_questions(FIX / "gold_fixture.jsonl", seed=17)
    run_experiment(cfg, clients={"m": MockClient(responder, model="m")}, store=store or _store(),
                   questions=qs, checkpoint=cp)
    return cp.rows(), qs


def _export(tmp_path, rows, qs, store=None, **kw):
    return export(rows, questions=qs, store=store or _store(), models={"m": {"kind": "mock"}}, judge="m",
                  manifest_hash=HASH, data_version="2026-11-14.1", out_dir=tmp_path / "out", **kw)


def test_runs_json_matches_schema_and_strips_detail(tmp_path):
    rows, qs = _rows(tmp_path)
    data = json.loads(_export(tmp_path, rows, qs).read_text())
    jsonschema.validate(data, SCHEMA)
    assert data["schema"] == "tree-rag-runs/v1" and data["data_version"] == "2026-11-14.1"
    assert all(tuple(r) == RUN_FIELDS for r in data["rows"])
    assert {a["id"] for a in data["arms"]} == {"oracle"}
    assert data["models"] == [{"id": "m", "local": True, "roles": ["answerer", "judge"]}]


def test_not_applicable_rows_are_kept_with_null_correct(tmp_path):
    rows, qs = _rows(tmp_path, arms=("oracle", "always_na"))
    data = json.loads(_export(tmp_path, rows, qs).read_text())
    na = [r for r in data["rows"] if r["arm"] == "always_na"]
    assert len(na) == 3 and all(r["correct"] is None and r["failure"] == "not_applicable" for r in na)


def test_error_rows_block_export_unless_allowed(tmp_path):
    rows, qs = _rows(tmp_path)
    rows[0] = rows[0] | {"failure": "error", "correct": None}
    with pytest.raises(ExportError, match="error"):
        _export(tmp_path, rows, qs)
    assert _export(tmp_path, rows, qs, allow_errors=True).exists()


def test_explorer_snippets_are_short_and_never_full_pages(tmp_path):
    store = _store(long_page=True)
    rows, qs = _rows(tmp_path, store=store)
    _export(tmp_path, rows, qs, store=store)
    index = json.loads((tmp_path / "out" / "explorer" / "index.json").read_text())
    shard = json.loads((tmp_path / "out" / "explorer" / index["shards"][0]["file"]).read_text())
    q1 = next(q for q in shard["questions"] if q["qid"] == "fx-0001")
    snippet = q1["results"][0]["evidence"][0]["snippet"]
    assert len(snippet) <= 300 and snippet.startswith("The barrier level is 60%.")
    assert q1["docs"][0]["n_pages"] == 3 and q1["gold_pages"] == [["fx-doc-1", 2]]


def test_sharding(tmp_path):
    rows, qs = _rows(tmp_path)
    _export(tmp_path, rows, qs, shard_size=2)
    index = json.loads((tmp_path / "out" / "explorer" / "index.json").read_text())
    assert [len(s["qids"]) for s in index["shards"]] == [2, 1]
    assert {s["file"] for s in index["shards"]} == {"fixture-0.json", "fixture-1.json"}


def test_redacted_datasets_withhold_text_but_keep_rows(tmp_path):
    rows, qs = _rows(tmp_path)
    runs_path = _export(tmp_path, rows, qs, redact_datasets={"fixture"})
    assert len(json.loads(runs_path.read_text())["rows"]) == len(rows)
    shard = json.loads((tmp_path / "out" / "explorer" / "fixture-0.json").read_text())
    for q in shard["questions"]:
        assert q["redacted"] and q["question"] == f"fixture question {q['qid']}" and q["gold"]["value"] is None
        assert all(r["answer"] is None and all(e["snippet"] == "" for e in r["evidence"]) for r in q["results"])
    text = json.dumps(shard)
    assert all(qq.question not in text for qq in qs)


def test_financebench_is_redacted_by_default_other_datasets_are_not(tmp_path):
    rows, qs = _rows(tmp_path)
    _export(tmp_path, rows, qs)
    shard = json.loads((tmp_path / "out" / "explorer" / "fixture-0.json").read_text())
    assert not any(q.get("redacted") for q in shard["questions"])


def test_rows_carry_the_question_form(tmp_path):
    from dataclasses import replace
    rows, qs = _rows(tmp_path)
    assert all(r["form"] is None for r in rows)
    data = json.loads(_export(tmp_path, rows, qs).read_text())
    assert all(r["form"] is None for r in data["rows"])
    old = [{k: v for k, v in r.items() if k != "form"} for r in rows]            # pre-form checkpoint rows
    tagged = [r | {"form": "description"} for r in rows]
    for variant in (old, tagged):
        jsonschema.validate(json.loads(_export(tmp_path, variant, qs).read_text()), SCHEMA)
    q = replace(qs[0], pair_id="p", form="isin")
    from aex.experiments.tree_rag.run import _base_row
    assert _base_row(q, "oracle", "none", "m")["form"] == "isin"

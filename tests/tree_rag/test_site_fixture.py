import importlib.util
import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "src/aex/experiments/tree_rag/schema/tree-rag-runs.v1.schema.json").read_text())


def _load_script():
    spec = importlib.util.spec_from_file_location("make_site_fixture", ROOT / "scripts" / "make_site_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_is_valid_complete_and_deterministic(tmp_path):
    script = _load_script()
    first = json.loads(script.build_fixture(tmp_path / "a").read_text())
    second = json.loads(script.build_fixture(tmp_path / "b").read_text())
    jsonschema.validate(first, SCHEMA)
    assert first["data_version"] == "2000-01-01.0"
    assert first["rows"] == second["rows"]

    rows = first["rows"]
    assert {a["id"] for a in first["arms"]} == {"oracle", "chunk_embed", "hybrid_rerank", "raptor",
                                                "pageindex", "vec_tree", "long_context"}
    assert {r["regime"] for r in rows} == {"A", "B", "claim"}
    failures = {r["failure"] for r in rows}
    for expected in ["wrong_doc", "wrong_page", "right_evidence_wrong_answer",
                     "answered_unanswerable", "not_applicable"]:
        assert expected in failures, expected
    assert {r["navigator"] for r in rows if r["arm"] == "pageindex"} == {"qwen3.6-35b-a3b", "frontier-ref"}
    assert any((r["cost_usd"] or 0) > 0 for r in rows if r["navigator"] == "frontier-ref")
    oracle = [r for r in rows if r["arm"] == "oracle"]
    assert sum(r["correct"] for r in oracle) / len(oracle) > 0.85

    index = json.loads((tmp_path / "a" / "explorer" / "index.json").read_text())
    shard = json.loads((tmp_path / "a" / "explorer" / index["shards"][0]["file"]).read_text())
    traces = [res["trace"] for q in shard["questions"] for res in q["results"] if res["arm"] == "pageindex"]
    assert any(t.get("visited") for t in traces) and any(t.get("backtracks") for t in traces)

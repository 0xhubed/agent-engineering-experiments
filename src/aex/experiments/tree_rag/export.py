"""Turn checkpoint rows into the compact JSON the site analyses (spec §4.2).

The site recomputes every aggregate from these rows; nothing pre-aggregated
leaves this repo.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from aex.experiments.tree_rag.arms import ARMS, Store
from aex.experiments.tree_rag.types import Question

ARM_LABELS = {
    "oracle": "Oracle (gold pages)", "chunk_embed": "Chunks + embeddings",
    "hybrid_rerank": "Hybrid + reranker", "raptor": "RAPTOR", "pageindex": "PageIndex",
    "vec_tree": "Vectors → tree", "long_context": "Whole document",
    "pageindex_native": "PageIndex (its own answer)",
}
RUN_FIELDS = ("qid", "dataset", "regime", "qtype", "arm", "navigator", "answerer", "correct", "judge",
              "evidence_recall", "evidence_precision", "wrong_doc", "latency_s", "llm_calls_sequential",
              "tokens_query", "tokens_index_amortised", "cost_usd", "gpu_s", "failure")
_SCHEMA = json.loads((Path(__file__).parent / "schema" / "tree-rag-runs.v1.schema.json").read_text())


class ExportError(ValueError):
    pass


def _snippet(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _evidence_item(e, store: Store, limit: int) -> dict:
    if isinstance(e, list):   # phase-0 rows: [doc_id, page]
        doc_id, page = e
        return {"doc_id": doc_id, "page": page, "kind": "page", "snippet": _snippet(store.get(doc_id).page(page).text, limit)}
    return {"doc_id": e["doc_id"], "page": e["page"], "kind": e.get("kind", "page"), "snippet": _snippet(e["snippet"], limit)}


def _is_local(spec: dict) -> bool:
    return spec.get("local", spec.get("kind") == "mock")


def _model_meta(rows: list[dict], models: dict[str, dict], judge: str | None) -> list[dict]:
    roles: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        roles[r["answerer"]].add("answerer")
        if r["navigator"] != "none":
            roles[r["navigator"]].add("navigator")
    if judge is not None:
        roles[judge].add("judge")
    order = ["answerer", "navigator", "judge"]
    return [{"id": mid, "local": _is_local(models.get(mid, {})),
             "roles": [role for role in order if role in roles[mid]]} for mid in sorted(roles)]


def _shard_entries(rows, questions, store, snippet_chars):
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_qid[r["qid"]].append(r)
    entries = []
    for q in sorted((q for q in questions if q.qid in by_qid), key=lambda q: (q.dataset, q.qid)):
        docs = [store.get(d) for d in q.doc_ids]
        results = []
        for r in by_qid[q.qid]:
            detail = r.get("detail", {})
            evidence = [_evidence_item(e, store, snippet_chars) for e in detail.get("evidence", [])]
            results.append({"arm": r["arm"], "navigator": r["navigator"], "answerer": r["answerer"],
                            "answer": detail.get("answer"), "correct": r["correct"], "failure": r["failure"],
                            "latency_s": r["latency_s"], "tokens_query": r["tokens_query"],
                            "evidence": evidence, "trace": detail.get("trace", {})})
        entries.append({"qid": q.qid, "dataset": q.dataset, "regime": q.regime, "qtype": q.qtype,
                        "question": q.question, "gold": {"kind": q.gold.kind, "value": q.gold.value},
                        "gold_pages": [[d, p] for d, p in q.evidence],
                        "docs": [{"doc_id": d.doc_id, "title": d.title, "n_pages": d.n_pages,
                                  "tree": [asdict(n) for n in d.tree]} for d in docs],
                        "results": results})
    return entries


def export(rows: list[dict], *, questions: list[Question], store: Store, models: dict[str, dict],
           judge: str | None, manifest_hash: str, data_version: str, out_dir: str | Path, shard_size: int = 50,
           snippet_chars: int = 300, allow_errors: bool = False) -> Path:
    errors = [r for r in rows if r.get("failure") == "error"]
    if errors and not allow_errors:
        raise ExportError(f"{len(errors)} rows have failure='error'; rerun them or pass allow_errors")
    out_dir = Path(out_dir)
    arm_ids = sorted({r["arm"] for r in rows})
    runs = {
        "schema": "tree-rag-runs/v1", "data_version": data_version, "manifest_hash": manifest_hash,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arms": [{"id": a, "label": ARM_LABELS.get(a, a), "family": ARMS[a].family} for a in arm_ids],
        "models": _model_meta(rows, models, judge),
        "rows": [{k: r[k] for k in RUN_FIELDS} for r in rows],
    }
    try:
        jsonschema.validate(runs, _SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ExportError(f"runs JSON violates tree-rag-runs/v1: {exc.message}") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "tree-rag-runs.json"
    runs_path.write_text(json.dumps(runs, indent=1))

    explorer_dir = out_dir / "explorer"
    explorer_dir.mkdir(exist_ok=True)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for entry in _shard_entries(rows, questions, store, snippet_chars):
        by_dataset[entry["dataset"]].append(entry)
    shards = []
    for dataset, entries in sorted(by_dataset.items()):
        for i in range(0, len(entries), shard_size):
            chunk = entries[i:i + shard_size]
            name = f"{dataset}-{i // shard_size}.json"
            (explorer_dir / name).write_text(json.dumps({"schema": "tree-rag-explorer/v1", "questions": chunk}))
            shards.append({"file": name, "dataset": dataset, "qids": [e["qid"] for e in chunk]})
    (explorer_dir / "index.json").write_text(
        json.dumps({"schema": "tree-rag-explorer-index/v1", "shards": shards}, indent=1))
    return runs_path


def main(argv: list[str]) -> int:
    from aex.common.checkpoint import Checkpoint
    from aex.experiments.tree_rag.gold import load_questions
    from aex.experiments.tree_rag.run import RunConfig, load_store

    parser = argparse.ArgumentParser(prog="export")
    parser.add_argument("config")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args(argv[1:])
    cfg = RunConfig.from_yaml(args.config)
    manifest = json.loads(Path(f"{cfg.checkpoint}.manifest.json").read_text())
    store = load_store(cfg.parsed_dir)
    path = export(Checkpoint(cfg.checkpoint).rows(),
                  questions=load_questions(cfg.gold, seed=cfg.seed, dev_fraction=cfg.dev_fraction),
                  store=store, models=cfg.models, judge=cfg.judge, manifest_hash=manifest["manifest_hash"],
                  data_version=args.data_version, out_dir=args.out, allow_errors=args.allow_errors)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

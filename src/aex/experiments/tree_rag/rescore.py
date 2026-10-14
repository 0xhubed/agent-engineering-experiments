"""Re-apply the deterministic scorers to a checkpoint's stored answers (no model calls).

A scorer fix (e.g. reading "11. Februar 2028" or "EUR 40,00") changes no answer, only its grading, so the
rows are rescored in place instead of rerun. Rows graded by the LLM judge, error rows and not-applicable
rows are left alone; each changed row records its previous verdict under detail.rescored.

Usage: python -m aex.experiments.tree_rag.rescore configs/tree_rag/<run>.yaml [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone

from aex.common.checkpoint import Checkpoint, row_key
from aex.common.scoring import score
from aex.experiments.tree_rag.arms import Store
from aex.experiments.tree_rag.gold import load_questions
from aex.experiments.tree_rag.run import RunConfig, classify_failure, load_store
from aex.experiments.tree_rag.types import Question

DETERMINISTIC = {"exact", "numeric", "date", "date_list", "refusal"}


def rescore(rows: list[dict], questions: dict[str, Question], store: Store) -> list[dict]:
    """The rows whose verdict changes, updated (correct, judge, failure, detail.rescored)."""
    changed = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        raw = (row.get("detail") or {}).get("raw")
        if row.get("judge") not in DETERMINISTIC or raw is None or row["qid"] not in questions:
            continue
        q = questions[row["qid"]]
        result = score(raw, q.gold)
        if result.correct is None or result.correct == row["correct"]:
            continue
        parse_failure = any(not store.get(d).page(p).text.strip() for d, p in q.evidence)
        failure = classify_failure(correct=result.correct, prior=result.failure, recall=row.get("evidence_recall"),
                                   wrong_doc=row.get("wrong_doc"), parse_failure=parse_failure)
        previous = {"correct": row["correct"], "judge": row["judge"], "failure": row["failure"], "at": stamp}
        changed.append(row | {"correct": result.correct, "judge": result.method, "failure": failure,
                              "detail": row["detail"] | {"rescored": [*row["detail"].get("rescored", []), previous]}})
    return changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="rescore")
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv[1:])
    cfg = RunConfig.from_yaml(args.config)
    questions = {q.qid: q for q in load_questions(cfg.gold, seed=cfg.seed, dev_fraction=cfg.dev_fraction)}
    checkpoint = Checkpoint(cfg.checkpoint)
    changed = rescore(checkpoint.rows(), questions, load_store(cfg.parsed_dir))
    for row in changed if not args.dry_run else []:
        checkpoint.put(row_key(row["qid"], row["arm"], row["navigator"], row["answerer"]), row)
    flips = Counter((r["arm"], r["detail"]["rescored"][-1]["correct"], r["correct"]) for r in changed)
    for (arm, old, new), n in sorted(flips.items()):
        print(f"{arm:16} {old} -> {new}: {n}")
    print(f"{len(changed)} row(s) {'would change' if args.dry_run else 'rescored'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

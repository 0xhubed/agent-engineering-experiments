"""Gold quotas (spec §5.5, §11.4; plan 1b Task 8).

Usage: python -m aex.gold.report gold/termsheets.jsonl gold/prospectuses.jsonl [--seed 17]

Counts verified questions, with the two forms of a regime-A pair counted once (the pair is one question
asked two ways). Quotas, for regimes A and B:
- at least 150 test-split questions per regime;
- at least 10% unanswerable per regime;
- at least 40 questions of every type across the structured-products corpora.
Exit status 1 while any quota is unmet.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from aex.experiments.tree_rag.gold import load_questions
from aex.experiments.tree_rag.types import Question

QTYPES = ("lookup", "table", "computation", "cross_doc", "disambiguation", "unanswerable")
MIN_TEST_PER_REGIME = 150
MIN_UNANSWERABLE_SHARE = 0.10
MIN_PER_TYPE = 40
REGIMES = ("A", "B")


def unique(questions: list[Question]) -> list[Question]:
    seen: set[str] = set()
    out = []
    for q in questions:
        key = q.pair_id or q.qid
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def quota_report(questions: list[Question]) -> tuple[str, list[str]]:
    qs = [q for q in unique(questions) if q.regime in REGIMES]
    by_type = Counter((q.regime, q.qtype) for q in qs)
    lines = ["| type | " + " | ".join(f"regime {r}" for r in REGIMES) + " | total |",
             "|---|" + "---|" * (len(REGIMES) + 1)]
    for t in QTYPES:
        lines.append(f"| {t} | " + " | ".join(str(by_type[(r, t)]) for r in REGIMES)
                     + f" | {sum(by_type[(r, t)] for r in REGIMES)} |")
    failures = []
    for r in REGIMES:
        mine = [q for q in qs if q.regime == r]
        test = sum(1 for q in mine if q.split == "test")
        share = (by_type[(r, "unanswerable")] / len(mine)) if mine else 0.0
        lines.append(f"| **regime {r}** | {len(mine)} questions, {test} test, {share:.0%} unanswerable | | |")
        if test < MIN_TEST_PER_REGIME:
            failures.append(f"regime {r}: {test} test questions < {MIN_TEST_PER_REGIME}")
        if share < MIN_UNANSWERABLE_SHARE:
            failures.append(f"regime {r}: {share:.0%} unanswerable < {MIN_UNANSWERABLE_SHARE:.0%}")
    for t in QTYPES:
        n = sum(by_type[(r, t)] for r in REGIMES)
        if n < MIN_PER_TYPE:
            failures.append(f"{t}: {n} < {MIN_PER_TYPE}")
    pairs = sum(1 for q in qs if q.pair_id)
    lines.append(f"\nRegime-A pairs: {pairs} (each asked by ISIN and, where unique, by description).")
    return "\n".join(lines), failures


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gold-report")
    ap.add_argument("gold", nargs="+")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv[1:])
    questions = [q for path in args.gold for q in load_questions(path, seed=args.seed)]
    table, failures = quota_report(questions)
    print(table)
    print("\nAll quotas met." if not failures else "\nUnmet:\n" + "\n".join(f"- {f}" for f in failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

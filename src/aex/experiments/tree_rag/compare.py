"""Compare two runs row by row: the determinism check (plan 1c Task 3) and the clean-checkout reproduction (Task 5).

Usage: python -m aex.experiments.tree_rag.compare runs/a.sqlite runs/b.sqlite [--json out.json]

Rows are matched by (qid, arm, navigator, answerer); only rows present in both runs are compared. Per arm and
overall it reports how often the two runs give identical raw answer text, the same verdict (`correct`), and the
same evidence pages — the candidate reproducibility criteria.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from aex.common.checkpoint import Checkpoint, row_key

CRITERIA = ("raw", "correct", "evidence", "correct_and_evidence")


def _key(r: dict) -> str:
    return row_key(r["qid"], r["arm"], r["navigator"], r["answerer"])


def _pages(r: dict) -> list:
    return [(e["doc_id"], e["page"]) if isinstance(e, dict) else tuple(e)
            for e in (r.get("detail") or {}).get("evidence", [])]


def compare(rows_a: list[dict], rows_b: list[dict]) -> dict:
    a, b = {_key(r): r for r in rows_a}, {_key(r): r for r in rows_b}
    shared = sorted(set(a) & set(b))
    tally: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(("n", *CRITERIA), 0))
    differing: list[str] = []
    for k in shared:
        ra, rb = a[k], b[k]
        same = {"raw": (ra.get("detail") or {}).get("raw") == (rb.get("detail") or {}).get("raw"),
                "correct": ra["correct"] == rb["correct"], "evidence": _pages(ra) == _pages(rb)}
        same["correct_and_evidence"] = same["correct"] and same["evidence"]
        for bucket in (ra["arm"], "all"):
            tally[bucket]["n"] += 1
            for c in CRITERIA:
                tally[bucket][c] += same[c]
        if not same["correct_and_evidence"]:
            differing.append(k)
    rates = {arm: {"n": t["n"], **{c: round(t[c] / t["n"], 4) for c in CRITERIA}}
             for arm, t in sorted(tally.items()) if t["n"]}
    return {"shared_rows": len(shared), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
            "rates": rates, "differing": differing}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="compare")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    args = ap.parse_args(argv[1:])
    report = compare(Checkpoint(args.a).rows(), Checkpoint(args.b).rows())
    print(f"shared rows {report['shared_rows']} (only in a: {report['only_a']}, only in b: {report['only_b']})")
    print(f"{'arm':22} {'n':>5} " + " ".join(f"{c:>21}" for c in CRITERIA))
    for arm, r in report["rates"].items():
        print(f"{arm:22} {r['n']:5} " + " ".join(f"{r[c]:21.1%}" for c in CRITERIA))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Three near-duplicate synthetic termsheets + six verified questions, parsed with the shared parser.

Usage: uv run python scripts/make_live_fixture.py data/live
Writes <out>/pdfs/*.pdf, <out>/parsed/*.json (+ groups.json) and <out>/gold.jsonl. All content is fictional.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aex.experiments.tree_rag.parse import load_or_parse
from aex.experiments.tree_rag.synthetic_pdf import make_termsheet_pdf

PRODUCTS = {"ts-alpha": ("Alpha Nestle reverse convertible", "60%", "4.0"),
            "ts-beta": ("Beta Nestle reverse convertible", "65%", "5.0"),
            "ts-gamma": ("Gamma Nestle reverse convertible", "70%", "3.5")}
QUESTIONS = [
    ("live-lookup", "lookup", "What is the barrier level of the Alpha Nestle reverse convertible?",
     ["ts-alpha"], {"kind": "numeric", "value": "60%"}, [("ts-alpha", 3)]),
    ("live-table", "table", "List the coupon payment dates of the Alpha Nestle reverse convertible.",
     ["ts-alpha"], {"kind": "date_list", "value": ["2026-06-15", "2026-12-15"]}, [("ts-alpha", 5)]),
    ("live-computation", "computation",
     "What is the semi-annual coupon amount in CHF on one Alpha Nestle reverse convertible?",
     ["ts-alpha"], {"kind": "numeric", "value": "112.5"}, [("ts-alpha", 1), ("ts-alpha", 5)]),
    ("live-disambiguation-1", "disambiguation",
     "How many shares per note are delivered after a barrier event for the Beta Nestle reverse convertible?",
     ["ts-beta"], {"kind": "numeric", "value": "5.0"}, [("ts-beta", 7)]),
    ("live-disambiguation-2", "disambiguation", "What is the barrier level of the Gamma Nestle reverse convertible?",
     ["ts-gamma"], {"kind": "numeric", "value": "70%"}, [("ts-gamma", 3)]),
    ("live-unanswerable", "unanswerable", "What is the issuer's credit rating for the Alpha Nestle reverse convertible?",
     ["ts-alpha"], {"kind": "unanswerable", "value": None}, []),
]


def main(out: Path) -> None:
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    for doc_id, (product, barrier, shares) in PRODUCTS.items():
        pdf = out / "pdfs" / f"{doc_id}.pdf"
        if not pdf.exists():
            make_termsheet_pdf(pdf, product=product, barrier=barrier, shares=shares)
        load_or_parse(pdf, doc_id, out / "parsed")
    (out / "parsed" / "groups.json").write_text(json.dumps({d: "live-termsheets" for d in PRODUCTS}))
    with open(out / "gold.jsonl", "w") as fh:
        for qid, qtype, text, docs, gold, evidence in QUESTIONS:
            fh.write(json.dumps({"qid": qid, "dataset": "live-termsheets", "regime": "A", "qtype": qtype,
                                 "question": text, "doc_ids": docs, "gold": gold,
                                 "evidence": [{"doc_id": d, "page": p} for d, p in evidence],
                                 "verified_by": "fixture"}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

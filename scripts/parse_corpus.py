"""Parse every downloaded corpus document once with the shared parser; write data/parsed/groups.json.

Usage: uv run python scripts/parse_corpus.py corpus/*/manifest.csv [--data data]
Resumable: load_or_parse reuses a cached parse while the PDF's sha256 is unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from aex.corpus.manifest import ExclusionList, load_manifest
from aex.experiments.tree_rag.parse import load_or_parse


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--data", default="data")
    args = ap.parse_args(argv[1:])
    excluded = ExclusionList.load()
    data, parsed = Path(args.data), Path(args.data) / "parsed"
    groups_file = parsed / "groups.json"
    groups = json.loads(groups_file.read_text()) if groups_file.exists() else {}
    entries = [e for m in args.manifests for e in load_manifest(m, excluded=excluded)]
    for i, e in enumerate(sorted(entries, key=lambda e: e.pages or 0), 1):   # small documents first
        pdf = data / e.corpus / f"{e.doc_id}.pdf"
        if not pdf.exists():
            print(f"[{i}/{len(entries)}] {e.doc_id}: missing {pdf}, skipped", flush=True)
            continue
        started = time.time()
        doc = load_or_parse(pdf, e.doc_id, parsed)
        groups[e.doc_id] = e.corpus
        parsed.mkdir(parents=True, exist_ok=True)
        groups_file.write_text(json.dumps(groups, indent=1, sort_keys=True))
        empty = sum(1 for p in doc.pages if not p.text.strip())
        print(f"[{i}/{len(entries)}] {e.doc_id}: {doc.n_pages} pp, {len(doc.tree)} tree nodes, {empty} empty pages, "
              f"{time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

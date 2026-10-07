"""FinanceBench open-150: the claim-check slice (spec §5.3).

Usage:
  python -m aex.corpus.financebench manifest   # source JSONL -> corpus/financebench/manifest.csv
  python -m aex.corpus.fetch corpus/financebench/manifest.csv
  python -m aex.corpus.financebench verify     # evidence page numbers vs the PDF text layer
  python -m aex.corpus.financebench gold       # -> gold/financebench.jsonl (gold.v1)

Everything comes from the dataset's GitHub repository pinned at one commit: the question file, the
document information file (both sha256-pinned below) and the PDFs (pinned in the manifest by the
fetcher). The gold file is not committed: it is regenerated from the pinned source, and FinanceBench
is CC-BY-NC, so this public repo carries only URLs, checksums and code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from aex.corpus.manifest import DocEntry, ExclusionList, ManifestError, load_manifest, write_manifest

DATASET = "financebench"
CORPUS = "financebench"
REPO = "patronus-ai/financebench"
COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
SOURCES = {
    "financebench_open_source.jsonl":
        "sha256:a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
    "financebench_document_information.jsonl":
        "sha256:1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575",
}
LICENCE_NOTE = f"FinanceBench (CC-BY-NC-4.0), {REPO}@{COMMIT[:7]}; URL + sha256 only, never redistributed"
# The dataset's evidence_page_num is 0-based; our pages are 1-based. Checked against the PDF text
# layer for every evidence item with `verify` (see corpus/financebench/README.md).
PAGE_OFFSET = 1

_NUMERIC = re.compile(r"[-−]?\$?[-−]?\d[\d,]*(\.\d+)?%?")
_DATED = re.compile(r"dated[-_](\d{4}-\d{2}-\d{2})$")


class FinanceBenchError(ValueError):
    pass


def doc_id_for(doc_name: str) -> str:
    """'3M_2018_10K' -> 'fb-3m-2018-10k' (manifest doc_id pattern)."""
    return "fb-" + re.sub(r"[^a-z0-9]+", "-", doc_name.lower()).strip("-")


def qid_for(financebench_id: str) -> str:
    """'financebench_id_03029' -> 'financebench-03029'."""
    match = re.fullmatch(r"financebench_id_(\d+)", financebench_id)
    if not match:
        raise FinanceBenchError(f"unexpected financebench_id {financebench_id!r}")
    return f"financebench-{match.group(1)}"


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def check_source(path: str | Path) -> None:
    path = Path(path)
    want = SOURCES[path.name]
    got = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise FinanceBenchError(f"{path.name}: sha256 {got} does not match the pinned {want}")


def build_manifest(questions: list[dict], doc_info: list[dict],
                   existing: list[DocEntry] = ()) -> list[DocEntry]:
    """One row per document the questions use; checksums and page counts already pinned are kept."""
    info = {d["doc_name"]: d for d in doc_info}
    pinned = {e.doc_id: e for e in existing}
    rows = []
    for name in sorted({q["doc_name"] for q in questions}):
        if name not in info:
            raise FinanceBenchError(f"{name}: not in the document information file")
        d, doc_id = info[name], doc_id_for(name)
        dated = _DATED.search(name)
        old = pinned.get(doc_id)
        rows.append(DocEntry(
            doc_id=doc_id, corpus=CORPUS, regime="claim", issuer=d["company"], isin="",
            product_type=d["doc_type"].lower(), language="en", issue_date=dated.group(1) if dated else "",
            url=f"{RAW}/pdfs/{name}.pdf", sha256=old.sha256 if old else "", pages=old.pages if old else None,
            licence_note=LICENCE_NOTE))
    return rows


def qtype_for(q: dict) -> str:
    """Map FinanceBench's question type and reasoning label onto ours.

    Any question whose reasoning label mentions numerical reasoning needs arithmetic → computation.
    A metrics question that is pure extraction reads one line of a financial statement → table.
    Everything else (domain-relevant, novel-generated) → lookup.
    """
    reasoning = (q.get("question_reasoning") or "").lower()
    if "numerical reasoning" in reasoning:
        return "computation"
    if q["question_type"] == "metrics-generated":
        return "table"
    return "lookup"


def gold_answer(answer: str) -> dict:
    text = answer.strip()
    if _NUMERIC.fullmatch(text):
        return {"kind": "numeric", "value": text.replace("−", "-")}
    return {"kind": "free", "value": text}


def to_gold(questions: list[dict], *, available: set[str] | None = None) -> list[dict]:
    """gold.v1 records. Questions on documents outside `available` (e.g. excluded by the fetcher's
    text screen) are dropped."""
    records = []
    for q in sorted(questions, key=lambda q: q["financebench_id"]):
        doc_id = doc_id_for(q["doc_name"])
        if available is not None and doc_id not in available:
            continue
        pages = sorted({e["evidence_page_num"] + PAGE_OFFSET for e in q["evidence"]})
        if any(e["doc_name"] != q["doc_name"] for e in q["evidence"]):
            raise FinanceBenchError(f"{q['financebench_id']}: evidence from another document")
        records.append({
            "qid": qid_for(q["financebench_id"]), "dataset": DATASET, "regime": "claim", "qtype": qtype_for(q),
            "question": q["question"].strip(), "doc_ids": [doc_id], "gold": gold_answer(q["answer"]),
            "evidence": [{"doc_id": doc_id, "page": p} for p in pages], "verified_by": DATASET,
        })
    return records


def _similarity(a: str, b: str) -> float:
    norm = lambda s: " ".join(s.split()).casefold()
    return SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()


def verify_pages(questions: list[dict], page_texts: dict[str, list[str]], *, window: int = 2) -> list[dict]:
    """For every evidence item, find which PDF page its `evidence_text_full_page` best matches
    (looking `window` pages either side of the mapped page). Returns one report row per item."""
    report = []
    for q in questions:
        doc_id = doc_id_for(q["doc_name"])
        pages = page_texts.get(doc_id)
        if pages is None:
            continue
        for e in q["evidence"]:
            mapped = e["evidence_page_num"] + PAGE_OFFSET
            lo, hi = max(1, mapped - window), min(len(pages), mapped + window)
            scores = {p: _similarity(e["evidence_text_full_page"], pages[p - 1]) for p in range(lo, hi + 1)}
            best = max(scores, key=scores.get) if scores else None
            report.append({"qid": qid_for(q["financebench_id"]), "doc_id": doc_id, "mapped": mapped,
                           "best": best, "best_score": scores.get(best, 0.0) if best else 0.0,
                           "mapped_score": scores.get(mapped, 0.0)})
    return report


def _pdf_pages(path: Path) -> list[str]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    finally:
        pdf.close()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="financebench")
    ap.add_argument("command", choices=["sources", "manifest", "verify", "gold"])
    ap.add_argument("--src", default="data/financebench-src")
    ap.add_argument("--data", default="data")
    ap.add_argument("--manifest", default="corpus/financebench/manifest.csv")
    ap.add_argument("--out", default="gold/financebench.jsonl")
    args = ap.parse_args(argv[1:])
    src = Path(args.src)

    if args.command == "sources":
        from aex.corpus.fetch import _Polite
        client = _Polite(None, 2.0)
        src.mkdir(parents=True, exist_ok=True)
        for name in SOURCES:
            response = client.get(f"{RAW}/data/{name}")
            response.raise_for_status()
            (src / name).write_bytes(response.content)
            check_source(src / name)
        print(f"downloaded and verified {len(SOURCES)} source files into {src}")
        return 0

    for name in SOURCES:
        check_source(src / name)
    questions = read_jsonl(src / "financebench_open_source.jsonl")
    excluded = ExclusionList.load()
    manifest = Path(args.manifest)

    if args.command == "manifest":
        existing = load_manifest(manifest, excluded=excluded) if manifest.exists() else []
        rows = build_manifest(questions, read_jsonl(src / "financebench_document_information.jsonl"), existing)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest, rows)
        load_manifest(manifest, excluded=excluded)   # validate what we wrote
        print(f"wrote {len(rows)} documents to {manifest}")
        return 0

    entries = load_manifest(manifest, excluded=excluded)
    on_disk = {e.doc_id for e in entries if e.sha256 and (Path(args.data) / CORPUS / f"{e.doc_id}.pdf").exists()}

    if args.command == "verify":
        texts = {d: _pdf_pages(Path(args.data) / CORPUS / f"{d}.pdf") for d in sorted(on_disk)}
        report = verify_pages(questions, texts)
        off = [r for r in report if r["best"] != r["mapped"]]
        for r in off:
            print(f"{r['qid']} {r['doc_id']}: mapped p{r['mapped']} ({r['mapped_score']:.2f}), "
                  f"best p{r['best']} ({r['best_score']:.2f})")
        print(f"{len(report)} evidence items checked, {len(report) - len(off)} on the mapped page, {len(off)} elsewhere")
        return 0 if not off else 1

    records = to_gold(questions, available=on_disk)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    from aex.experiments.tree_rag.gold import load_questions
    load_questions(out, seed=0)   # validate against gold.v1
    print(f"wrote {len(records)} questions to {out} ({len(questions) - len(records)} dropped: document unavailable)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (FinanceBenchError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

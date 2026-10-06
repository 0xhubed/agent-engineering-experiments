# FinanceBench open-150 (claim check)

The 150 open-source questions of [FinanceBench](https://github.com/patronus-ai/financebench) (CC-BY-NC-4.0),
pinned at commit `cc39aeb`, over the 84 filings they use (64 10-K, 8 10-Q, 6 8-K, 6 earnings releases;
12,013 pages). `manifest.csv` holds URLs into that commit plus sha256; questions and PDFs are never committed.

```bash
uv run python -m aex.corpus.financebench sources    # question + document files, sha256-checked
uv run python -m aex.corpus.financebench manifest   # rebuilds manifest.csv, keeps pinned checksums
uv run python -m aex.corpus.fetch corpus/financebench/manifest.csv
uv run python -m aex.corpus.financebench verify     # evidence pages vs the PDF text layer
uv run python -m aex.corpus.financebench gold       # -> gold/financebench.jsonl (gitignored)
```

**Mapping to `gold.v1`**
- `qtype`: reasoning label mentions numerical reasoning → `computation` (67); metrics question that is pure
  extraction from a statement → `table` (14); everything else → `lookup` (69).
- `gold.kind`: `numeric` when the whole answer is one number (`$1577.00`, `4.2%`, `-0.02`; 52), else `free`
  (LLM judge; 98).
- Evidence pages: the dataset's `evidence_page_num` is **0-based**. Checked for all 189 evidence items: the
  dataset's full-page text matches PDF page `evidence_page_num + 1` (similarity 0.95–1.00, median 0.998),
  never a neighbour.
- `verified_by: financebench`; dev/test split by the usual hash. No document failed the exclusion screen.

**Explorer:** question text, answers and snippets are redacted in the public explorer until the licence
sign-off (decision D3); aggregates are unaffected (`export(..., redact_datasets=...)`).

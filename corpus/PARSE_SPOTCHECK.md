# Parse spot-check — 2026-10-06

Shared parser: docling, OCR off (`PARSER_ID = docling-no-ocr-v1`). Corpus: 78 termsheets + 9 prospectuses, 87 documents,
~7,200 pages, parsed in ~1.6 h on the ThinkCentre CPU.

## Text fidelity (automated, 20 documents: 5 JB, 5 RBI, 4 DB, 6 prospectuses; seed 17)
Per page, word-multiset overlap between our parsed text and the PDF's own text layer (pages with < 15 words skipped).

| Group | Median page overlap | Pages < 0.8 |
|---|---|---|
| Julius Bär final terms | 0.99 | 0–1 per document |
| RBI final terms | 1.00 | 0–1 |
| Deutsche Bank final terms | 0.99–1.00 | 0 |
| Prospectuses (BCV, BNPP, JB SN II / RD II, RBI SSP) | 0.96–1.00 | 4–25 of 190–1,709 |

## Tables (by eye)
JB final terms p. 28 (issuer financials): every cell of both tables matches the page image, headings and footnotes intact.

## Empty pages
18 pages across all documents parse as empty. 14 hold ≤ 12 words in the PDF itself (blank / "left blank" pages).
The rest are text inside graphics (BCV pp. 312, 319: an org chart and a contact page in the appended annual report),
which the parser does not extract. None is plausible evidence; left as is.

## Flat heading trees (fixed by `TREE_ID = outline-or-numbering-v1`)
Docling gives every section header level 1, so our `ParsedDoc.tree` is flat (e.g. BNPP: 3,680 sibling sections).
Only 3/9 prospectuses were thought to carry a usable PDF outline (BCV 227 entries / 5 levels; RBI SSP 2025SN and 2026RD, 179 / 3);
no termsheet does. No arm's retrieval depends on our tree (PageIndex builds its own; RAPTOR clusters chunks; vector
arms ignore it; vec_tree's document card uses titles only). It feeds the signature visual and the mapping of
PageIndex reads onto sections. Fix before publishing: build the tree from the PDF outline when present, otherwise
relevel docling headings by their numbering ("4.2.1" -> level 3; BNPP: 587 numbered headings).

**Fix (2026-10-06).** The tree now comes from the PDF outline when it has ≥ 10 entries (4/9 prospectuses — JB's
Registration Document II also has one — and 14/84 FinanceBench filings; no termsheet), otherwise from docling's headings
re-levelled by numbering scheme (`parse.relevel`: Part/Annex > Roman > ALL CAPS > Item/Article/Section > A. >
4.2.1 (one level per component) > § > Note > (a); a plain heading sits below the last numbered one). Cached text
is reused; only the tree is rebuilt. Measured against the outlines where both exist (docling headings matched to
outline entries by title and page ±1; does each step between consecutive headings go up/down/stay as in the outline?):

| | steps where the outline changes level | steps where it stays |
|---|---|---|
| prospectuses (4 docs, 632 steps) | 77% right (flat tree: 0%) | 96% (flat: 100%) |
| FinanceBench filings (13 parsed so far, 1,269 steps) | 43% right (flat: 0%) | 92% (flat: 100%) |

10-K headings are mostly unnumbered, so their trees stay rough; FinanceBench is the claim check, where PageIndex
builds its own trees anyway.

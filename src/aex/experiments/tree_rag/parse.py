"""The single PDF parse every arm shares (spec §4.1).

Arms must never parse documents themselves: a parser difference would look
like a retrieval difference.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable

from aex.experiments.tree_rag.types import Page, ParsedDoc, TreeNode

Heading = tuple[str, int, int]

# Identifies the parse configuration; cached parses from any other configuration are ignored.
# OCR is off: the corpus PDFs are born-digital, and on a 31-page termsheet OCR took 3.5x longer for the
# same text (20,751 vs 20,746 words; median per-page word-sequence similarity 1.000). Pages without a
# text layer then come out empty and are reported as the parse_failure class, never silently filled.
PARSER_ID = "docling-no-ocr-v1"
# Identifies how the heading tree is built from docling's headings (which all come out at level 1) and the
# PDF outline. A tree-only change: cached text is reused and only the tree is rebuilt.
TREE_ID = "outline-or-numbering-v2"
MIN_OUTLINE_ENTRIES = 10
MAX_LEVEL = 6

# Heading numbering schemes, highest in the hierarchy first. A document's levels follow the order of the
# schemes it actually uses; "4.2.1"-style numbers add one level per component.
_SCHEMES: list[tuple[str, re.Pattern]] = [
    ("part", re.compile(r"^(part|teil)\s+([ivx]+|\d+|[a-z])\b", re.IGNORECASE)),
    ("annex", re.compile(r"^(annex|anhang|appendix|schedule)\b", re.IGNORECASE)),
    ("roman", re.compile(r"^[IVX]+\.\s")),
    ("caps", re.compile(r"^(?=[^a-zäöü]*$)(?=.*[A-ZÄÖÜ]{3}.*\s.*[A-ZÄÖÜ]{3})")),
    ("item", re.compile(r"^(item|article|artikel|section|abschnitt)\s+[\divx]+", re.IGNORECASE)),
    ("letter", re.compile(r"^[A-Z]\.\s")),
    ("arabic", re.compile(r"^\d+(\.\d+)*\.?\s")),
    ("para", re.compile(r"^§\s*\d+")),
    ("note", re.compile(r"^note\s+\d+", re.IGNORECASE)),
    ("paren", re.compile(r"^\(([a-z]|[ivx]+|\d+)\)\s")),
]


def build_tree(headings: list[Heading], n_pages: int, doc_title: str) -> tuple[TreeNode, ...]:
    if not headings:
        return (TreeNode("n0", None, doc_title, 1, n_pages, 0),)
    nodes: list[TreeNode] = []
    stack: list[tuple[int, str]] = []          # (level, node id) of open ancestors
    for i, (title, level, page) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        node_id = f"n{i + 1}"
        end = n_pages
        for later_title, later_level, later_page in headings[i + 1:]:
            if later_level <= level:
                end = max(page, later_page)
                break
        nodes.append(TreeNode(node_id, parent, title, page, end, level))
        stack.append((level, node_id))
    return tuple(nodes)


def heading_scheme(title: str) -> tuple[str, int] | None:
    """(scheme, extra depth) of a numbered or all-caps heading; None for a plain or placeholder heading."""
    t = title.strip()
    if not t or t.startswith("["):
        return None
    for name, pattern in _SCHEMES:
        if pattern.match(t):
            depth = len(re.match(r"^(\d+(?:\.\d+)*)", t).group(1).split(".")) - 1 if name == "arabic" else 0
            return name, depth
    return None


def relevel(headings: list[Heading]) -> list[Heading]:
    """Levels from numbering: schemes ranked by _SCHEMES among those the document uses (Parts move below
    Roman sections when a Roman heading comes first); a plain heading sits one level below the last
    numbered one."""
    schemes = [heading_scheme(title) for title, _, _ in headings]
    used = [name for name, _ in _SCHEMES if any(s and s[0] == name for s in schemes)]
    first = {name: next(i for i, s in enumerate(schemes) if s and s[0] == name) for name in used}
    if "part" in first and "roman" in first and first["roman"] < first["part"]:
        # "VI. Terms and Conditions … Part A": this document's parts sit inside its Roman sections.
        used.remove("part")
        used.insert(used.index("roman") + 1, "part")
    rank = {name: i + 1 for i, name in enumerate(used)}
    out: list[Heading] = []
    last = 0
    for (title, _, page), scheme in zip(headings, schemes):
        if scheme is None:
            level = min(last + 1, MAX_LEVEL)
        else:
            level = last = min(rank[scheme[0]] + scheme[1], MAX_LEVEL)
        out.append((title, level, page))
    return out


def outline_headings(pdf_path: str | Path) -> list[Heading] | None:
    """The PDF's own outline (bookmarks) as headings, when it is substantial enough to trust."""
    import pypdfium2 as pdfium
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except pdfium.PdfiumError:
        return None
    try:
        entries: list[Heading] = []
        for mark in pdf.get_toc(max_depth=MAX_LEVEL):
            dest, title = mark.get_dest(), " ".join((mark.get_title() or "").split())
            index = dest.get_index() if dest is not None else None
            if title and index is not None:
                entries.append((title, mark.level + 1, index + 1))
    finally:
        pdf.close()
    return entries if len(entries) >= MIN_OUTLINE_ENTRIES else None


def structure(doc: ParsedDoc, headings: list[Heading], pdf_path: str | Path | None) -> ParsedDoc:
    """The document's tree from its PDF outline if it has one, else from docling's headings re-levelled."""
    outline = outline_headings(pdf_path) if pdf_path is not None else None
    chosen = outline if outline is not None else relevel(headings)
    return replace(doc, tree=build_tree(chosen, doc.n_pages, doc.title))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def parse_pdf(path: str | Path, doc_id: str) -> ParsedDoc:
    """Docling adapter. If docling's API differs from this, adapt the adapter only —
    the build_tree contract and ParsedDoc shape are what the rest of the code relies on."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import SectionHeaderItem, TableItem, TextItem, TitleItem

    path = Path(path)
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(do_ocr=False))})
    document = converter.convert(str(path)).document
    n_pages = len(document.pages)
    texts: dict[int, list[str]] = {n: [] for n in range(1, n_pages + 1)}
    headings: list[Heading] = []
    title = path.stem
    for item, _depth in document.iterate_items():
        provenance = getattr(item, "prov", None)
        if not provenance:
            continue
        page = provenance[0].page_no
        if isinstance(item, TableItem):
            texts[page].append(item.export_to_markdown(doc=document))
            continue
        if not isinstance(item, TextItem):
            continue
        texts[page].append(item.text)
        if isinstance(item, TitleItem):
            title = item.text
        elif isinstance(item, SectionHeaderItem):
            headings.append((item.text, max(1, int(item.level)), page))
    pages = tuple(Page(n, "\n".join(texts[n])) for n in range(1, n_pages + 1))
    return ParsedDoc(doc_id, title, file_sha256(path), pages, build_tree(headings, n_pages, title))


def load_or_parse(pdf_path: str | Path, doc_id: str, cache_dir: str | Path, *,
                  parser: Callable[[Path, str], ParsedDoc] = parse_pdf) -> ParsedDoc:
    """Cached parse. The cache keeps docling's raw headings, so a TREE_ID change rebuilds only the tree."""
    pdf_path, cache_dir = Path(pdf_path), Path(cache_dir)
    cache_file = cache_dir / f"{doc_id}.json"
    current_sha = file_sha256(pdf_path)
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        cached = ParsedDoc.from_dict(data)
        if cached.sha256 == current_sha and data.get("parser") == PARSER_ID:
            if data.get("tree_id") == TREE_ID:
                return cached
            # Older caches hold docling's headings as the tree itself.
            headings = [tuple(h) for h in data.get("headings") or
                        [(n.title, n.level, n.start_page) for n in cached.tree if n.level > 0]]
            return _write_cache(cache_file, structure(cached, headings, pdf_path), headings)
    raw = parser(pdf_path, doc_id)
    headings = [(n.title, n.level, n.start_page) for n in raw.tree if n.level > 0]
    return _write_cache(cache_file, structure(raw, headings, pdf_path), headings)


def _write_cache(cache_file: Path, doc: ParsedDoc, headings: list[Heading]) -> ParsedDoc:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(doc.to_dict() | {"parser": PARSER_ID, "tree_id": TREE_ID,
                                                      "headings": [list(h) for h in headings]}))
    return doc

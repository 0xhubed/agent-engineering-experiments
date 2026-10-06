"""The single PDF parse every arm shares (spec §4.1).

Arms must never parse documents themselves: a parser difference would look
like a retrieval difference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from aex.experiments.tree_rag.types import Page, ParsedDoc, TreeNode

Heading = tuple[str, int, int]

# Identifies the parse configuration; cached parses from any other configuration are ignored.
# OCR is off: the corpus PDFs are born-digital, and on a 31-page termsheet OCR took 3.5x longer for the
# same text (20,751 vs 20,746 words; median per-page word-sequence similarity 1.000). Pages without a
# text layer then come out empty and are reported as the parse_failure class, never silently filled.
PARSER_ID = "docling-no-ocr-v1"


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
    pdf_path, cache_dir = Path(pdf_path), Path(cache_dir)
    cache_file = cache_dir / f"{doc_id}.json"
    current_sha = file_sha256(pdf_path)
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        cached = ParsedDoc.from_dict(data)
        if cached.sha256 == current_sha and data.get("parser") == PARSER_ID:
            return cached
    doc = parser(pdf_path, doc_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(doc.to_dict() | {"parser": PARSER_ID}))
    return doc

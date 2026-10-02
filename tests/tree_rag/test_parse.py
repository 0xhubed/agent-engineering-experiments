import json

import pytest

from aex.experiments.tree_rag.parse import build_tree, file_sha256, load_or_parse
from aex.experiments.tree_rag.types import Page, ParsedDoc


def test_no_headings_gives_single_root():
    tree = build_tree([], n_pages=4, doc_title="Termsheet")
    assert len(tree) == 1
    root = tree[0]
    assert (root.id, root.parent, root.title, root.start_page, root.end_page, root.level) == \
        ("n0", None, "Termsheet", 1, 4, 0)


def test_nesting_and_page_ranges():
    headings = [("1 Terms", 1, 1), ("1.1 Barrier", 2, 2), ("1.2 Coupon", 2, 3), ("2 Risks", 1, 5)]
    tree = {n.title: n for n in build_tree(headings, n_pages=6, doc_title="x")}
    assert tree["1 Terms"].parent is None and (tree["1 Terms"].start_page, tree["1 Terms"].end_page) == (1, 5)
    assert tree["1.1 Barrier"].parent == tree["1 Terms"].id
    assert (tree["1.1 Barrier"].start_page, tree["1.1 Barrier"].end_page) == (2, 3)
    assert (tree["1.2 Coupon"].start_page, tree["1.2 Coupon"].end_page) == (3, 5)
    assert (tree["2 Risks"].start_page, tree["2 Risks"].end_page) == (5, 6)


def test_skipped_levels_attach_to_nearest_shallower_heading():
    tree = build_tree([("A", 1, 1), ("A.x.y", 3, 1), ("B", 1, 2)], n_pages=2, doc_title="x")
    by_title = {n.title: n for n in tree}
    assert by_title["A.x.y"].parent == by_title["A"].id


def test_two_headings_on_same_page():
    tree = build_tree([("A", 1, 3), ("B", 1, 3)], n_pages=3, doc_title="x")
    assert [(n.start_page, n.end_page) for n in tree] == [(3, 3), (3, 3)]


def test_load_or_parse_uses_cache_until_pdf_changes(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 version one")
    calls = []

    def fake_parser(path, doc_id):
        calls.append(doc_id)
        return ParsedDoc(doc_id, "T", file_sha256(path), (Page(1, "text"),), ())

    cache = tmp_path / "cache"
    first = load_or_parse(pdf, "d1", cache, parser=fake_parser)
    second = load_or_parse(pdf, "d1", cache, parser=fake_parser)
    assert first == second and calls == ["d1"]
    assert json.loads((cache / "d1.json").read_text())["doc_id"] == "d1"

    pdf.write_bytes(b"%PDF-1.4 version two")
    load_or_parse(pdf, "d1", cache, parser=fake_parser)
    assert calls == ["d1", "d1"]


@pytest.mark.slow
def test_parse_pdf_on_generated_document(tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    from aex.experiments.tree_rag.parse import parse_pdf

    path = tmp_path / "gen.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    for title, body in [("1 Product Terms", "Issuer: Example Co."),
                        ("2 Barrier", "The barrier level is 60% of the initial level."),
                        ("3 Coupon", "Coupon payment dates: 15 June 2026 and 15 December 2026.")]:
        c.setFont("Helvetica-Bold", 18); c.drawString(72, 760, title)
        c.setFont("Helvetica", 11); c.drawString(72, 730, body)
        c.showPage()
    c.save()

    doc = parse_pdf(path, "gen")
    assert doc.n_pages == 3
    assert "60%" in doc.page(2).text
    assert len(doc.tree) >= 1
    assert doc.sha256 == file_sha256(path)

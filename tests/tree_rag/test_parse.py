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


def test_cache_is_tied_to_the_parser_version(tmp_path, monkeypatch):
    import aex.experiments.tree_rag.parse as parse
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 same bytes")
    calls = []

    def fake_parser(path, doc_id):
        calls.append(doc_id)
        return ParsedDoc(doc_id, "T", file_sha256(path), (Page(1, "text"),), ())

    cache = tmp_path / "cache"
    load_or_parse(pdf, "d1", cache, parser=fake_parser)
    assert json.loads((cache / "d1.json").read_text())["parser"] == parse.PARSER_ID
    monkeypatch.setattr(parse, "PARSER_ID", "something-else")
    load_or_parse(pdf, "d1", cache, parser=fake_parser)
    assert calls == ["d1", "d1"]                      # same PDF, different parser config -> reparse


def test_parser_id_says_ocr_is_off():
    from aex.experiments.tree_rag.parse import PARSER_ID
    assert "no-ocr" in PARSER_ID


def test_relevel_by_numbering():
    from aex.experiments.tree_rag.parse import heading_scheme, relevel
    heads = [("VI. TERMS AND CONDITIONS", 1, 1), ("1. General", 1, 1), ("1.1 Issuer", 1, 2), ("Definitions", 1, 2),
             ("1.2 Form", 1, 3), ("[bei nur einem Basiswert:", 1, 3), ("2. Payments", 1, 4),
             ("VII. FORM OF FINAL TERMS", 1, 5), ("RISK FACTORS", 1, 6)]
    assert [lvl for _, lvl, _ in relevel(heads)] == [1, 3, 4, 5, 4, 5, 3, 1, 2]
    assert heading_scheme("Item 7A. Quantitative and Qualitative") == ("item", 0)
    assert heading_scheme("4.2.1 Barrier") == ("arabic", 2)
    assert heading_scheme("Beobachtungstage:") is None and heading_scheme("[if applicable:") is None


def test_tree_only_change_rebuilds_tree_from_cached_headings(tmp_path, monkeypatch):
    from aex.experiments.tree_rag import parse
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-fake")
    calls = []

    def fake_parser(path, doc_id):
        calls.append(doc_id)
        return ParsedDoc(doc_id, "T", parse.file_sha256(path), (Page(1, "a"), Page(2, "b")),
                         parse.build_tree([("1. A", 1, 1), ("1.1 B", 1, 2)], 2, "T"))

    doc = load_or_parse(pdf, "d", tmp_path / "c", parser=fake_parser)
    assert [n.level for n in doc.tree] == [1, 2]
    cache = json.loads((tmp_path / "c" / "d.json").read_text())
    assert cache["tree_id"] == parse.TREE_ID and cache["headings"] == [["1. A", 1, 1], ["1.1 B", 1, 2]]
    monkeypatch.setattr(parse, "TREE_ID", "next")
    again = load_or_parse(pdf, "d", tmp_path / "c", parser=fake_parser)
    assert calls == ["d"] and [n.level for n in again.tree] == [1, 2]   # no re-parse
    assert json.loads((tmp_path / "c" / "d.json").read_text())["tree_id"] == "next"


def test_parts_nest_inside_roman_sections_when_roman_comes_first():
    from aex.experiments.tree_rag.parse import relevel
    jb = [("VI. TERMS AND CONDITIONS", 1, 1), ("Part A: Product Specific Conditions", 1, 1), ("1. Issue", 1, 2)]
    assert [lvl for _, lvl, _ in relevel(jb)] == [1, 2, 3]
    tenk = [("PART I", 1, 1), ("Item 1. Business", 1, 2), ("PART II", 1, 9), ("Item 5. Market", 1, 9)]
    assert [lvl for _, lvl, _ in relevel(tenk)] == [1, 2, 1, 2]

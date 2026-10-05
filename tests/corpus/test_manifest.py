import pytest

from aex.corpus.manifest import (FIELDS, DocEntry, ExclusionList, ManifestError, load_manifest,
                                 write_manifest)

HEADER = ",".join(FIELDS)


def _row(**over):
    base = dict(doc_id="ts-0001", corpus="termsheets", regime="A", issuer="Example Bank AG", isin="CH0000000001",
                product_type="barrier_reverse_convertible", language="en", issue_date="2026-09-01",
                url="https://example.org/ts1.pdf", sha256="", pages="", licence_note="issuer publication")
    base.update(over)
    return ",".join(str(base[f]) for f in FIELDS)


def _write(tmp_path, *rows):
    p = tmp_path / "manifest.csv"
    p.write_text("\n".join([HEADER, *rows]) + "\n")
    return p


@pytest.fixture
def excl(tmp_path):
    p = tmp_path / "excluded.txt"
    p.write_text("# comment\nForbidden Securities\n\n")
    return ExclusionList.load(p)


def test_loads_valid_rows(tmp_path, excl):
    entries = load_manifest(_write(tmp_path, _row(), _row(doc_id="bp-0001", corpus="prospectuses", regime="B",
                                                          isin="", product_type="base_prospectus")), excluded=excl)
    assert [e.doc_id for e in entries] == ["ts-0001", "bp-0001"]
    assert entries[0].sha256 == "" and entries[0].pages is None


@pytest.mark.parametrize("override,message", [
    (dict(doc_id="TS_1"), "doc_id"),
    (dict(regime="C"), "regime"),
    (dict(sha256="md5:abc"), "sha256"),
    (dict(url="ftp://x/y.pdf"), "url"),
    (dict(issue_date="01.09.2026"), "issue_date"),
    (dict(language="de-CH"), "language"),
])
def test_rejects_bad_fields_with_line_numbers(tmp_path, excl, override, message):
    with pytest.raises(ManifestError, match=rf"manifest\.csv:2: .*{message}"):
        load_manifest(_write(tmp_path, _row(**override)), excluded=excl)


def test_rejects_duplicate_ids(tmp_path, excl):
    with pytest.raises(ManifestError, match="manifest.csv:3: duplicate doc_id"):
        load_manifest(_write(tmp_path, _row(), _row()), excluded=excl)


def test_rejects_excluded_issuer_without_echoing_the_name(tmp_path, excl):
    with pytest.raises(ManifestError) as err:
        load_manifest(_write(tmp_path, _row(issuer="forbidden securities (Guernsey) Ltd")), excluded=excl)
    assert "excluded issuer" in str(err.value) and "orbidden" not in str(err.value)


def test_exclusion_list_fails_closed(tmp_path):
    with pytest.raises(ManifestError, match="exclusion list"):
        ExclusionList.load(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(ManifestError, match="exclusion list"):
        ExclusionList.load(empty)


def test_text_check_finds_mentions_anywhere(excl):
    assert excl.mentioned_in("Calculation agent: FORBIDDEN  SECURITIES AG, Zurich")
    assert excl.mentioned_in("calculation agent: forbidden\nsecurities")          # line-wrapped
    assert not excl.mentioned_in("Calculation agent: Example Bank AG")


def test_write_roundtrip(tmp_path, excl):
    p = _write(tmp_path, _row())
    entries = load_manifest(p, excluded=excl)
    pinned = [DocEntry(**{**e.__dict__, "sha256": "sha256:" + "a" * 64, "pages": 12}) for e in entries]
    write_manifest(p, pinned)
    again = load_manifest(p, excluded=excl)
    assert again[0].sha256 == "sha256:" + "a" * 64 and again[0].pages == 12

import httpx
import pytest

from aex.corpus.fetch import FetchError, fetch_all
from aex.corpus.manifest import DocEntry, ExclusionList


def _pdf_bytes(text: str) -> bytes:
    import io
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for line in text.split("|"):
        c.drawString(72, 760, line)
        c.showPage()
    c.save()
    return buf.getvalue()


PDFS = {
    "/ok.pdf": _pdf_bytes("Barrier reverse convertible|Issuer: Example Bank AG|Barrier 60%"),
    "/platform.pdf": _pdf_bytes("Issuer: Partner Bank|Calculation Agent: Forbidden Securities AG"),
    "/blocked/x.pdf": _pdf_bytes("never fetched"),
}
ROBOTS = "User-agent: *\nDisallow: /blocked/\n"


def _transport(log):
    def handler(request: httpx.Request) -> httpx.Response:
        log.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS) if request.url.host == "docs.example" else httpx.Response(404)
        if request.url.path == "/html":
            return httpx.Response(200, text="<html>login</html>", headers={"content-type": "text/html"})
        body = PDFS.get(request.url.path)
        return httpx.Response(200, content=body, headers={"content-type": "application/pdf"}) if body else httpx.Response(404)
    return httpx.MockTransport(handler)


def _entry(doc_id, path, sha=""):
    return DocEntry(doc_id, "ts", "A", "Example Bank AG", "", "brc", "en", "", f"https://docs.example{path}",
                    sha, None, "")


@pytest.fixture
def excl(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("Forbidden Securities\n")
    return ExclusionList.load(p)


def test_downloads_pins_checksum_and_counts_pages(tmp_path, excl):
    log = []
    (out,) = fetch_all([_entry("ok", "/ok.pdf")], tmp_path / "data", excluded=excl, transport=_transport(log),
                       min_interval_s=0)
    assert out.sha256.startswith("sha256:") and out.pages == 3
    assert (tmp_path / "data" / "ts" / "ok.pdf").read_bytes() == PDFS["/ok.pdf"]
    assert log[0] == "/robots.txt"


def test_existing_file_with_matching_checksum_is_not_refetched(tmp_path, excl):
    log = []
    (pinned,) = fetch_all([_entry("ok", "/ok.pdf")], tmp_path / "data", excluded=excl, transport=_transport(log), min_interval_s=0)
    log.clear()
    fetch_all([pinned], tmp_path / "data", excluded=excl, transport=_transport(log), min_interval_s=0)
    assert "/ok.pdf" not in log


def test_checksum_mismatch_is_an_error(tmp_path, excl):
    with pytest.raises(FetchError, match="ok: sha256 mismatch"):
        fetch_all([_entry("ok", "/ok.pdf", sha="sha256:" + "0" * 64)], tmp_path / "data", excluded=excl,
                  transport=_transport([]), min_interval_s=0)


def test_robots_disallow_is_respected_and_nothing_is_fetched(tmp_path, excl):
    log = []
    with pytest.raises(FetchError, match="x: robots.txt disallows"):
        fetch_all([_entry("x", "/blocked/x.pdf")], tmp_path / "data", excluded=excl, transport=_transport(log),
                  min_interval_s=0)
    assert "/blocked/x.pdf" not in log


def test_excluded_text_deletes_the_file_and_never_names_the_match(tmp_path, excl):
    with pytest.raises(FetchError) as err:
        fetch_all([_entry("platform", "/platform.pdf")], tmp_path / "data", excluded=excl,
                  transport=_transport([]), min_interval_s=0)
    assert "excluded" in str(err.value) and "orbidden" not in str(err.value)
    assert not (tmp_path / "data" / "ts" / "platform.pdf").exists()


def test_non_pdf_response_is_an_error(tmp_path, excl):
    with pytest.raises(FetchError, match="not a PDF"):
        fetch_all([_entry("h", "/html")], tmp_path / "data", excluded=excl, transport=_transport([]), min_interval_s=0)


def test_collects_all_failures_when_asked(tmp_path, excl):
    entries = [_entry("ok", "/ok.pdf"), _entry("x", "/blocked/x.pdf"), _entry("h", "/html")]
    errors = []
    out = fetch_all(entries, tmp_path / "data", excluded=excl, transport=_transport([]), min_interval_s=0,
                    errors=errors)
    assert [e.doc_id for e in out] == ["ok"] and len(errors) == 2

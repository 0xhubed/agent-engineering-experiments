"""Download corpus documents politely, pin their checksums, and screen them before use.

Usage: python -m aex.corpus.fetch corpus/<corpus>/manifest.csv [--data data]

Documents whose licence_note starts with "manual download" are never fetched: their host's terms
forbid automated access, so a person downloads them and this only verifies and screens the file.
Rules this enforces for the rest, in order: robots.txt allows the URL (no robots.txt = allowed); the
response is a PDF; the checksum matches the manifest (or is pinned on first download); the
document text names no excluded issuer anywhere (otherwise the file is deleted). Requests to
one host are spaced by `min_interval_s`. Bot protection is never worked around: a 403/429
is reported like any other failure.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from aex.corpus.manifest import DocEntry, ExclusionList, load_manifest, write_manifest

USER_AGENT = "agent-engineering-experiments research (+https://www.agent-engineering.ch)"


class FetchError(RuntimeError):
    def __init__(self, doc_id: str, reason: str) -> None:
        super().__init__(f"{doc_id}: {reason}")
        self.doc_id = doc_id


def pdf_text_and_pages(path: Path) -> tuple[str, int]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    try:
        return "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf))), len(pdf)
    finally:
        pdf.close()


class _Polite:
    """One HTTP client, robots.txt per host, and a minimum gap between requests to a host."""

    def __init__(self, transport: httpx.BaseTransport | None, min_interval_s: float) -> None:
        self._http = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120,
                                  transport=transport)
        self._robots: dict[str, RobotFileParser] = {}
        self._last: dict[str, float] = {}
        self._gap = min_interval_s

    def _wait(self, host: str) -> None:
        delay = self._last.get(host, 0.0) + self._gap - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser = RobotFileParser()
            self._wait(parts.netloc)
            response = self._http.get(f"{origin}/robots.txt")
            parser.parse(response.text.splitlines() if response.status_code == 200 else [])
            self._robots[origin] = parser
        return self._robots[origin].can_fetch(USER_AGENT, url)

    def get(self, url: str) -> httpx.Response:
        self._wait(urlsplit(url).netloc)
        return self._http.get(url)


def fetch_one(entry: DocEntry, data_dir: Path, *, excluded: ExclusionList, client: _Polite) -> DocEntry:
    target = data_dir / entry.corpus / f"{entry.doc_id}.pdf"
    manual = entry.licence_note.startswith("manual download")
    if manual and not target.exists():
        # The host's terms forbid automated access: a person downloads it in a browser.
        raise FetchError(entry.doc_id, f"manual document: download it by hand from {entry.url} to {target}")
    if not manual and not (target.exists() and entry.sha256 and _sha(target.read_bytes()) == entry.sha256):
        if not client.allowed(entry.url):
            raise FetchError(entry.doc_id, f"robots.txt disallows {entry.url}")
        response = client.get(entry.url)
        if response.status_code != 200:
            raise FetchError(entry.doc_id, f"HTTP {response.status_code} for {entry.url}")
        if not response.content.startswith(b"%PDF"):
            raise FetchError(entry.doc_id, f"not a PDF ({response.headers.get('content-type', '?')})")
        sha = _sha(response.content)
        if entry.sha256 and sha != entry.sha256:
            raise FetchError(entry.doc_id, f"sha256 mismatch: manifest {entry.sha256}, downloaded {sha}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
    if manual and entry.sha256 and _sha(target.read_bytes()) != entry.sha256:
        raise FetchError(entry.doc_id, f"sha256 mismatch for the hand-downloaded file {target}")
    text, pages = pdf_text_and_pages(target)
    if excluded.mentioned_in(text):
        target.unlink()
        raise FetchError(entry.doc_id, "document text names an excluded issuer; file deleted")
    return replace(entry, sha256=_sha(target.read_bytes()), pages=pages)


def fetch_all(entries: list[DocEntry], data_dir: str | Path, *, excluded: ExclusionList,
              transport: httpx.BaseTransport | None = None, min_interval_s: float = 2.0,
              errors: list[FetchError] | None = None) -> list[DocEntry]:
    """Fetch every entry. With `errors` given, failures are collected and the rest continue;
    otherwise the first failure raises."""
    client = _Polite(transport, min_interval_s)
    done: list[DocEntry] = []
    for entry in entries:
        try:
            done.append(fetch_one(entry, Path(data_dir), excluded=excluded, client=client))
        except FetchError as exc:
            if errors is None:
                raise
            errors.append(exc)
    return done


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="fetch")
    parser.add_argument("manifest")
    parser.add_argument("--data", default="data")
    args = parser.parse_args(argv[1:])
    excluded = ExclusionList.load()
    entries = load_manifest(args.manifest, excluded=excluded)
    errors: list[FetchError] = []
    fetched = {e.doc_id: e for e in fetch_all(entries, args.data, excluded=excluded, errors=errors)}
    failed = {e.doc_id for e in errors}
    # Pin what succeeded; drop documents that were excluded by their text; keep other failures for a retry.
    kept = [fetched.get(e.doc_id, e) for e in entries
            if not any(err.doc_id == e.doc_id and "excluded issuer" in str(err) for err in errors)]
    write_manifest(args.manifest, kept)
    for err in errors:
        print(f"FAILED {err}", file=sys.stderr)
    print(f"fetched {len(fetched)}, failed {len(failed)}, manifest rows {len(kept)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

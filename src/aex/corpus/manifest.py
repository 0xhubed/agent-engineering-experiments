"""Corpus manifests: the only committed trace of a document (URL, sha256, metadata — never the file).

The exclusion list (issuers whose documents may never enter a corpus) lives outside the repo,
because this repo is public and the list itself is private. Every check fails closed when the
list is missing, and no error message ever echoes a listed name.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_EXCLUSION_FILE = Path("~/.config/aex/excluded_issuers.txt").expanduser()

FIELDS = ("doc_id", "corpus", "regime", "issuer", "isin", "product_type", "language", "issue_date",
          "url", "sha256", "pages", "licence_note")

_PATTERNS = {
    "doc_id": re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$"),
    "regime": re.compile(r"^(A|B|claim)$"),
    "sha256": re.compile(r"^(sha256:[0-9a-f]{64})?$"),
    "url": re.compile(r"^https?://\S+$"),
    "issue_date": re.compile(r"^(\d{4}-\d{2}-\d{2})?$"),
    "language": re.compile(r"^[a-z]{2}$"),
    "pages": re.compile(r"^(\d+)?$"),
}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class DocEntry:
    doc_id: str
    corpus: str
    regime: str
    issuer: str
    isin: str
    product_type: str
    language: str
    issue_date: str
    url: str
    sha256: str
    pages: int | None
    licence_note: str


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


class ExclusionList:
    def __init__(self, names: list[str]) -> None:
        self._names = [_normalise(n) for n in names]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_EXCLUSION_FILE) -> ExclusionList:
        path = Path(path)
        if not path.exists():
            raise ManifestError(f"exclusion list not found at {path}; refusing to continue")
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if not names:
            raise ManifestError(f"exclusion list at {path} is empty; refusing to continue")
        return cls(names)

    def mentioned_in(self, text: str) -> bool:
        """True if any listed name occurs anywhere in `text` (case- and whitespace-insensitive)."""
        haystack = _normalise(text)
        return any(name in haystack for name in self._names)


def load_manifest(path: str | Path, *, excluded: ExclusionList) -> list[DocEntry]:
    path = Path(path)
    entries: list[DocEntry] = []
    seen: set[str] = set()
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ManifestError(f"{path.name}:1: header must be {','.join(FIELDS)}")
        for lineno, row in enumerate(reader, start=2):
            where = f"{path.name}:{lineno}"
            for name, pattern in _PATTERNS.items():
                if not pattern.match(row[name] or ""):
                    raise ManifestError(f"{where}: invalid {name} {row[name]!r}")
            if row["doc_id"] in seen:
                raise ManifestError(f"{where}: duplicate doc_id {row['doc_id']!r}")
            seen.add(row["doc_id"])
            if excluded.mentioned_in(row["issuer"]) or excluded.mentioned_in(row["url"]):
                raise ManifestError(f"{where}: excluded issuer (see the private exclusion list)")
            entries.append(DocEntry(**{**row, "pages": int(row["pages"]) if row["pages"] else None}))
    return entries


def write_manifest(path: str | Path, entries: list[DocEntry]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for e in entries:
            row = {f.name: getattr(e, f.name) for f in fields(DocEntry)}
            row["pages"] = "" if e.pages is None else e.pages
            writer.writerow(row)

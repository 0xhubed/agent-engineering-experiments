"""Select Julius Bär barrier reverse convertibles from the public "recently issued" list (static HTML,
allowed by robots.txt) and append them to corpus/termsheets/manifest.csv.

Usage: uv run python scripts/corpus/julius_baer.py <saved recently-issued.html> [--per-family 14]
Final terms URL: https://docs.juliusbaer.com/documents/alias/FinalTerms_{valor}_EN
"""
from __future__ import annotations

import argparse
import html
import re
from collections import defaultdict
from pathlib import Path

from aex.corpus.manifest import FIELDS, DocEntry, ExclusionList, load_manifest, write_manifest

MANIFEST = Path("corpus/termsheets/manifest.csv")
FAMILIES = [("autocallable_barrier_reverse_convertible", "Autocallable Barrier Reverse Convertible"),
            ("callable_barrier_reverse_convertible", "Callable Barrier Reverse Convertible"),
            ("barrier_reverse_convertible", "JB Barrier Reverse Convertible")]


def isin_from_valor(valor: str) -> str:
    """Swiss ISIN: 'CH' + 9-digit zero-padded valor + Luhn check digit over the letter-expanded string."""
    body = "CH" + valor.zfill(9)
    digits = "".join(str(int(ch, 36)) for ch in body)
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 == 0 else 1)
        total += n // 10 + n % 10
    return body + str((10 - total % 10) % 10)


def rows(page: str) -> list[dict]:
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split())
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) >= 4 and cells[3].isdigit():
            out.append({"name": cells[0], "maturity": cells[1], "currency": cells[2], "valor": cells[3]})
    return out


def family(name: str) -> str | None:
    for fam, marker in FAMILIES:
        if marker in name:
            return fam
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--per-family", type=int, default=14)
    args = ap.parse_args()
    excluded = ExclusionList.load()
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows(Path(args.page).read_text()):
        fam = family(r["name"])
        if fam and not excluded.mentioned_in(r["name"]):
            by_family[fam].append(r)
    existing = load_manifest(MANIFEST, excluded=excluded) if MANIFEST.exists() else []
    have = {e.doc_id for e in existing}
    new = []
    for fam, _ in FAMILIES:
        # Keep list order (newest first on the page); a family of near-identical templates is the point.
        for r in by_family[fam][:args.per_family]:
            doc_id = f"jb-{r['valor']}"
            if doc_id in have:
                continue
            new.append(DocEntry(doc_id, "termsheets", "A", "Bank Julius Baer & Co. Ltd.", isin_from_valor(r["valor"]),
                                fam, "en", "", f"https://docs.juliusbaer.com/documents/alias/FinalTerms_{r['valor']}_EN",
                                "", None, "issuer final terms, public; URL + sha256 only, never redistributed"))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(MANIFEST, existing + new)
    print({fam: min(len(by_family[fam]), args.per_family) for fam, _ in FAMILIES}, "added", len(new))


if __name__ == "__main__":
    main()

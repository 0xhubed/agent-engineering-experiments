"""Select Raiffeisen Bank International reverse convertible bonds (raiffeisenzertifikate.at, robots allows)
and append their final terms to corpus/termsheets/manifest.csv.

Usage: uv run python scripts/corpus/rbi.py [--limit 30]
Listing -> product page (one request each, 2 s apart) -> the "Final Terms" link on the product page.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from aex.corpus.fetch import _Polite
from aex.corpus.manifest import DocEntry, ExclusionList, load_manifest, write_manifest

BASE = "https://www.raiffeisenzertifikate.at"
LISTING = f"{BASE}/en/certificates/investment-products/reverse-convertible-bonds"
MANIFEST = Path("corpus/termsheets/manifest.csv")


def final_terms(page: str) -> tuple[str, str] | None:
    """(url, date) of the Final Terms link on a product page."""
    for m in re.finditer(r'href="(/en/file\?ISIN=[^"]+)"', page):
        context = " ".join(re.sub(r"<[^>]+>", " ", page[m.end():m.end() + 400]).split())
        found = re.search(r"Final Terms \((\d{2})\.(\d{2})\.(\d{4})\)", context)
        if found:
            d, mth, y = found.groups()
            return BASE + html.unescape(m.group(1)), f"{y}-{mth}-{d}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    excluded = ExclusionList.load()
    client = _Polite(None, 2.0)
    if not client.allowed(LISTING):
        raise SystemExit("robots.txt disallows the listing")
    listing = client.get(LISTING).text
    products = sorted(set(re.findall(r'href="(/en/certificate/[^"]+-(at0000[a-z0-9]{6}))"', listing)))
    existing = load_manifest(MANIFEST, excluded=excluded) if MANIFEST.exists() else []
    have = {e.doc_id for e in existing}
    new, skipped = [], []
    for path, isin in products[:args.limit]:
        doc_id = f"rbi-{isin}"
        if doc_id in have or not client.allowed(BASE + path):
            continue
        page = client.get(BASE + path).text
        title = " ".join(html.unescape(re.search(r"<title>(.*?)</title>", page, re.S).group(1)).split())
        ft = final_terms(page)
        if ft is None or excluded.mentioned_in(title):
            skipped.append(isin)
            continue
        new.append(DocEntry(doc_id, "termsheets", "A", "Raiffeisen Bank International AG", isin.upper(),
                            "reverse_convertible_bond", "en", ft[1], ft[0], "", None,
                            f"issuer final terms, public; URL + sha256 only, never redistributed; {title[:80]}"))
    write_manifest(MANIFEST, existing + new)
    print("added", len(new), "skipped (no final terms link)", skipped)


if __name__ == "__main__":
    main()

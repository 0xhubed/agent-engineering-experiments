"""Select Deutsche Bank X-markets products (robots.txt allows everything) and append their final terms
(German) to corpus/termsheets/manifest.csv. First listing page per product family (20 products each).

Usage: uv run python scripts/corpus/deutsche_bank.py
Final terms: https://www.xmarkets.db.com/DE/Download/Offering/{ISIN}
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from aex.corpus.fetch import _Polite
from aex.corpus.manifest import DocEntry, ExclusionList, load_manifest, write_manifest

BASE = "https://www.xmarkets.db.com"
FAMILIES = {"express_certificate": "/DE/Produkt_Uebersicht/Express-Zertifikate",
            "reverse_convertible_bond": "/DE/Produkt_Uebersicht/Aktienanleihen"}
MANIFEST = Path("corpus/termsheets/manifest.csv")


def products(page: str) -> list[tuple[str, str]]:
    found = re.findall(r'<a href="/DE/Produkt_Detail/(DE000D[A-Z0-9]{6})">(.*?)</a>', page)
    seen, out = set(), []
    for isin, name in found:
        if isin not in seen:
            seen.add(isin)
            out.append((isin, " ".join(html.unescape(re.sub(r"<[^>]+>", " ", name)).split())))
    return out


def main() -> None:
    excluded = ExclusionList.load()
    client = _Polite(None, 2.0)
    existing = load_manifest(MANIFEST, excluded=excluded) if MANIFEST.exists() else []
    have = {e.doc_id for e in existing}
    new = []
    for family, path in FAMILIES.items():
        if not client.allowed(BASE + path):
            raise SystemExit(f"robots.txt disallows {path}")
        for isin, name in products(client.get(BASE + path).text):
            doc_id = f"db-{isin.lower()}"
            if doc_id in have or excluded.mentioned_in(name):
                continue
            new.append(DocEntry(doc_id, "termsheets", "A", "Deutsche Bank AG", isin, family, "de", "",
                                f"{BASE}/DE/Download/Offering/{isin}", "", None,
                                f"issuer final terms, public; URL + sha256 only, never redistributed; {name[:80]}"))
    write_manifest(MANIFEST, existing + new)
    print("added", len(new), {f: sum(1 for e in new if e.product_type == f) for f in FAMILIES})


if __name__ == "__main__":
    main()

"""Deterministic answer scoring. The LLM judge is only used for kind='free'."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

GoldKind = Literal["exact", "numeric", "date", "date_list", "free", "unanswerable"]
NOT_STATED = "NOT_STATED"


@dataclass(frozen=True)
class GoldAnswer:
    kind: GoldKind
    value: str | list[str] | None


@dataclass(frozen=True)
class ScoreResult:
    correct: bool | None
    method: str
    failure: str | None


_REFUSAL = re.compile(
    r"\bNOT_STATED\b"
    r"|\bnot (?:stated|specified|mentioned|provided|available|found|contained)\b"
    r"|\bcannot be determined\b|\bno information\b"
    r"|\bdoes not (?:state|specify|mention|contain)\b",
    re.IGNORECASE,
)
_CURRENCY = re.compile(r"\b(?:CHF|EUR|USD|GBP|JPY)\b|[$€£]", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d[\d.,]*\s*%?")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# English and German month names (the Deutsche Bank final terms are German): "11. Februar 2028", "15 June 2027".
_MONTH_NUMBER = {name.casefold(): n for n, names in enumerate([
    ("January", "Januar", "Jänner", "Jan"), ("February", "Februar", "Feb"), ("March", "März", "Maerz", "Mar", "Mär"),
    ("April", "Apr"), ("May", "Mai"), ("June", "Juni", "Jun"), ("July", "Juli", "Jul"), ("August", "Aug"),
    ("September", "Sept", "Sep"), ("October", "Oktober", "Oct", "Okt"), ("November", "Nov"),
    ("December", "Dezember", "Dec", "Dez")], 1) for name in names}
_DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})\.? (" + "|".join(sorted(_MONTH_NUMBER, key=len, reverse=True))
                             + r")\.? (\d{4})\b", re.IGNORECASE)
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "%Y-%m-%d"),
    (re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"), "%d.%m.%Y"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"), "%d/%m/%Y"),
    (re.compile(rf"\b(?:{_MONTHS}) \d{{1,2}}, \d{{4}}\b"), "%B %d, %Y"),
    (re.compile(r"\b\d{1,2}-[A-Z][a-z]{2}-\d{4}\b"), "%d-%b-%Y"),
]


def extract_final(text: str) -> str:
    if "ANSWER:" in text:
        tail = text.rsplit("ANSWER:", 1)[1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))


def _readings(raw: str, *, percent: bool = False) -> set[float]:
    """Every value the token can mean. English and German separators both occur in the corpus: the last of
    two different separators is the decimal one; a lone separator followed by 1-2 or 4+ digits is decimal;
    a lone one followed by exactly 3 digits ("500.000", "1,000") is ambiguous and yields both readings, except
    in a percentage ("10.875%"), which never has thousands."""
    raw = raw.strip().rstrip(".,")
    seps = [c for c in raw if c in ",."]
    if not seps:
        return {float(raw)}
    if len(set(seps)) == 2:
        dec = seps[-1]
        return {float(raw.replace("," if dec == "." else ".", "").replace(",", "."))}
    sep = seps[0]
    head, *groups = raw.split(sep)
    if len(groups) > 1:                                              # 1.000.000 / 1,000,000
        return {float(raw.replace(sep, ""))}
    decimal = float(f"{head}.{groups[0]}")
    return {decimal, float(head + groups[0])} if len(groups[0]) == 3 and not percent else {decimal}


def parse_numbers(s: str) -> set[float] | None:
    cleaned = _CURRENCY.sub("", s)
    for sep in ("'", "’", " ", " "):
        cleaned = cleaned.replace(sep, "")
    match = _NUMBER.search(cleaned)
    if not match:
        return None
    token = match.group(0)
    is_pct = token.rstrip().endswith("%")
    values = _readings(token.replace("%", ""), percent=is_pct)
    return values | {v / 100 for v in values} if is_pct else values


def parse_dates(s: str) -> list[date]:
    found: list[tuple[int, date]] = []
    for pattern, fmt in _DATE_PATTERNS:
        for m in pattern.finditer(s):
            try:
                found.append((m.start(), datetime.strptime(m.group(0), fmt).date()))
            except ValueError:
                continue
    for m in _DAY_MONTH_YEAR.finditer(s):
        try:
            found.append((m.start(), date(int(m.group(3)), _MONTH_NUMBER[m.group(2).casefold()], int(m.group(1)))))
        except ValueError:
            continue
    found.sort(key=lambda pair: pair[0])
    return [d for _, d in found]


def _numbers_match(a: set[float], g: set[float], rel_tol: float) -> bool:
    for x in a:
        for y in g:
            if y == 0 and abs(x) < 1e-9:
                return True
            if y != 0 and abs(x - y) <= rel_tol * abs(y):
                return True
    return False


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip(" .,;:")


def score(answer_text: str, gold: GoldAnswer, *, rel_tol: float = 0.005) -> ScoreResult:
    answer = extract_final(answer_text)
    if gold.kind == "unanswerable":
        ok = is_refusal(answer)
        return ScoreResult(ok, "refusal", None if ok else "answered_unanswerable")
    if is_refusal(answer):
        return ScoreResult(False, gold.kind, None)
    if gold.kind == "free":
        return ScoreResult(None, "llm", None)
    if gold.kind == "exact":
        # The gold may be followed by a qualifier: "Deutsche Bank AG, Taunusanlage 12, …", "Austrian law (…)".
        a, g = _norm_text(answer), _norm_text(str(gold.value))
        return ScoreResult(a == g or bool(re.match(re.escape(g) + r"(?:\s*[,(;]|\s+[–-]\s)", a)), "exact", None)
    if gold.kind == "numeric":
        a, g = parse_numbers(answer), parse_numbers(str(gold.value))
        return ScoreResult(bool(a and g and _numbers_match(a, g, rel_tol)), "numeric", None)
    if gold.kind == "date":
        a, g = parse_dates(answer), parse_dates(str(gold.value))
        return ScoreResult(bool(a and g and a[0] == g[0]), "date", None)
    if gold.kind == "date_list":
        want = {d for item in (gold.value or []) for d in parse_dates(item)}
        return ScoreResult(set(parse_dates(answer)) == want and bool(want), "date_list", None)
    raise ValueError(f"unknown gold kind {gold.kind!r}")

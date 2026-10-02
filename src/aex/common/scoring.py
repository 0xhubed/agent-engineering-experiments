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
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "%Y-%m-%d"),
    (re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"), "%d.%m.%Y"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"), "%d/%m/%Y"),
    (re.compile(rf"\b\d{{1,2}} (?:{_MONTHS}) \d{{4}}\b"), "%d %B %Y"),
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


def _to_float(raw: str) -> float:
    raw = raw.strip()
    if re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})+,\d+", raw):      # European 1.000,50
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", "")
    return float(raw.rstrip("."))


def parse_numbers(s: str) -> set[float] | None:
    cleaned = _CURRENCY.sub("", s)
    for sep in ("'", "’", " ", " "):
        cleaned = cleaned.replace(sep, "")
    match = _NUMBER.search(cleaned)
    if not match:
        return None
    token = match.group(0)
    is_pct = token.rstrip().endswith("%")
    value = _to_float(token.replace("%", ""))
    return {value, value / 100} if is_pct else {value}


def parse_dates(s: str) -> list[date]:
    found: list[tuple[int, date]] = []
    for pattern, fmt in _DATE_PATTERNS:
        for m in pattern.finditer(s):
            try:
                found.append((m.start(), datetime.strptime(m.group(0), fmt).date()))
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
        return ScoreResult(_norm_text(answer) == _norm_text(str(gold.value)), "exact", None)
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

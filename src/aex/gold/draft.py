"""Draft gold-question candidates for the structured-products corpora (plan 1b, Task 6).

Usage: python -m aex.gold.draft configs/gold/draft.yaml [--stages facts,window,...] [--limit N]

Drafts never enter a run. They are written to gold/drafts/ with status "draft" (or "auto_rejected"
with a reason) and become gold only through the review app, which a person drives.

Stages, in order (each resumable; finished tasks are logged in gold/drafts/_done.jsonl):
- facts          regime A: key terms per product, each with page and verbatim quote
- disambiguation regime A, no model call: product descriptions (needed before review starts) and
                 template questions about a field whose value differs between near-duplicate
                 products (same issuer, type and underlyings)
- window         lookup / table / computation drafts from page windows (regime A: the key-terms
                 pages; regime B: seeded windows across each prospectus)
- unanswerable   candidates from the same windows, each confirmed absent against the whole document
                 (or its best-matching pages when the document is too long)
- cross_doc      regime A termsheet <-> the securities note it completes

Regime-A questions refer to the product as "{product}". The review app renders every accepted item
in two forms sharing a pair_id: by ISIN, and by the product's description from `descriptions.json`
(built here from the extracted facts and unique among all regime-A products).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

import yaml

from aex.common.llm import ChatClient, LLMError
from aex.corpus.manifest import DocEntry, ExclusionList, load_manifest
from aex.experiments.tree_rag.arms.base import Store
from aex.experiments.tree_rag.retrieval import BM25, top_k
from aex.experiments.tree_rag.types import ParsedDoc

PROMPT_VERSION = "draft-v1"
PROMPTS = Path(__file__).parent / "prompts"
STAGES = ("facts", "disambiguation", "window", "unanswerable", "cross_doc")
KINDS = {"exact", "numeric", "date", "date_list", "free"}

ISSUER_SHORT = {"Bank Julius Baer & Co. Ltd.": "Julius Baer", "Raiffeisen Bank International AG": "Raiffeisen",
                "Deutsche Bank AG": "Deutsche Bank"}
PRODUCT_LABEL = {
    "autocallable_barrier_reverse_convertible": "autocallable barrier reverse convertible",
    "callable_barrier_reverse_convertible": "callable barrier reverse convertible",
    "barrier_reverse_convertible": "barrier reverse convertible",
    "reverse_convertible_bond": "reverse convertible", "express_certificate": "express certificate",
    "capped_bonus_certificate": "capped bonus certificate",
}
FACT_FIELDS = ("underlyings", "currency", "denomination", "issue_price", "issue_size", "coupon_rate",
               "barrier_level", "strike_level", "autocall_trigger_level", "initial_fixing_date", "issue_date",
               "final_fixing_date", "maturity_date", "listing")
# Template questions for disambiguation drafts: field -> (question, kind)
DISAMBIGUATION_TEMPLATES = {
    "barrier_level": ("What is the barrier level of {product}, as a percentage of the initial level?", "numeric"),
    "coupon_rate": ("What coupon or interest rate per annum does {product} pay?", "numeric"),
    "strike_level": ("What is the strike level of {product}, as a percentage of the initial level?", "numeric"),
    "autocall_trigger_level": ("At what level, as a percentage of the initial level, is {product} redeemed early?",
                               "numeric"),
    "maturity_date": ("On what date is {product} redeemed at maturity?", "date"),
    "final_fixing_date": ("What is the final fixing (valuation) date of {product}?", "date"),
    "initial_fixing_date": ("What is the initial fixing date of {product}?", "date"),
    "issue_size": ("What is the issue size of {product}?", "numeric"),
}
# Regime-A window drafts all read the same key-terms pages of near-identical documents; without a
# per-document nudge every product gets the same three questions. Barrier, coupon and maturity are
# left out: the disambiguation templates already ask about them.
FOCUS = {
    "lookup": ("the calculation agent or paying agent", "the governing law or place of jurisdiction",
               "the exchange listing or last trading day", "the minimum investment or trading lot",
               "the settlement type (cash or physical delivery)", "an underlying's initial level or strike price",
               "the conversion ratio (shares per product)", "the issue size or number of products",
               "a security code (Valor, WKN, common code) or the clearing system", "fees or distribution costs",
               "the issue price or denomination", "the fixing time or reference exchange of an underlying"),
    "table": ("the per-underlying table (initial level, barrier, strike, ratio)",
              "the schedule of observation, trigger or early-redemption dates and levels",
              "coupon or interest payment dates and amounts", "the cost or fee breakdown",
              "a scenario or performance table"),
    "computation": ("the redemption amount for a stated final level below the barrier",
                    "the redemption amount when no barrier event occurs",
                    "the number of shares delivered and any fractional cash amount",
                    "the amount of a single coupon payment per product", "the total of all coupons over the life",
                    "the distance between the initial level and the barrier"),
    "unanswerable": ("a feature other structured products have but this one lacks (a cap, participation rate, "
                     "capital-protection level, autocall trigger, lock-in, quanto feature...)",
                     "a date or level for an event this product does not have",
                     "a fee, cost or spread the document does not quantify",
                     "an operational or legal detail the document does not give",
                     "a figure about the underlying that the document does not report"),
}
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


class DraftError(ValueError):
    pass


# --- small helpers -------------------------------------------------------------------------------

def parse_json(text: str) -> dict:
    """The first JSON object in a model reply (tolerates ``` fences and prose around it)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise DraftError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise DraftError("reply is not a JSON object")
    return obj


def _norm(s: str) -> str:
    s = s.replace("­", "").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"-\s*\n\s*", "", s)   # hyphenation across lines
    return " ".join(s.casefold().split())


def quote_on_page(quote: str | None, page_text: str, *, min_ratio: float = 0.9) -> bool:
    """True if `quote` occurs on the page, allowing small differences (parsing, typography)."""
    if not quote or len(quote.split()) < 3:
        return False
    q, p = _norm(quote), _norm(page_text)
    if q in p:
        return True
    matcher = SequenceMatcher(None, p, q, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    if matched / len(q) < min_ratio:
        return False
    # The matching characters must also be close together on the page, not scattered.
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    span = blocks[-1].a + blocks[-1].size - blocks[0].a
    return span <= 1.5 * len(q)


def render_pages(doc: ParsedDoc, numbers: Iterable[int], *, max_words_per_page: int = 1500) -> str:
    out = []
    for n in numbers:
        words = doc.page(n).text.split()
        out.append(f"[page {n}]\n{' '.join(words[:max_words_per_page])}")
    return "\n\n".join(out)


def human_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def doc_label(entry: DocEntry) -> str:
    kind = entry.product_type.replace("_", " ")
    dated = f" dated {human_date(entry.issue_date)}" if entry.issue_date else ""
    return f"the {entry.issuer} {kind}{dated}"


def prompt(name: str, **values) -> str:
    return (PROMPTS / f"{name}.txt").read_text().format(**values)


def content_pages(doc: ParsedDoc, *, min_words: int = 40) -> list[int]:
    return [p.number for p in doc.pages if len(p.text.split()) >= min_words]


def key_pages(doc: ParsedDoc, n: int = 12) -> list[int]:
    """The pages where final terms state the product's terms: the first n pages with text."""
    return content_pages(doc)[:n]


def has_table(text: str) -> bool:
    return bool(re.search(r"^\|.*\|\s*$\n^\|[\s:|-]+\|\s*$", text, re.MULTILINE))


def prospectus_windows(doc: ParsedDoc, n: int, *, size: int = 3, seed: int = 0) -> list[list[int]]:
    """n non-overlapping windows of `size` consecutive content pages; half start on a page with a table."""
    rng = random.Random(f"{seed}:{doc.doc_id}")
    pages = content_pages(doc, min_words=120)
    starts_table = [p for p in pages if has_table(doc.page(p).text)]
    rng.shuffle(starts_table)
    starts_other = pages[:]
    rng.shuffle(starts_other)
    chosen: list[list[int]] = []
    used: set[int] = set()

    def take(candidates: list[int], k: int) -> None:
        for start in candidates:
            if len(chosen) >= k:
                return
            window = [p for p in range(start, start + size) if p <= doc.n_pages]
            if not used.intersection(range(start - 1, start + size + 1)):
                chosen.append(window)
                used.update(window)

    take(starts_table, n // 2)
    take(starts_other, n)
    return sorted(chosen)


# --- facts and descriptions -----------------------------------------------------------------------

def check_facts(raw: dict, doc: ParsedDoc) -> dict:
    """Keep the known fields; mark each with whether its quote is really on its page."""
    facts = {}
    for name in FACT_FIELDS:
        f = raw.get(name) or {}
        value, page, quote = f.get("value"), f.get("page"), f.get("quote")
        ok = (value not in (None, "", []) and isinstance(page, int) and 1 <= page <= doc.n_pages
              and quote_on_page(quote, doc.page(page).text))
        facts[name] = {"value": value, "page": page, "quote": quote, "quote_ok": ok}
    return facts


def _num(value) -> float | None:
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(value or "").replace("'", ""))
    return float(m.group(0).replace(",", ".")) if m else None


def _attr_value(facts: dict, name: str):
    f = facts.get(name) or {}
    if not f.get("quote_ok"):
        return None
    v = f["value"]
    if name == "underlyings":
        return tuple(sorted(_norm(u) for u in v)) if isinstance(v, list) and v else None
    if name.endswith("_date"):
        try:
            d = date.fromisoformat(str(v))
        except ValueError:
            return None
        return (d.year, d.month)
    if name in ("currency", "listing"):
        return _norm(str(v))
    return _num(v)


DESCRIPTION_ATTRS = ("maturity_date", "coupon_rate", "barrier_level", "currency", "strike_level")


def describe_all(entries: dict[str, DocEntry], facts: dict[str, dict]) -> dict[str, dict]:
    """For each product: the shortest description (issuer, type, underlyings, then attributes in
    DESCRIPTION_ATTRS order) that no other regime-A product matches. Products whose facts cannot
    make them unique get description None — they are asked about by ISIN only."""
    keys = {}
    for doc_id, f in facts.items():
        e = entries[doc_id]
        keys[doc_id] = {"issuer": e.issuer, "type": e.product_type, "underlyings": _attr_value(f, "underlyings"),
                        **{a: _attr_value(f, a) for a in DESCRIPTION_ATTRS}}
    out = {}
    for doc_id, mine in keys.items():
        if mine["underlyings"] is None:
            out[doc_id] = {"description": None, "attrs": [], "siblings": [], "reason": "underlyings not verified"}
            continue
        base = ("issuer", "type", "underlyings")
        siblings = sorted(o for o, k in keys.items() if o != doc_id and all(k[a] == mine[a] for a in base))
        used = list(base)
        rivals = siblings
        for attr in DESCRIPTION_ATTRS:
            if not rivals:
                break
            if mine[attr] is None:
                continue
            narrowed = [o for o in rivals if keys[o][attr] == mine[attr]]
            if len(narrowed) < len(rivals):
                used.append(attr)
                rivals = narrowed
        out[doc_id] = {"description": render_description(entries[doc_id], facts[doc_id], used) if not rivals else None,
                       "attrs": used, "siblings": siblings,
                       **({"reason": f"not unique among {len(rivals) + 1}"} if rivals else {})}
    return out


def render_description(entry: DocEntry, facts: dict, attrs: list[str]) -> str:
    issuer = ISSUER_SHORT.get(entry.issuer, entry.issuer)
    kind = PRODUCT_LABEL.get(entry.product_type, entry.product_type.replace("_", " "))
    unders = facts["underlyings"]["value"]
    on = unders[0] if len(unders) == 1 else ", ".join(unders[:-1]) + " and " + unders[-1]
    text = f"the {issuer} {kind} on {on}"
    parts = []
    if "currency" in attrs:
        text = f"the {facts['currency']['value']}-denominated {issuer} {kind} on {on}"
    if "maturity_date" in attrs:
        d = date.fromisoformat(str(facts["maturity_date"]["value"]))
        parts.append(f"maturing in {_MONTHS[d.month - 1]} {d.year}")
    if "coupon_rate" in attrs:
        parts.append(f"paying {facts['coupon_rate']['value']} p.a.")
    if "barrier_level" in attrs:
        parts.append(f"with a {facts['barrier_level']['value']} barrier")
    if "strike_level" in attrs:
        parts.append(f"with a {facts['strike_level']['value']} strike")
    return " ".join([text, *parts]) if parts else text


def disambiguation_drafts(entries: dict[str, DocEntry], facts: dict[str, dict], descriptions: dict[str, dict],
                          *, per_product: int = 2) -> list[dict]:
    """Template questions about fields whose value differs from a near-duplicate sibling's, so a
    retriever that picks the sibling answers wrongly."""
    drafts = []
    for doc_id, d in sorted(descriptions.items()):
        if not d["siblings"] or d["description"] is None:
            continue
        mine = facts[doc_id]
        made = 0
        for field_name, (template, kind) in DISAMBIGUATION_TEMPLATES.items():
            if made >= per_product:
                break
            if field_name in d["attrs"]:   # the description already names it (e.g. "maturing in June 2027")
                continue
            f = mine.get(field_name) or {}
            if not f.get("quote_ok"):
                continue
            ours = _attr_value(mine, field_name) if not field_name.endswith("_date") else f["value"]
            theirs = [(_attr_value(facts[s], field_name) if not field_name.endswith("_date")
                       else (facts[s].get(field_name) or {}).get("value")) for s in d["siblings"]]
            if not any(t is not None and t != ours for t in theirs):
                continue
            drafts.append(_draft(
                task=f"disambiguation:{doc_id}", n=made, corpus=entries[doc_id].corpus, regime="A",
                qtype="disambiguation", doc_ids=[doc_id], question=template, answer=str(f["value"]), kind=kind,
                evidence=[{"doc_id": doc_id, "page": f["page"]}],
                quotes=[{"doc_id": doc_id, "page": f["page"], "text": f["quote"], "ok": True}],
                extra={"field": field_name, "siblings": d["siblings"]}))
            made += 1
    return drafts


# --- draft records --------------------------------------------------------------------------------

def _draft(*, task: str, n: int, corpus: str, regime: str, qtype: str, doc_ids: list[str], question: str,
           answer, kind: str, evidence: list[dict], quotes: list[dict], status: str = "draft",
           reason: str | None = None, extra: dict | None = None) -> dict:
    draft_id = re.sub(r"[^a-z0-9]+", "-", f"{task}-{n}".lower()).strip("-")
    return {"draft_id": draft_id, "task": task, "status": status, "corpus": corpus, "regime": regime,
            "qtype": qtype, "doc_ids": doc_ids, "question": question, "answer": answer, "kind": kind,
            "evidence": evidence, "quotes": quotes, "prompt_version": PROMPT_VERSION,
            **({"reason": reason} if reason else {}), **(extra or {})}


def focus_for(doc_id: str, seed: int) -> str:
    rng = random.Random(f"{seed}:focus:{doc_id}")
    picks = "; ".join(f"{qtype}: {rng.choice(topics)}" for qtype, topics in FOCUS.items() if qtype != "unanswerable")
    return ("- For variety, suggested topics (use them if these pages support them, otherwise pick another "
            f"specific term; do not ask about the barrier level, coupon rate or maturity date): {picks}.\n")


def unanswerable_focus_for(doc_id: str, window: int, seed: int) -> str:
    rng = random.Random(f"{seed}:focus-u:{doc_id}:{window}")
    return (f"- For variety, aim at: {rng.choice(FOCUS['unanswerable'])}. Do not ask about the issuer's "
            "credit rating.\n")


_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUM_PREFIX = re.compile(r"^((?:[A-Z]{3}\s?)?-?\d[\d.,']*\s?%?)(?:\s+(?:of|per)\b.*)?$")


def normalise_answer(kind, answer) -> tuple[str, object]:
    """Coerce the model's (kind, answer) into what the scorer expects; fall back to free text."""
    kind = kind if kind in KINDS else "free"
    if isinstance(answer, list):
        if all(isinstance(a, str) and _ISO.fullmatch(a.strip()) for a in answer) and answer:
            return "date_list", [a.strip() for a in answer]
        return "free", "; ".join(map(str, answer))
    text = str(answer if answer is not None else "").strip()
    parts = [x.strip() for x in re.split(r"[,;]\s*|\s+and\s+", text) if x.strip()]
    if kind in ("date_list", "free") and len(parts) > 1 and all(_ISO.fullmatch(x) for x in parts):
        return "date_list", parts
    if kind in ("numeric", "exact"):
        m = _NUM_PREFIX.match(text)
        if m:
            return "numeric", m.group(1).strip()
        if kind == "numeric":
            return "free", text
    if kind == "date" and not _ISO.fullmatch(text):
        return "free", text
    if kind == "date_list":
        return "free", text
    return kind, text


def _check_window_item(item: dict, doc: ParsedDoc, allowed_pages: set[int]) -> tuple[list[int], list[dict], str | None]:
    pages = sorted({p for p in item.get("evidence") or [] if isinstance(p, int)})
    if not pages:
        return pages, [], "no evidence pages"
    if not set(pages) <= allowed_pages:
        return pages, [], f"evidence outside the window: {pages}"
    quote = item.get("quote")
    hits = [p for p in pages if quote_on_page(quote, doc.page(p).text)]
    quotes = [{"doc_id": doc.doc_id, "page": hits[0] if hits else pages[0], "text": quote, "ok": bool(hits)}]
    return pages, quotes, None if hits else "quote not found on the evidence pages"


# --- the drafter ------------------------------------------------------------------------------------

def _timed(fn: Callable[[], list[dict]]) -> tuple[list[dict], float]:
    started = time.monotonic()
    return fn(), time.monotonic() - started


@dataclass
class Task:
    key: str
    stage: str
    run: Callable[[], list[dict]]


class Drafter:
    def __init__(self, client: ChatClient, store: Store, entries: dict[str, DocEntry], out_dir: str | Path, *,
                 seed: int = 17, max_tokens: int = 8192, prospectus_windows: int = 14,
                 confirm_max_words: int = 45000) -> None:
        self.client, self.store, self.entries = client, store, entries
        self.out = Path(out_dir)
        self.seed, self.max_tokens = seed, max_tokens
        self.n_windows, self.confirm_max_words = prospectus_windows, confirm_max_words
        self._lock = threading.Lock()
        self.out.mkdir(parents=True, exist_ok=True)

    # infrastructure
    def ask(self, text: str) -> dict:
        completion = self.client.complete([{"role": "user", "content": text}], max_tokens=self.max_tokens,
                                          seed=self.seed)
        if completion.finish_reason == "length":
            raise LLMError(f"{self.client.model}: reply truncated at max_tokens={self.max_tokens}")
        return parse_json(completion.text)

    def done(self) -> set[str]:
        path = self.out / "_done.jsonl"
        if not path.exists():
            return set()
        return {json.loads(ln)["task"] for ln in path.read_text().splitlines() if ln.strip()}

    def _append(self, name: str, records: list[dict]) -> None:
        with open(self.out / name, "a") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def run(self, tasks: list[Task], *, workers: int = 4, log=print) -> int:
        finished = self.done()
        todo = [t for t in tasks if t.key not in finished]
        log(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run")
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_timed, t.run): t for t in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                t = futures[fut]
                try:
                    records, elapsed = fut.result()
                except (LLMError, DraftError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    failures += 1
                    log(f"[{i}/{len(todo)}] {t.key}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
                    continue
                with self._lock:
                    file = "facts.jsonl" if t.stage == "facts" else f"{self._corpus_of(t)}.jsonl"
                    self._append(file, records)
                    self._append("_done.jsonl", [{"task": t.key, "records": len(records)}])
                kept = sum(1 for r in records if r.get("status") == "draft")
                log(f"[{i}/{len(todo)}] {t.key}: {len(records)} records, {kept} drafts, {elapsed:.0f}s")
        return failures

    def _corpus_of(self, task: Task) -> str:
        doc_id = task.key.split(":")[1]
        return self.entries[doc_id].corpus

    # loaded artefacts
    def facts(self) -> dict[str, dict]:
        path = self.out / "facts.jsonl"
        if not path.exists():
            return {}
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        return {r["doc_id"]: r["facts"] for r in rows}

    def regime(self, regime: str) -> list[str]:
        return sorted(d for d, e in self.entries.items() if e.regime == regime and d in self.store.docs)

    # stages
    def facts_tasks(self) -> list[Task]:
        def make(doc_id: str) -> Callable[[], list[dict]]:
            def run() -> list[dict]:
                doc = self.store.get(doc_id)
                raw = self.ask(prompt("facts", doc_id=doc_id, n_pages=doc.n_pages,
                                      pages=render_pages(doc, key_pages(doc))))
                return [{"doc_id": doc_id, "facts": check_facts(raw, doc), "prompt_version": PROMPT_VERSION}]
            return run
        return [Task(f"facts:{d}", "facts", make(d)) for d in self.regime("A")]

    def _windows(self, doc_id: str) -> list[list[int]]:
        doc = self.store.get(doc_id)
        if self.entries[doc_id].regime == "A":
            return [key_pages(doc)]
        return prospectus_windows(doc, self.n_windows, seed=self.seed)

    def _reference_rule(self, doc_id: str) -> str:
        if self.entries[doc_id].regime == "A":
            return ("- Refer to the product only as {product} (a placeholder that will be replaced by an ISIN or "
                    "a description), e.g. \"What is the barrier level of {product}?\"\n")
        return (f"- Name the document in every question as \"{doc_label(self.entries[doc_id])}\" so the question "
                "identifies it among several prospectuses.\n")

    def window_tasks(self) -> list[Task]:
        def make(doc_id: str, w: int, pages: list[int]) -> Callable[[], list[dict]]:
            def run() -> list[dict]:
                doc, entry = self.store.get(doc_id), self.entries[doc_id]
                reply = self.ask(prompt(
                    "window", doc_label=doc_label(entry), first=pages[0], last=pages[-1], n_pages=doc.n_pages,
                    per_type=1, types="lookup, table, computation", reference_rule=self._reference_rule(doc_id),
                    focus=focus_for(doc_id, self.seed) if entry.regime == "A" else "",
                    pages=render_pages(doc, pages)))
                out = []
                for n, item in enumerate(reply.get("questions") or []):
                    qtype = item.get("qtype")
                    if qtype not in ("lookup", "table", "computation") or not item.get("question"):
                        continue
                    ev, quotes, problem = _check_window_item(item, doc, set(pages))
                    if entry.regime == "A" and "{product}" not in item["question"]:
                        problem = problem or "question does not use the {product} placeholder"
                    kind, answer = normalise_answer(item.get("kind"), item.get("answer"))
                    out.append(_draft(task=f"window:{doc_id}:{w}", n=n, corpus=entry.corpus, regime=entry.regime,
                                      qtype=qtype, doc_ids=[doc_id], question=item["question"].strip(),
                                      answer=answer, kind=kind,
                                      evidence=[{"doc_id": doc_id, "page": p} for p in ev], quotes=quotes,
                                      status="auto_rejected" if problem else "draft", reason=problem,
                                      extra={"window": pages}))
                return out
            return run
        tasks = []
        for doc_id in self.regime("A") + self.regime("B"):
            for w, pages in enumerate(self._windows(doc_id)):
                tasks.append(Task(f"window:{doc_id}:{w}", "window", make(doc_id, w, pages)))
        return tasks

    def _confirm_scope(self, doc: ParsedDoc, query: str) -> tuple[list[int], str]:
        words = sum(len(p.text.split()) for p in doc.pages)
        if words <= self.confirm_max_words:
            return [p.number for p in doc.pages if p.text.strip()], "the whole document"
        bm25 = BM25([p.text for p in doc.pages])
        best = sorted(p + 1 for p in top_k(bm25.scores(query), 15))
        return best, f"the 15 of {doc.n_pages} pages that best match the question"

    def unanswerable_tasks(self, round_: int = 1) -> list[Task]:
        """Round n > 1 draws a different topic nudge and gets its own task keys."""
        suffix = f":r{round_}" if round_ > 1 else ""
        def make(doc_id: str, w: int, pages: list[int]) -> Callable[[], list[dict]]:
            def run() -> list[dict]:
                doc, entry = self.store.get(doc_id), self.entries[doc_id]
                reply = self.ask(prompt("unanswerable", doc_label=doc_label(entry), first=pages[0], last=pages[-1],
                                        n_pages=doc.n_pages, n=1, reference_rule=self._reference_rule(doc_id),
                                        focus=unanswerable_focus_for(doc_id, w, self.seed + 1000 * (round_ - 1)),
                                        pages=render_pages(doc, pages)))
                out = []
                for n, item in enumerate(reply.get("questions") or []):
                    question = (item.get("question") or "").strip()
                    if not question:
                        continue
                    keywords = [k for k in item.get("keywords") or [] if isinstance(k, str)]
                    scope_pages, scope = self._confirm_scope(doc, question + " " + " ".join(keywords))
                    check = self.ask(prompt("confirm_absent", question=question.replace("{product}", "this product"),
                                            scope=scope, pages=render_pages(doc, scope_pages)))
                    problem = None
                    if entry.regime == "A" and "{product}" not in question:
                        problem = "question does not use the {product} placeholder"
                    elif check.get("answered") is not False:
                        problem = f"checker found an answer on p.{check.get('page')}: {check.get('quote')!r}"
                    out.append(_draft(task=f"unanswerable:{doc_id}:{w}{suffix}", n=n, corpus=entry.corpus,
                                      regime=entry.regime, qtype="unanswerable", doc_ids=[doc_id], question=question,
                                      answer=None, kind="unanswerable", evidence=[], quotes=[],
                                      status="auto_rejected" if problem else "draft", reason=problem,
                                      extra={"keywords": keywords, "why_absent": item.get("why_absent"),
                                             "checked": scope}))
                return out
            return run
        tasks = []
        for doc_id in self.regime("A") + self.regime("B"):
            windows = self._windows(doc_id)
            if self.entries[doc_id].regime == "B":
                windows = windows[::2]   # every other window: unanswerable needs ~10% of the questions
            for w, pages in enumerate(windows):
                tasks.append(Task(f"unanswerable:{doc_id}:{w}{suffix}", "unanswerable", make(doc_id, w, pages)))
        return tasks

    def linked_notes(self, doc_id: str) -> list[str]:
        """The securities note(s) in the corpus that this termsheet's final terms complete."""
        entry, doc = self.entries[doc_id], self.store.get(doc_id)
        notes = [e for e in self.entries.values()
                 if e.regime == "B" and e.issuer == entry.issuer and e.product_type == "securities_note"]
        if len(notes) <= 1:
            return [e.doc_id for e in notes if e.doc_id in self.store.docs]
        # The date that follows "securities note" (a registration document of the same date may also be cited).
        opening = _norm(" ".join(p.text for p in doc.pages[:2]))
        m = re.search(r"securities note[^.]{0,200}?(?:dated|approved on) (\d{1,2} [a-z]+ \d{4})", opening)
        if not m:
            return []
        dated = [e for e in notes if e.issue_date and _norm(human_date(e.issue_date)) == m.group(1)]
        return [e.doc_id for e in dated if e.doc_id in self.store.docs]

    def cross_doc_tasks(self, round_: int = 1) -> list[Task]:
        """Round 1 uses the first two provisions the final terms rely on; round n the next two."""
        def make(ts_id: str, sn_id: str) -> Callable[[], list[dict]]:
            def run() -> list[dict]:
                ts, sn, entry = self.store.get(ts_id), self.store.get(sn_id), self.entries[ts_id]
                pages = key_pages(ts)
                terms = self.ask(prompt("cross_doc_terms", pages=render_pages(ts, pages)))
                provisions = [p for p in terms.get("provisions") or [] if isinstance(p, dict)][2 * (round_ - 1):2 * round_]
                if not provisions:
                    return []
                bm25 = BM25([p.text for p in sn.pages])
                sn_pages: list[int] = []
                ts_pages: list[int] = []
                for prov in provisions:
                    query = " ".join([prov.get("wording") or "", *(prov.get("keywords") or [])])
                    sn_pages += [i + 1 for i in top_k(bm25.scores(query), 3) if i + 1 not in sn_pages]
                    if isinstance(prov.get("page"), int) and prov["page"] in pages and prov["page"] not in ts_pages:
                        ts_pages.append(prov["page"])
                ts_pages = sorted(ts_pages or pages[:3])
                reply = self.ask(prompt("cross_doc", ts_id=ts_id, sn_id=sn_id, n=2,
                                        ts_pages=render_pages(ts, ts_pages), sn_pages=render_pages(sn, sorted(sn_pages))))
                docs = {ts_id: ts, sn_id: sn}
                out = []
                for n, item in enumerate(reply.get("questions") or []):
                    question = (item.get("question") or "").strip()
                    if not question:
                        continue
                    evidence = [e for e in item.get("evidence") or [] if isinstance(e, dict)
                                and e.get("doc_id") in docs and isinstance(e.get("page"), int)
                                and 1 <= e["page"] <= docs[e["doc_id"]].n_pages]
                    quotes = [{**q, "ok": quote_on_page(q.get("text"), docs[q["doc_id"]].page(q["page"]).text)}
                              for q in item.get("quotes") or [] if isinstance(q, dict) and q.get("doc_id") in docs
                              and isinstance(q.get("page"), int) and 1 <= q["page"] <= docs[q["doc_id"]].n_pages]
                    problem = None
                    if "{product}" not in question:
                        problem = "question does not use the {product} placeholder"
                    elif {e["doc_id"] for e in evidence} != set(docs):
                        problem = "evidence does not cover both documents"
                    elif not all(any(q["ok"] and q["doc_id"] == d for q in quotes) for d in docs):
                        problem = "a quote was not found on its page"
                    kind, answer = normalise_answer(item.get("kind"), item.get("answer"))
                    out.append(_draft(task=f"cross_doc:{ts_id}:{sn_id}" + (f":r{round_}" if round_ > 1 else ""), n=n, corpus=entry.corpus, regime="A",
                                      qtype="cross_doc", doc_ids=[ts_id, sn_id], question=question,
                                      answer=answer, kind=kind,
                                      evidence=[{"doc_id": e["doc_id"], "page": e["page"]} for e in evidence],
                                      quotes=quotes, status="auto_rejected" if problem else "draft", reason=problem,
                                      extra={"provisions": provisions}))
                return out
            return run
        tasks = []
        for ts_id in self.regime("A"):
            for sn_id in self.linked_notes(ts_id):
                key = f"cross_doc:{ts_id}:{sn_id}" + (f":r{round_}" if round_ > 1 else "")
                tasks.append(Task(key, "cross_doc", make(ts_id, sn_id)))
        return tasks

    def write_descriptions_and_disambiguation(self, log=print) -> None:
        facts = self.facts()
        descriptions = describe_all(self.entries, facts)
        (self.out / "descriptions.json").write_text(json.dumps(descriptions, indent=1, ensure_ascii=False))
        drafts = disambiguation_drafts(self.entries, facts, descriptions)
        # Deterministic: rewrite this stage's drafts wholesale instead of appending.
        by_corpus: dict[str, list[dict]] = {}
        for d in drafts:
            by_corpus.setdefault(d["corpus"], []).append(d)
        for corpus in {e.corpus for e in self.entries.values() if e.regime == "A"}:
            path = self.out / f"{corpus}.jsonl"
            kept = [ln for ln in (path.read_text().splitlines() if path.exists() else [])
                    if ln.strip() and json.loads(ln)["qtype"] != "disambiguation"]
            path.write_text("".join(ln + "\n" for ln in kept)
                            + "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in by_corpus.get(corpus, [])))
        unique = sum(1 for d in descriptions.values() if d["description"])
        log(f"descriptions: {unique}/{len(descriptions)} unique; disambiguation drafts: {len(drafts)}")


# --- CLI ------------------------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    from aex.experiments.tree_rag.run import build_client, load_store

    ap = argparse.ArgumentParser(prog="draft")
    ap.add_argument("config")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--limit", type=int, default=None, help="run at most N tasks per stage (trial runs)")
    ap.add_argument("--docs", default=None, help="regex: only tasks whose key matches (trial runs)")
    ap.add_argument("--round", type=int, default=1,
                    help="cross_doc: the next pair of provisions; unanswerable: a new topic nudge")
    args = ap.parse_args(argv[1:])
    cfg = yaml.safe_load(Path(args.config).read_text())
    excluded = ExclusionList.load()
    entries = {e.doc_id: e for m in cfg["manifests"] for e in load_manifest(m, excluded=excluded)}
    store = load_store(cfg["parsed_dir"])
    store.docs = {d: doc for d, doc in store.docs.items() if d in entries}
    client = build_client(cfg["model"], cfg["models"][cfg["model"]])
    drafter = Drafter(client, store, entries, cfg["out_dir"], seed=cfg.get("seed", 17),
                      max_tokens=cfg.get("max_tokens", 8192), prospectus_windows=cfg.get("prospectus_windows", 14))
    failures = 0
    for stage in args.stages.split(","):
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
        print(f"== {stage}", flush=True)
        if stage == "disambiguation":
            drafter.write_descriptions_and_disambiguation()
            continue
        tasks = (drafter.cross_doc_tasks(args.round) if stage == "cross_doc"
                 else drafter.unanswerable_tasks(args.round) if stage == "unanswerable"
                 else getattr(drafter, f"{stage}_tasks")())
        if args.docs:
            tasks = [t for t in tasks if re.search(args.docs, t.key)]
        failures += drafter.run(tasks[:args.limit] if args.limit else tasks, workers=cfg.get("workers", 4),
                                log=lambda s: print(s, flush=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

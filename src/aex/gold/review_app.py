"""Review drafted questions from a phone and promote them to gold (plan 1b, Task 7).

Usage: python -m aex.gold.review_app configs/gold/draft.yaml [--host 100.84.235.48] [--port 8765]

Serves one draft per screen over Tailscale only: the question (for regime A in both forms, by ISIN
and by description), the drafted answer and kind (editable), and every evidence page as an image
beside its parsed text with the drafted quote marked. Accept / Reject / Skip; Undo reverts the last
decision.

Decisions are appended to gold/review/decisions.jsonl (the source of truth; nothing is lost on a
crash). After every decision the gold files gold/<corpus>.jsonl are rebuilt from the drafts and the
latest decision per draft, each record validated against gold.v1 before anything is written.
"""
from __future__ import annotations

import argparse
import html
import ipaddress
import json
import re
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import jsonschema

from aex.corpus.manifest import DocEntry
from aex.experiments.tree_rag.gold import _SCHEMA
from aex.experiments.tree_rag.types import ParsedDoc

TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)
QTYPES = ("lookup", "table", "computation", "cross_doc", "disambiguation", "unanswerable")
KINDS = ("numeric", "date", "date_list", "exact", "free", "unanswerable")
TARGET_PER_TYPE = 40   # spec §5.5: every type >= 40 verified (plan Task 8 checks the full quotas)


class ReviewError(ValueError):
    pass


def check_bind_address(host: str) -> None:
    """Only a Tailscale address: the review page has no login, so it must not face any other network."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ReviewError(f"bind address must be an IP literal, got {host!r}") from exc
    if addr not in TAILSCALE_NET:
        raise ReviewError(f"refusing to bind to {host}: not a Tailscale address (100.64.0.0/10)")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_question(template: str, reference: str) -> str:
    text = template.replace("{product}", reference)
    return text[0].upper() + text[1:] if template.startswith("{product}") else text


def parse_evidence(text: str) -> list[dict]:
    """'ts-1:3, sn-1:41' -> [{'doc_id': 'ts-1', 'page': 3}, ...]"""
    out = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        doc_id, _, page = part.rpartition(":")
        if not doc_id or not page.isdigit():
            raise ReviewError(f"evidence {part!r}: expected doc_id:page")
        out.append({"doc_id": doc_id, "page": int(page)})
    return out


def parse_answer(kind: str, text: str):
    if kind == "unanswerable":
        return None
    if kind == "date_list":
        return [x.strip() for x in re.split(r"[,;\n]", text) if x.strip()]
    return text.strip()


def gold_records(draft: dict, edits: dict, *, isin: str | None, description: str | None,
                 verified_by: str = "daniel") -> list[dict]:
    """The gold.v1 record(s) for an accepted draft. Regime A gives an ISIN form and, when the product
    has a unique description, a description form, sharing a pair_id."""
    kind = edits["kind"]
    base = {"dataset": draft["corpus"], "regime": draft["regime"], "qtype": edits.get("qtype", draft["qtype"]),
            "doc_ids": draft["doc_ids"], "gold": {"kind": kind, "value": parse_answer(kind, edits["answer"])},
            "evidence": [] if kind == "unanswerable" else edits["evidence"], "verified_by": verified_by}
    question = edits["question"].strip()
    if draft["regime"] != "A":
        return [{"qid": draft["draft_id"], "question": question, **base}]
    if "{product}" not in question:
        raise ReviewError("regime-A questions must keep the {product} placeholder")
    if not isin:
        raise ReviewError(f"{draft['doc_ids'][0]} has no ISIN in the manifest")
    pair = draft["draft_id"]
    records = [{"qid": f"{pair}-isin", "question": render_question(question, f"the product with ISIN {isin}"),
                "pair_id": pair, "form": "isin", **base}]
    if description:
        records.append({"qid": f"{pair}-desc", "question": render_question(question, description),
                        "pair_id": pair, "form": "description", **base})
    return records


def validate(record: dict) -> None:
    errors = sorted(_VALIDATOR.iter_errors(record), key=lambda e: e.path)
    if errors:
        raise ReviewError(f"{record['qid']}: {errors[0].message}")
    unanswerable = record["gold"]["kind"] == "unanswerable"
    if unanswerable == bool(record["evidence"]):
        raise ReviewError(f"{record['qid']}: " + ("unanswerable needs empty evidence" if unanswerable
                                                    else "answerable needs evidence pages"))


class Review:
    """Drafts + decisions -> what to show next, and the gold files."""

    def __init__(self, drafts_dir: str | Path, gold_dir: str | Path, entries: dict[str, DocEntry],
                 docs: dict[str, ParsedDoc]) -> None:
        self.drafts_dir, self.gold_dir = Path(drafts_dir), Path(gold_dir)
        self.entries, self.docs = entries, docs
        self.decisions_path = self.gold_dir / "review" / "decisions.jsonl"
        self.lock = threading.Lock()
        self.skipped: list[str] = []
        self.reload()

    def _stamp(self) -> tuple:
        return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(self.drafts_dir.glob("*.json*")))

    def refresh(self) -> None:
        """Pick up drafts written since the last look (drafting may still be running)."""
        if self._stamp() != self._loaded:
            self.reload()

    def reload(self) -> None:
        self._loaded = self._stamp()
        self.drafts: dict[str, dict] = {}
        for path in sorted(self.drafts_dir.glob("*.jsonl")):
            if path.name.startswith("_") or path.name == "facts.jsonl":
                continue
            for ln in path.read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    if d.get("status") == "draft":
                        self.drafts[d["draft_id"]] = d
        desc_path = self.drafts_dir / "descriptions.json"
        self.descriptions = json.loads(desc_path.read_text()) if desc_path.exists() else {}

    def decisions(self) -> dict[str, dict]:
        """Latest decision per draft; an undo removes the draft's decision."""
        latest: dict[str, dict] = {}
        log: list[str] = []
        if self.decisions_path.exists():
            for ln in self.decisions_path.read_text().splitlines():
                if not ln.strip():
                    continue
                d = json.loads(ln)
                if d["action"] == "undo":
                    if log:
                        latest.pop(log.pop(), None)
                    continue
                latest[d["draft_id"]] = d
                log.append(d["draft_id"])
        return latest

    def description(self, draft: dict) -> str | None:
        return (self.descriptions.get(draft["doc_ids"][0]) or {}).get("description")

    def isin(self, draft: dict) -> str | None:
        entry = self.entries.get(draft["doc_ids"][0])
        return entry.isin if entry and entry.isin else None

    def counts(self) -> Counter:
        decided = self.decisions()
        return Counter((self.drafts[d]["regime"], self.drafts[d]["qtype"]) for d, x in decided.items()
                       if x["action"] == "accept" and d in self.drafts)

    def next_item(self) -> dict | None:
        """The undecided draft of the (regime, type) furthest below target; skipped drafts go last."""
        self.refresh()
        decided = self.decisions()
        open_ = [d for d in self.drafts.values() if d["draft_id"] not in decided]
        if not open_:
            return None
        counts = self.counts()
        skipped = {s: i for i, s in enumerate(self.skipped)}
        return min(open_, key=lambda d: (d["draft_id"] in skipped, skipped.get(d["draft_id"], 0),
                                         counts[(d["regime"], d["qtype"])], d["draft_id"]))

    def decide(self, draft_id: str, action: str, edits: dict | None = None, note: str = "") -> None:
        with self.lock:
            if action == "skip":
                if draft_id in self.skipped:
                    self.skipped.remove(draft_id)
                self.skipped.append(draft_id)
                return
            if action not in ("accept", "reject", "undo"):
                raise ReviewError(f"unknown action {action!r}")
            entry = {"draft_id": draft_id, "action": action, "at": _now()}
            if action == "accept":
                draft = self.drafts[draft_id]
                records = gold_records(draft, edits, isin=self.isin(draft), description=self.description(draft))
                for r in records:
                    validate(r)
                entry["edits"] = edits
            if note:
                entry["note"] = note
            self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.decisions_path, "a") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.write_gold()

    def write_gold(self) -> dict[str, int]:
        by_corpus: dict[str, list[dict]] = {}
        for draft_id, d in sorted(self.decisions().items()):
            if d["action"] != "accept" or draft_id not in self.drafts:
                continue
            draft = self.drafts[draft_id]
            # Decisions made outside this app (aex.gold.ai_review) carry their own verifier.
            for r in gold_records(draft, d["edits"], isin=self.isin(draft), description=self.description(draft),
                                  verified_by=d.get("verifier", "daniel")):
                validate(r)
                by_corpus.setdefault(draft["corpus"], []).append(r)
        for corpus in {d["corpus"] for d in self.drafts.values()}:
            records = by_corpus.get(corpus, [])
            path = self.gold_dir / f"{corpus}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        return {c: len(r) for c, r in by_corpus.items()}


# --- page images ------------------------------------------------------------------------------------

def page_png(pdf: Path, page: int, cache_dir: Path, *, scale: float = 1.6) -> bytes:
    target = cache_dir / pdf.stem / f"{page}.png"
    if not target.exists():
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf))
        try:
            if not 1 <= page <= len(doc):
                raise ReviewError(f"{pdf.name} has no page {page}")
            image = doc[page - 1].render(scale=scale).to_pil()
        finally:
            doc.close()
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, optimize=True)
    return target.read_bytes()


# --- HTML -------------------------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box} body{font:16px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:12px 16px 120px;
color:#1a1a1a;background:#faf8f3;max-width:860px;margin:auto} h1{font-size:15px;margin:0 0 8px;color:#555}
.q{font-size:17px;font-weight:600;margin:4px 0 6px} .form{font-size:12px;color:#666;text-transform:uppercase;letter-spacing:.04em}
.box{background:#fff;border:1px solid #ddd6c8;border-radius:8px;padding:10px 12px;margin:10px 0}
label{display:block;font-size:13px;color:#555;margin-top:8px} input,select,textarea{width:100%;font:inherit;
padding:8px;border:1px solid #bbb;border-radius:6px;background:#fff} textarea{min-height:64px}
.bar{position:fixed;left:0;right:0;bottom:0;background:#fffdf8;border-top:1px solid #ddd6c8;padding:10px 16px;
display:flex;gap:8px;max-width:860px;margin:auto} .bar button{flex:1;min-height:52px;font-size:17px;border:0;
border-radius:8px;color:#fff} .ok{background:#2f6b3a}.no{background:#9b2c2c}.skip{background:#777}
.page img{width:100%;border:1px solid #ccc} .text{white-space:pre-wrap;font-size:14px;max-height:340px;overflow:auto;
background:#fdfcf9;border:1px solid #eee;padding:8px} mark{background:#ffe27a} .meta{font-size:13px;color:#555}
.prog{font-size:12px;color:#555} .prog b{color:#1a1a1a} .err{background:#fde8e8;border-color:#e0a0a0}
details summary{cursor:pointer;color:#555;font-size:14px;padding:6px 0} .full{font-size:12px} a{color:#2b4f8a}
"""


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _tidy(text: str) -> str:
    """Display only: markdown table rules and runs of blank cells are noise on a phone."""
    text = re.sub(r"-{4,}", "—", text)
    return re.sub(r"(\|\s*){3,}", "| ", text)


def _find_quote(text: str, quote_text: str | None) -> tuple[int, int] | None:
    words = re.findall(r"\w+", quote_text or "")[:12]
    if len(words) < 3:
        return None
    m = re.search(r"\W+".join(map(re.escape, words)), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def excerpt_html(text: str, quote_text: str | None, *, context: int = 450) -> tuple[str, bool]:
    """The page text around the quote with the quote marked; (whole page from the top, False) if not found."""
    text = _tidy(text)
    span = _find_quote(text, quote_text)
    if span is None:
        return _esc(text[:2 * context]) + (" …" if len(text) > 2 * context else ""), False
    a, b = span
    lo, hi = max(0, a - context), min(len(text), b + context)
    return ((" … " if lo else "") + _esc(text[lo:a]) + "<mark>" + _esc(text[a:b]) + "</mark>"
            + _esc(text[b:hi]) + (" …" if hi < len(text) else "")), True


def progress_html(review: Review) -> str:
    counts = review.counts()
    regimes = sorted({d["regime"] for d in review.drafts.values()})
    cells = []
    for regime in regimes:
        parts = [f"{t.replace('_', ' ')} <b>{counts[(regime, t)]}</b>" for t in QTYPES
                 if any(d["regime"] == regime and d["qtype"] == t for d in review.drafts.values())]
        cells.append(f"<div>Regime {regime}: {' · '.join(parts)}</div>")
    decided = review.decisions()
    left = sum(1 for d in review.drafts if d not in decided)
    return f'<div class="prog">{"".join(cells)}<div>{left} drafts left · target ≥{TARGET_PER_TYPE} per type</div></div>'


def item_html(review: Review, draft: dict, error: str = "") -> str:
    regime_a = draft["regime"] == "A"
    question = draft["question"]
    if regime_a:
        desc = review.description(draft)
        isin_q = render_question(question, f"the product with ISIN {review.isin(draft)}")
        main = (render_question(question, desc) if desc
                else isin_q + " (no unique description for this product: ISIN form only)")
        shown = (f'<div class="form">by description</div><div class="q">{_esc(main)}</div>'
                 f'<div class="meta">By ISIN: {_esc(isin_q)}</div>')
    else:
        shown = f'<div class="q">{_esc(question)}</div>'
    answer = draft["answer"]
    answer_text = ", ".join(answer) if isinstance(answer, list) else (answer or "")
    kind_options = "".join(f'<option{" selected" if k == draft["kind"] else ""}>{k}</option>' for k in KINDS)
    evidence_text = ", ".join(f"{e['doc_id']}:{e['page']}" for e in draft["evidence"])
    meta = [f"{draft['qtype'].replace('_', ' ')}", f"regime {draft['regime']}", " + ".join(draft["doc_ids"])]
    extra = ""
    if draft["qtype"] == "unanswerable":
        extra = (f'<div class="box meta">Checked absent in {_esc(draft.get("checked"))}. Why absent: '
                 f'{_esc(draft.get("why_absent"))}<br>Keywords: {_esc(", ".join(draft.get("keywords") or []))}</div>')
    if draft["qtype"] == "disambiguation":
        extra = (f'<div class="box meta">Field <b>{_esc(draft.get("field"))}</b> differs from near-duplicates: '
                 f'{_esc(", ".join(draft.get("siblings") or []))}</div>')
    pages = []
    quotes = {(q["doc_id"], q["page"]): q.get("text") for q in draft.get("quotes") or []}
    for e in draft["evidence"]:
        doc = review.docs.get(e["doc_id"])
        text = doc.page(e["page"]).text if doc and 1 <= e["page"] <= doc.n_pages else "(page not in the parse)"
        q_text = quotes.get((e["doc_id"], e["page"]))
        snippet, found = excerpt_html(text, q_text)
        src = f"/page/{quote(e['doc_id'])}/{e['page']}.png"
        pages.append(f'<div class="box page"><div class="meta">{_esc(e["doc_id"])} p.{e["page"]}'
                     f'{"" if found else " · quote not located in the text"}</div>'
                     f'<div class="text">{snippet}</div>'
                     f'<details><summary>Whole page text</summary><div class="text full">{_esc(_tidy(text))}</div></details>'
                     f'<details><summary>Page image</summary><a href="{src}"><img loading="lazy" src="{src}" '
                     f'alt="page {e["page"]}"></a></details></div>')
    placeholder = "keep {product}" if regime_a else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold review</title><style>{CSS}</style></head><body>
<h1>{_esc(" · ".join(meta))} · <span class="meta">{_esc(draft["draft_id"])}</span></h1>
{progress_html(review)}
{f'<div class="box err">{_esc(error)}</div>' if error else ''}
<div class="box">{shown}</div>{extra}
<form id="decide" method="post" action="/decide">
<input type="hidden" name="draft_id" value="{_esc(draft["draft_id"])}">
<div class="box">
<label>Answer</label><textarea name="answer" rows="{max(2, min(8, len(answer_text) // 34 + 1))}">{_esc(answer_text)}</textarea>
<label>Kind</label><select name="kind">{kind_options}</select>
<details><summary>Edit question, type or evidence</summary>
<label>Question template {_esc(placeholder)}</label><textarea name="question">{_esc(question)}</textarea>
<label>Type</label><select name="qtype">{"".join(f'<option{" selected" if t == draft["qtype"] else ""}>{t}</option>' for t in QTYPES)}</select>
<label>Evidence (doc_id:page, …)</label><input name="evidence" value="{_esc(evidence_text)}">
<label>Note (optional)</label><input name="note">
</details></div>
{"".join(pages)}
</form>
<div class="meta"><form method="post" action="/undo" style="display:inline"><button style="font-size:14px;padding:6px 10px">Undo last decision</button></form> · <a href="/">Reload</a></div>
<div class="bar"><button class="no" form="decide" name="action" value="reject">Reject</button>
<button class="skip" form="decide" name="action" value="skip">Skip</button>
<button class="ok" form="decide" name="action" value="accept">Accept</button></div>
</body></html>"""


def done_html(review: Review) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>Gold review</title><style>{CSS}</style></head><body><h1>All drafts decided</h1>'
            f'{progress_html(review)}<div><form method="post" action="/undo"><button style="font-size:14px;padding:6px 10px">Undo last decision</button></form></div></body></html>')


def make_handler(review: Review, pdf_root: Path, cache_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # quiet; decisions are logged to the decisions file
            pass

        def _send(self, body: bytes, ctype: str = "text/html; charset=utf-8", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("text/html") else "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, to: str = "/") -> None:
            self.send_response(303)
            self.send_header("Location", to)
            self.end_headers()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                item = review.next_item()
                return self._send((item_html(review, item) if item else done_html(review)).encode())
            m = re.fullmatch(r"/page/([a-z0-9-]+)/(\d+)\.png", path)
            if m and m.group(1) in review.entries:
                entry = review.entries[m.group(1)]
                try:
                    png = page_png(pdf_root / entry.corpus / f"{entry.doc_id}.pdf", int(m.group(2)), cache_dir)
                except (ReviewError, FileNotFoundError, OSError) as exc:
                    return self._send(str(exc).encode(), "text/plain; charset=utf-8", 404)
                return self._send(png, "image/png")
            m = re.fullmatch(r"/item/([a-z0-9-]+)", path)
            if m and m.group(1) in review.drafts:
                return self._send(item_html(review, review.drafts[m.group(1)]).encode())
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if urlsplit(self.path).path == "/undo":   # POST only: a link prefetch must never undo
                review.decide("", "undo")
                return self._redirect()
            if urlsplit(self.path).path != "/decide":
                return self._send(b"not found", "text/plain", 404)
            length = int(self.headers.get("Content-Length", 0))
            form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode(), keep_blank_values=True).items()}
            draft_id = form.get("draft_id", "")
            if draft_id not in review.drafts:
                return self._send(b"unknown draft", "text/plain", 400)
            try:
                edits = None
                if form.get("action") == "accept":
                    kind = form.get("kind", "free")
                    edits = {"question": form.get("question", ""), "answer": form.get("answer", ""), "kind": kind,
                             "qtype": form.get("qtype") or review.drafts[draft_id]["qtype"],
                             "evidence": [] if kind == "unanswerable" else parse_evidence(form.get("evidence", ""))}
                review.decide(draft_id, form.get("action", ""), edits, note=form.get("note", ""))
            except ReviewError as exc:
                return self._send(item_html(review, review.drafts[draft_id], error=str(exc)).encode(), status=400)
            self._redirect()

    return Handler


def main(argv: list[str]) -> int:
    import yaml

    from aex.corpus.manifest import ExclusionList, load_manifest
    from aex.experiments.tree_rag.run import load_store

    ap = argparse.ArgumentParser(prog="review_app")
    ap.add_argument("config")
    ap.add_argument("--host", default="100.84.235.48")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--gold", default="gold")
    ap.add_argument("--data", default="data")
    args = ap.parse_args(argv[1:])
    check_bind_address(args.host)
    cfg = yaml.safe_load(Path(args.config).read_text())
    excluded = ExclusionList.load()
    entries = {e.doc_id: e for m in cfg["manifests"] for e in load_manifest(m, excluded=excluded)}
    store = load_store(cfg["parsed_dir"])
    review = Review(cfg["out_dir"], args.gold, entries, {d: doc for d, doc in store.docs.items() if d in entries})
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(review, Path(args.data), Path(args.data) / "review-cache"))
    print(f"review app on http://{args.host}:{args.port}/ ({len(review.drafts)} drafts)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

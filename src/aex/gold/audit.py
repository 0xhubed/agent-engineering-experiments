"""Daniel's audit of the model-verified gold (decided 2026-10-07).

Usage:
  python -m aex.gold.audit sample [--n 60]    # stratified random sample of the model's accepts
  python -m aex.gold.audit serve              # grade on the phone: http://100.84.235.48:8767/
  python -m aex.gold.audit report             # -> gold/audit.json (error rate, Wilson 95% CI)

The model reviewed every draft (aex.gold.ai_review, verified_by: claude-opus-5.5). Daniel grades a random
sample of its accepted items as Correct / Wrong / Unsure without seeing the model's reasons; the error rate of
the model-verified gold, with its confidence interval, is reported alongside the results.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from aex.gold.review_app import CSS, _tidy, check_bind_address, excerpt_html, page_png

REVIEW = Path("gold/review")
QUEUE, GRADES = REVIEW / "audit_queue.jsonl", REVIEW / "audit_grades.jsonl"
MODEL_VERIFIER = "claude-opus-5.5"
ACTIONS = ("correct", "wrong", "unsure")


def model_accepts(gold_paths: list[Path]) -> list[dict]:
    """Model-verified gold questions, one per pair (the description form when there is one)."""
    by_key: dict[str, dict] = {}
    for path in gold_paths:
        for ln in path.read_text().splitlines() if path.exists() else []:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["verified_by"] != MODEL_VERIFIER:
                continue
            key = r.get("pair_id") or r["qid"]
            if key not in by_key or r.get("form") == "description":
                by_key[key] = r
    return [by_key[k] for k in sorted(by_key)]


def sample(records: list[dict], *, n: int = 60, seed: int = 17, min_per_type: int = 5) -> list[dict]:
    """Stratified by (regime, qtype): at least `min_per_type` from each stratum (if it has that many), the rest
    proportional to stratum size."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        strata[(r["regime"], r["qtype"])].append(r)
    for items in strata.values():
        rng.shuffle(items)
    take = {k: min(min_per_type, len(v)) for k, v in strata.items()}
    remaining = n - sum(take.values())
    total = sum(len(v) for v in strata.values())
    for k, v in sorted(strata.items()):
        extra = round(remaining * len(v) / total) if total and remaining > 0 else 0
        take[k] = min(len(v), take[k] + extra)
    picked = [r for k, v in sorted(strata.items()) for r in v[:take[k]]]
    rng.shuffle(picked)
    return picked


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def latest(path: Path) -> dict[str, str]:
    grades: dict[str, str] = {}
    order: list[str] = []
    for ln in path.read_text().splitlines() if path.exists() else []:
        if not ln.strip():
            continue
        g = json.loads(ln)
        if g["action"] == "undo":
            if order:
                grades.pop(order.pop(), None)
            continue
        grades[g["qid"]] = g["action"]
        order.append(g["qid"])
    return grades


def report(queue: list[dict], grades: dict[str, str]) -> dict:
    graded = [q for q in queue if q["qid"] in grades]
    wrong = [q["qid"] for q in graded if grades[q["qid"]] == "wrong"]
    decided = sum(1 for q in graded if grades[q["qid"]] in ("correct", "wrong"))
    lo, hi = wilson(len(wrong), decided)
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for q in graded:
        if grades[q["qid"]] in ("correct", "wrong"):
            by_type[q["qtype"]][0] += grades[q["qid"]] == "wrong"
            by_type[q["qtype"]][1] += 1
    return {"sampled": len(queue), "graded": len(graded), "decided": decided, "wrong": len(wrong),
            "unsure": sum(1 for q in graded if grades[q["qid"]] == "unsure"),
            "error_rate": round(len(wrong) / decided, 4) if decided else None,
            "error_rate_ci95": [round(lo, 4), round(hi, 4)], "by_type": {t: {"wrong": w, "n": n} for t, (w, n) in sorted(by_type.items())},
            "wrong_qids": wrong}


# --- phone page ------------------------------------------------------------------------------------------

class Audit:
    def __init__(self, queue: list[dict], grades_path: Path, page_text) -> None:
        self.queue, self.grades_path, self.page_text = queue, grades_path, page_text
        self.skipped: list[str] = []
        self.lock = threading.Lock()

    def next_item(self) -> dict | None:
        grades = latest(self.grades_path)
        open_ = [q for q in self.queue if q["qid"] not in grades]
        return min(open_, key=lambda q: (q["qid"] in self.skipped, self.queue.index(q))) if open_ else None

    def record(self, qid: str, action: str, note: str = "") -> None:
        with self.lock:
            if action == "skip":
                self.skipped.append(qid)
                return
            if action not in (*ACTIONS, "undo"):
                raise ValueError(action)
            self.grades_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.grades_path, "a") as fh:
                fh.write(json.dumps({"qid": qid, "action": action, "note": note,
                                     "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}) + "\n")


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def audit_html(a: Audit) -> str:
    q = a.next_item()
    done = len(latest(a.grades_path))
    head = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>Gold audit</title><style>{CSS}</style></head><body>'
            f'<h1>Gold audit · {done}/{len(a.queue)} graded</h1>')
    undo = ('<div class="meta"><form method="post" action="/undo" style="display:inline"><button '
            'style="font-size:14px;padding:6px 10px">Undo last grade</button></form></div>')
    if q is None:
        return head + "<p>All sampled items graded. Tell Claude to run the audit report.</p>" + undo + "</body></html>"
    value = q["gold"]["value"]
    value = "; ".join(value) if isinstance(value, list) else (value if value is not None else "Not stated in the document")
    pages = []
    for e in q["evidence"]:
        text = a.page_text(e["doc_id"], e["page"])
        snippet, found = excerpt_html(text, value, context=700)   # mark the answer when it is a phrase on the page
        if not found:
            snippet = _esc(_tidy(text))
        src = f"/page/{quote(e['doc_id'])}/{e['page']}.png"
        pages.append(f'<div class="box page"><div class="meta">{_esc(e["doc_id"])} p.{e["page"]}</div>'
                     f'<details open><summary>Page text</summary><div class="text">{snippet}</div></details>'
                     f'<details><summary>Page image</summary><a href="{src}"><img loading="lazy" src="{src}" '
                     f'alt="page {e["page"]}"></a></details></div>')
    if not q["evidence"]:
        pages.append('<div class="box meta">Unanswerable: the model checked that the document does not state this. '
                     'Judge whether that is plausible and the question is fair.</div>')
    return head + f"""
<div class="box"><div class="form">{_esc(q["qtype"].replace("_", " "))} · regime {_esc(q["regime"])}</div>
<div class="q">{_esc(q["question"])}</div></div>
<div class="box"><div class="form">gold answer ({_esc(q["gold"]["kind"])}) — is it right, and is the question fair?</div><div>{_esc(value)}</div></div>
<form id="grade" method="post" action="/grade"><input type="hidden" name="qid" value="{_esc(q["qid"])}">
<div class="box"><label>Note (optional, e.g. what is wrong)</label><input name="note"></div></form>
{"".join(pages)}
{undo}
<div class="bar"><button class="no" form="grade" name="action" value="wrong">Wrong</button>
<button class="skip" form="grade" name="action" value="unsure">Unsure</button>
<button class="ok" form="grade" name="action" value="correct">Correct</button></div>
</body></html>"""


def make_handler(a: Audit, pdf_for):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body: bytes, ctype: str = "text/html; charset=utf-8", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self._send(audit_html(a).encode())
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "page" and parts[2].endswith(".png") and parts[2][:-4].isdigit():
                try:
                    return self._send(page_png(pdf_for(parts[1]), int(parts[2][:-4]), Path("data/review-cache")),
                                      "image/png")
                except (KeyError, OSError, ValueError) as exc:
                    return self._send(str(exc).encode(), "text/plain", 404)
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode(), keep_blank_values=True).items()}
            if path == "/undo":
                a.record("", "undo")
                return self._redirect()
            if path != "/grade" or form.get("qid") not in {q["qid"] for q in a.queue}:
                return self._send(b"bad request", "text/plain", 400)
            try:
                a.record(form["qid"], form.get("action", ""), form.get("note", ""))
            except ValueError:
                return self._send(b"bad action", "text/plain", 400)
            self._redirect()

    return Handler


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="audit")
    ap.add_argument("command", choices=["sample", "serve", "report"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--host", default="100.84.235.48")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args(argv[1:])
    if args.command == "sample":
        if QUEUE.exists():
            raise SystemExit(f"{QUEUE} exists; the sample is drawn once (delete it deliberately to redraw)")
        picked = sample(model_accepts([Path("gold/termsheets.jsonl"), Path("gold/prospectuses.jsonl")]), n=args.n)
        QUEUE.parent.mkdir(parents=True, exist_ok=True)
        QUEUE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked))
        print(f"sampled {len(picked)} model-verified questions into {QUEUE}")
        return 0
    queue = _read_jsonl(QUEUE)
    if args.command == "serve":
        from aex.gold.ai_review import entries, pages
        check_bind_address(args.host)
        ents = entries()
        audit = Audit(queue, GRADES, lambda d, p: pages(d)[p - 1])
        server = ThreadingHTTPServer((args.host, args.port), make_handler(
            audit, lambda d: Path("data") / ents[d].corpus / f"{d}.pdf"))
        print(f"gold audit on http://{args.host}:{args.port}/ ({len(queue)} items)", flush=True)
        server.serve_forever()
        return 0
    result = report(queue, latest(GRADES))
    Path("gold/audit.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "wrong_qids"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

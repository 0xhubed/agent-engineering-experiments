"""Validate the LLM judge against Daniel's grades (spec §6.3; plan 1b Task 8).

Usage:
  python -m aex.gold.judge_review queue configs/tree_rag/<dev-run>.yaml [--n 120]   # sample from a run
  python -m aex.gold.judge_review serve [--host 100.84.235.48] [--port 8766]        # grade on the phone
  python -m aex.gold.judge_review kappa [--by daniel]                               # -> gold/judge_kappa.json

  # Grading by a model instead (Daniel's decision, 2026-10-09: Claude grades, recorded as Claude's):
  python -m aex.gold.judge_review blind --by claude-opus-5.5 [--n 10]   # next ungraded items, no judge verdict
  python -m aex.gold.judge_review grade <item_id> correct|incorrect --by claude-opus-5.5 --reason "..."

The queue holds answers the LLM judge graded (gold kind "free", plus native PageIndex answers), sampled
evenly across arms. Grading is blind: the page shows the question, the gold answer and the model's
answer, never the judge's verdict. Daniel's grades are appended to gold/review/judge_grades.jsonl, any
other grader's to judge_grades.<by>.jsonl (latest per item wins; Undo removes the last). kappa compares
one grader's grades with the judge's verdicts and names the grader; below 0.7, adjust the judge prompt on
dev and repeat (spec §6.3).
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from aex.common.judge import cohen_kappa
from aex.experiments.tree_rag.types import Question
from aex.gold.review_app import CSS, check_bind_address

KAPPA_THRESHOLD = 0.7
REVIEW_DIR = Path("gold/review")
HUMAN = "daniel"


def grades_file(review_dir: Path, by: str) -> Path:
    return review_dir / ("judge_grades.jsonl" if by == HUMAN else f"judge_grades.{by}.jsonl")


def item_id(row: dict) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{row['qid']}-{row['arm']}-{row['navigator']}-{row['answerer']}".lower()).strip("-")


def build_queue(rows: list[dict], questions: list[Question], *, n: int = 120, seed: int = 17) -> list[dict]:
    """Up to n judge-graded answers from dev-split questions, round-robin across arms (seeded)."""
    by_qid = {q.qid: q for q in questions}
    pools: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        q = by_qid.get(r["qid"])
        detail = r.get("detail") or {}
        # Exactly what the judge graded: a native row's full prose (detail.answer keeps only 500 characters),
        # otherwise the extracted final answer.
        answer = detail.get("raw") if r["arm"].endswith("_native") else detail.get("answer")
        if (q is None or q.split != "dev" or r.get("judge") != "llm" or r.get("correct") is None
                or r.get("failure") == "error" or not answer):
            continue
        gold = q.gold.value if not isinstance(q.gold.value, list) else "; ".join(q.gold.value)
        pools[r["arm"]].append({"item_id": item_id(r), "qid": q.qid, "arm": r["arm"], "question": q.question,
                                "gold": gold if q.gold.kind != "unanswerable" else "Not stated in the document",
                                "answer": str(answer), "judge_correct": bool(r["correct"])})
    rng = random.Random(seed)
    for arm in pools:
        rng.shuffle(pools[arm])
    queue: list[dict] = []
    arms = sorted(pools)
    while len(queue) < n and any(pools.values()):
        for arm in arms:
            if pools[arm] and len(queue) < n:
                queue.append(pools[arm].pop())
    return queue


def latest_grades(path: Path) -> dict[str, bool]:
    grades: dict[str, bool] = {}
    order: list[str] = []
    if path.exists():
        for ln in path.read_text().splitlines():
            if not ln.strip():
                continue
            g = json.loads(ln)
            if g["action"] == "undo":
                if order:
                    grades.pop(order.pop(), None)
                continue
            grades[g["item_id"]] = g["action"] == "correct"
            order.append(g["item_id"])
    return grades


def kappa_report(queue: list[dict], grades: dict[str, bool], *, grader: str = HUMAN) -> dict:
    graded = [it for it in queue if it["item_id"] in grades]
    if not graded:
        raise ValueError("no graded items yet")
    human = [grades[it["item_id"]] for it in graded]
    judge = [it["judge_correct"] for it in graded]
    kappa = cohen_kappa(human, judge)
    return {"grader": grader, "n": len(graded), "kappa": round(kappa, 3), "agreement": round(sum(h == j for h, j in zip(human, judge)) / len(graded), 3),
            "grader_correct": sum(human), "judge_correct": sum(judge), "threshold": KAPPA_THRESHOLD,
            "passes": kappa >= KAPPA_THRESHOLD and len(graded) >= 100,
            "disagreements": [it["item_id"] for it, h in zip(graded, human) if h != it["judge_correct"]]}


# --- grading page -------------------------------------------------------------------------------------

class Grading:
    def __init__(self, queue: list[dict], grades_path: Path) -> None:
        self.queue, self.grades_path = queue, grades_path
        self.skipped: list[str] = []
        self.lock = threading.Lock()

    def next_item(self) -> dict | None:
        grades = latest_grades(self.grades_path)
        open_ = [it for it in self.queue if it["item_id"] not in grades]
        if not open_:
            return None
        skipped = {s: i for i, s in enumerate(self.skipped)}
        return min(open_, key=lambda it: (it["item_id"] in skipped, skipped.get(it["item_id"], 0),
                                          self.queue.index(it)))

    def record(self, item: str, action: str, *, reason: str | None = None) -> None:
        with self.lock:
            if action == "skip":
                if item in self.skipped:
                    self.skipped.remove(item)
                self.skipped.append(item)
                return
            if action not in ("correct", "incorrect", "undo"):
                raise ValueError(f"unknown action {action!r}")
            self.grades_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.grades_path, "a") as fh:
                entry = {"item_id": item, "action": action,
                         "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
                fh.write(json.dumps(entry | ({"reason": reason} if reason else {})) + "\n")


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def grading_html(g: Grading) -> str:
    item = g.next_item()
    done = len(latest_grades(g.grades_path))
    head = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>Judge check</title><style>{CSS}</style></head><body>'
            f'<h1>Judge check · {done}/{len(g.queue)} graded (≥100 needed)</h1>')
    undo = ('<div class="meta"><form method="post" action="/undo" style="display:inline"><button '
            'style="font-size:14px;padding:6px 10px">Undo last grade</button></form></div>')
    if item is None:
        return head + "<p>All answers graded.</p>" + undo + "</body></html>"
    return head + f"""
<div class="box"><div class="form">question</div><div class="q">{_esc(item["question"])}</div></div>
<div class="box"><div class="form">gold answer</div><div>{_esc(item["gold"])}</div></div>
<div class="box"><div class="form">model answer — is it correct?</div><div>{_esc(item["answer"])}</div></div>
<form id="grade" method="post" action="/grade"><input type="hidden" name="item_id" value="{_esc(item["item_id"])}"></form>
{undo}
<div class="bar"><button class="no" form="grade" name="action" value="incorrect">Incorrect</button>
<button class="skip" form="grade" name="action" value="skip">Skip</button>
<button class="ok" form="grade" name="action" value="correct">Correct</button></div>
</body></html>"""


def make_handler(g: Grading):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def do_GET(self):
            if urlsplit(self.path).path != "/":
                return self._send(b"not found", 404)
            self._send(grading_html(g).encode())

        def do_POST(self):
            path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
            if path == "/undo":
                g.record("", "undo")
                return self._redirect()
            if path != "/grade" or form.get("item_id") not in {it["item_id"] for it in g.queue}:
                return self._send(b"bad request", 400)
            try:
                g.record(form["item_id"], form.get("action", ""))
            except ValueError:
                return self._send(b"bad action", 400)
            self._redirect()

    return Handler


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="judge_review")
    ap.add_argument("command", choices=["queue", "serve", "kappa", "blind", "grade"])
    ap.add_argument("config", nargs="?", help="queue: the dev run's config; grade: the item id")
    ap.add_argument("action", nargs="?", choices=["correct", "incorrect"])
    ap.add_argument("--by", default=HUMAN, help="who grades (the web page is always Daniel's)")
    ap.add_argument("--reason")
    ap.add_argument("--n", type=int, help="queue: sample size (120); blind: items shown (10)")
    ap.add_argument("--host", default="100.84.235.48")
    ap.add_argument("--port", type=int, default=8766)   # 8765 is the gold review app
    ap.add_argument("--dir", default=str(REVIEW_DIR))
    ap.add_argument("--out", default="gold/judge_kappa.json")
    args = ap.parse_args(argv[1:])
    review_dir = Path(args.dir)
    queue_path, grades_path = review_dir / "judge_queue.jsonl", grades_file(review_dir, args.by)

    if args.command == "queue":
        from aex.common.checkpoint import Checkpoint
        from aex.experiments.tree_rag.gold import load_questions
        from aex.experiments.tree_rag.run import RunConfig
        if not args.config:
            raise SystemExit("queue needs the dev run's config")
        cfg = RunConfig.from_yaml(args.config)
        queue = build_queue(Checkpoint(cfg.checkpoint).rows(),
                            load_questions(cfg.gold, seed=cfg.seed, dev_fraction=cfg.dev_fraction), n=args.n or 120,
                            seed=cfg.seed)
        review_dir.mkdir(parents=True, exist_ok=True)
        queue_path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in queue))
        print(f"wrote {len(queue)} items to {queue_path}" + ("" if len(queue) >= 100 else " (fewer than 100!)"))
        return 0

    queue = _read_jsonl(queue_path)
    if args.command == "blind":
        done = latest_grades(grades_path)
        for it in [it for it in queue if it["item_id"] not in done][:args.n or 10]:
            print(json.dumps({k: it[k] for k in ("item_id", "question", "gold", "answer")}, ensure_ascii=False))
        print(f"# {len(done)}/{len(queue)} graded by {args.by}")
        return 0
    if args.command == "grade":
        if args.by == HUMAN:
            raise SystemExit("Daniel grades on the web page; pass --by for any other grader")
        if args.config not in {it["item_id"] for it in queue} or not args.action or not (args.reason or "").strip():
            raise SystemExit("grade needs a queued item id, correct|incorrect and --reason")
        Grading(queue, grades_path).record(args.config, args.action, reason=args.reason.strip())
        print(f"{args.config}: {args.action}")
        return 0
    if args.command == "serve":
        if args.by != HUMAN:
            raise SystemExit("the grading page records Daniel's grades only")
        check_bind_address(args.host)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(Grading(queue, grades_path)))
        print(f"judge check on http://{args.host}:{args.port}/ ({len(queue)} items)", flush=True)
        server.serve_forever()
        return 0

    report = kappa_report(queue, latest_grades(grades_path), grader=args.by)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

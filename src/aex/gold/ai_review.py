"""Model review of drafted gold questions, recorded as such (decided with Daniel, 2026-10-07).

Daniel delegated the first-pass review to Claude: every decision made here is stored with
`verifier: claude-opus-5.5`, so accepted items become gold with `verified_by: claude-opus-5.5`, never
"daniel". A random sample of these accepts is then audited by Daniel (aex.gold.audit) and the measured
error rate is reported with the results.

Usage (one reviewer per shard; decisions go to gold/review/ai/<by>.jsonl until merged):
  python -m aex.gold.ai_review list --shard 0/8
  python -m aex.gold.ai_review show <draft_id>
  python -m aex.gold.ai_review page <doc_id> <page>
  python -m aex.gold.ai_review search <doc_id> <regex>
  python -m aex.gold.ai_review decide <draft_id> accept|reject --by r0 --reason "..." [--answer ...] [--kind ...]
                                      [--question ...] [--evidence "doc:page,doc:page"] [--qtype ...]
  python -m aex.gold.ai_review merge          # ai decisions -> gold/review/decisions.jsonl, gold files rebuilt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from aex.gold.review_app import ReviewError, _tidy, gold_records, parse_evidence, render_question, validate

VERIFIER = "claude-opus-5.5"
DRAFTS = Path("gold/drafts")
AI_DIR = Path("gold/review/ai")
DECISIONS = Path("gold/review/decisions.jsonl")
PARSED = Path("data/parsed")
MANIFESTS = ("corpus/termsheets/manifest.csv", "corpus/prospectuses/manifest.csv")


def load_drafts() -> dict[str, dict]:
    out = {}
    for name in ("termsheets", "prospectuses"):
        path = DRAFTS / f"{name}.jsonl"
        for ln in path.read_text().splitlines() if path.exists() else []:
            if ln.strip():
                d = json.loads(ln)
                if d["status"] == "draft":
                    out[d["draft_id"]] = d
    return out


def decided_ids() -> set[str]:
    ids = set()
    for path in [DECISIONS, *sorted(AI_DIR.glob("*.jsonl"))]:
        for ln in path.read_text().splitlines() if path.exists() else []:
            if ln.strip():
                d = json.loads(ln)
                if d["action"] in ("accept", "reject"):
                    ids.add(d["draft_id"])
    return ids


def entries() -> dict:
    from aex.corpus.manifest import ExclusionList, load_manifest
    excluded = ExclusionList.load()
    return {e.doc_id: e for m in MANIFESTS for e in load_manifest(m, excluded=excluded)}


def pages(doc_id: str) -> list[str]:
    return [p["text"] for p in json.loads((PARSED / f"{doc_id}.json").read_text())["pages"]]


def descriptions() -> dict:
    path = DRAFTS / "descriptions.json"
    return json.loads(path.read_text()) if path.exists() else {}


def facts() -> dict:
    path = DRAFTS / "facts.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []
    return {r["doc_id"]: r["facts"] for r in rows}


def show(draft: dict) -> str:
    ents, desc = entries(), descriptions()
    out = [f"draft_id: {draft['draft_id']}", f"type: {draft['qtype']}  regime: {draft['regime']}  "
           f"corpus: {draft['corpus']}  docs: {', '.join(draft['doc_ids'])}"]
    q = draft["question"]
    if draft["regime"] == "A":
        doc0 = draft["doc_ids"][0]
        d = (desc.get(doc0) or {}).get("description")
        out.append(f"question template: {q}")
        out.append(f"  by ISIN:        {render_question(q, 'the product with ISIN ' + ents[doc0].isin)}")
        out.append(f"  by description: {render_question(q, d) if d else '(no unique description: ISIN form only)'}")
    else:
        out.append(f"question: {q}")
    answer = draft["answer"]
    out.append(f"drafted answer: {', '.join(answer) if isinstance(answer, list) else answer!r}   kind: {draft['kind']}")
    ev = ", ".join(f"{e['doc_id']}:{e['page']}" for e in draft["evidence"])
    out.append(f"evidence: {ev or '(none)'}")
    for qu in draft.get("quotes") or []:
        out.append(f"quote {qu['doc_id']} p.{qu['page']} (found on page: {qu.get('ok')}): {qu.get('text')!r}")
    if draft["qtype"] == "unanswerable":
        out.append(f"why absent (drafter): {draft.get('why_absent')}")
        out.append(f"keywords: {draft.get('keywords')}  checked by drafter in: {draft.get('checked')}")
        out.append("=> verify absence yourself with `search` over the whole document before accepting.")
    if draft["qtype"] == "disambiguation":
        f_all, field = facts(), draft.get("field")
        out.append(f"field: {field}; this product's description: {(desc.get(draft['doc_ids'][0]) or {}).get('description')}")
        for s in draft.get("siblings") or []:
            sf = (f_all.get(s) or {}).get(field) or {}
            out.append(f"  sibling {s}: {field} = {sf.get('value')!r}; description: {(desc.get(s) or {}).get('description')}")
    for e in draft["evidence"]:
        text = pages(e["doc_id"])[e["page"] - 1]
        out.append(f"\n----- {e['doc_id']} page {e['page']} -----\n{_tidy(text)}")
    return "\n".join(out)


def search(doc_id: str, pattern: str, limit: int = 25) -> str:
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for n, text in enumerate(pages(doc_id), 1):
        flat = " ".join(text.split())
        for m in rx.finditer(flat):
            hits.append(f"p.{n}: …{flat[max(0, m.start() - 120):m.end() + 120]}…")
            if len(hits) >= limit:
                return "\n".join(hits) + f"\n(stopped at {limit} hits)"
    return "\n".join(hits) or "(no match)"


def decide(draft: dict, action: str, *, by: str, reason: str, answer=None, kind=None, question=None, evidence=None,
           qtype=None) -> dict:
    if not reason.strip():
        raise ReviewError("every decision needs a reason")
    entry = {"draft_id": draft["draft_id"], "action": action, "verifier": VERIFIER, "reason": reason.strip(),
             "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if action == "accept":
        drafted = draft["answer"]
        kind = kind or draft["kind"]
        edits = {"question": question if question is not None else draft["question"],
                 "answer": answer if answer is not None else (", ".join(drafted) if isinstance(drafted, list) else (drafted or "")),
                 "kind": kind, "qtype": qtype or draft["qtype"],
                 "evidence": [] if kind == "unanswerable" else (parse_evidence(evidence) if evidence is not None else draft["evidence"])}
        ents, desc = entries(), descriptions()
        doc0 = draft["doc_ids"][0]
        isin = ents[doc0].isin if doc0 in ents and ents[doc0].isin else None
        for r in gold_records(draft, edits, isin=isin, description=(desc.get(doc0) or {}).get("description"),
                              verified_by=VERIFIER):
            validate(r)
        entry["edits"] = edits
    elif action != "reject":
        raise ReviewError(f"unknown action {action!r}")
    AI_DIR.mkdir(parents=True, exist_ok=True)
    with open(AI_DIR / f"{by}.jsonl", "a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def merge() -> str:
    from aex.gold.review_app import Review
    already = set()
    if DECISIONS.exists():
        already = {json.loads(ln)["draft_id"] for ln in DECISIONS.read_text().splitlines() if ln.strip()}
    # Several reviewers may have decided the same draft. An adjudication (written by `by=adjudication`) wins;
    # otherwise the reviewers must agree on accept/reject, or the merge stops.
    per_draft: dict[str, dict[str, dict]] = {}
    for path in sorted(AI_DIR.glob("*.jsonl")):
        for ln in path.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                per_draft.setdefault(d["draft_id"], {})[path.stem] = d
    drafts = load_drafts()
    new, unresolved = [], []
    for draft_id, by_reviewer in sorted(per_draft.items()):
        if draft_id in already or draft_id not in drafts:
            continue
        if "adjudication" in by_reviewer:
            new.append(by_reviewer["adjudication"])
        elif len({d["action"] for d in by_reviewer.values()}) == 1:
            new.append(max(by_reviewer.values(), key=lambda d: d["at"]))
        else:
            unresolved.append(draft_id)
    if unresolved:
        raise ReviewError(f"{len(unresolved)} conflicting decisions need an adjudication: {', '.join(unresolved)}")
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS, "a") as fh:
        for d in sorted(new, key=lambda d: d["at"]):
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    counts = Review(DRAFTS, "gold", entries(), {}).write_gold()
    return f"merged {len(new)} decisions; gold records: {counts}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ai_review")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--qtype")
    p = sub.add_parser("show")
    p.add_argument("draft_id")
    p = sub.add_parser("page")
    p.add_argument("doc_id")
    p.add_argument("page", type=int)
    p = sub.add_parser("search")
    p.add_argument("doc_id")
    p.add_argument("pattern")
    p = sub.add_parser("decide")
    p.add_argument("draft_id")
    p.add_argument("action", choices=["accept", "reject"])
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    for opt in ("--answer", "--kind", "--question", "--evidence", "--qtype"):
        p.add_argument(opt)
    sub.add_parser("merge")
    args = ap.parse_args(argv[1:])
    drafts = load_drafts()
    try:
        if args.cmd == "list":
            i, k = map(int, args.shard.split("/"))
            done = decided_ids()
            ids = [d for d in sorted(drafts) if d not in done and (not args.qtype or drafts[d]["qtype"] == args.qtype)]
            # Shard by a stable hash, not by position: drafts are still being added while reviewers work.
            print("\n".join(d for d in ids if int(hashlib.sha256(d.encode()).hexdigest(), 16) % k == i))
        elif args.cmd == "show":
            print(show(drafts[args.draft_id]))
        elif args.cmd == "page":
            print(_tidy(pages(args.doc_id)[args.page - 1]))
        elif args.cmd == "search":
            print(search(args.doc_id, args.pattern))
        elif args.cmd == "decide":
            if args.draft_id not in drafts:
                raise ReviewError(f"unknown draft {args.draft_id}")
            e = decide(drafts[args.draft_id], args.action, by=args.by, reason=args.reason, answer=args.answer,
                       kind=args.kind, question=args.question, evidence=args.evidence, qtype=args.qtype)
            print(f"recorded {e['action']} for {e['draft_id']}")
        else:
            print(merge())
    except (ReviewError, KeyError, IndexError, re.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

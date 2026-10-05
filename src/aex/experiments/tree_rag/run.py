"""Run arms × navigators × answerers over the gold questions, one checkpointed row each."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from aex.common.checkpoint import Checkpoint, row_key
from aex.common.judge import LLMJudge
from aex.common.llm import ChatClient, LLMError, MockClient, OpenAICompatClient
from aex.common.manifest import build_manifest, manifest_hash
from aex.common.scoring import extract_final, score
from aex.experiments.tree_rag.answer import answer
from aex.experiments.tree_rag.arms import IndexStats, Store, make_arm
from aex.experiments.tree_rag.gold import load_questions
from aex.experiments.tree_rag.types import ParsedDoc, Question, Retrieval


@dataclass(frozen=True)
class RunConfig:
    seed: int
    gold: str
    parsed_dir: str
    checkpoint: str
    arms: list[str]
    navigators: list[str]
    answerers: list[str]
    judge: str
    models: dict[str, dict]
    splits: tuple[str, ...] = ("test",)
    dev_fraction: float = 0.2
    max_evidence_words: int = 9000
    max_answer_tokens: int = 1024
    prices: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        data = yaml.safe_load(Path(path).read_text())
        data["splits"] = tuple(data.get("splits", ("test",)))
        return cls(**data)


def build_client(model_id: str, spec: dict) -> ChatClient:
    if spec["kind"] == "mock":
        return MockClient(lambda messages: "ANSWER: NOT_STATED", model=model_id)
    if spec["kind"] == "openai":
        api_key = os.environ.get(spec.get("api_key_env", ""), "EMPTY")
        return OpenAICompatClient(spec["base_url"], spec["model"], api_key=api_key,
                                  local=spec.get("local", True), extra_body=spec.get("extra_body"))
    raise ValueError(f"model {model_id!r}: unknown kind {spec['kind']!r}")


def evidence_metrics(q: Question, retrieval: Retrieval) -> tuple[float | None, float | None, bool | None]:
    gold = set(q.evidence)
    got = {(e.doc_id, e.page) for e in retrieval.evidence}
    recall = len(gold & got) / len(gold) if gold else None
    precision = len(gold & got) / len(got) if got else None
    wrong_doc = None if not got else not any(doc_id in q.doc_ids for doc_id, _ in got)
    return recall, precision, wrong_doc


def classify_failure(*, correct: bool | None, prior: str | None, recall: float | None,
                     wrong_doc: bool | None, parse_failure: bool) -> str | None:
    if correct is not False or prior:
        return prior
    if wrong_doc:
        return "wrong_doc"
    if parse_failure:
        return "parse_failure"
    if not recall:
        return "wrong_page"
    return "right_evidence_wrong_answer"


def _add(a: float | None, b: float | None) -> float | None:
    return b if a is None else a if b is None else a + b


def _base_row(q: Question, arm_name: str, navigator: str, answerer: str) -> dict:
    return {"qid": q.qid, "dataset": q.dataset, "regime": q.regime, "qtype": q.qtype,
            "arm": arm_name, "navigator": navigator, "answerer": answerer}


def _error_row(q, arm_name, navigator, answerer, message: str) -> dict:
    return _base_row(q, arm_name, navigator, answerer) | {
        "correct": None, "judge": None, "evidence_recall": None, "evidence_precision": None,
        "wrong_doc": None, "latency_s": None, "llm_calls_sequential": None, "tokens_query": None,
        "tokens_index_amortised": None, "cost_usd": None, "gpu_s": None, "failure": "error",
        "detail": {"answer": None, "raw": None, "evidence": [], "trace": {}, "truncated": False,
                   "error": message},
    }


def _evaluate_one(cfg: RunConfig, q: Question, arm, navigator: str, answerer_id: str,
                  clients: dict[str, ChatClient], store: Store, index: IndexStats, n_questions: int) -> dict:
    retrieval = arm.retrieve(q, store)
    recall, precision, wrong_doc = evidence_metrics(q, retrieval)
    row = _base_row(q, arm.name, navigator, answerer_id)
    amortised_tokens = (index.input_tokens + index.output_tokens) // max(n_questions, 1)
    detail = {"evidence": [[e.doc_id, e.page] for e in retrieval.evidence], "trace": retrieval.trace,
              "error": None, "answer": None, "raw": None, "truncated": False}
    if retrieval.not_applicable:
        return row | {"correct": None, "judge": None, "evidence_recall": None, "evidence_precision": None,
                      "wrong_doc": None, "latency_s": None, "llm_calls_sequential": None,
                      "tokens_query": None, "tokens_index_amortised": amortised_tokens,
                      "cost_usd": None, "gpu_s": None, "failure": "not_applicable", "detail": detail}

    ans = answer(q, retrieval, clients[answerer_id], max_evidence_words=cfg.max_evidence_words,
                 max_tokens=cfg.max_answer_tokens, seed=cfg.seed)
    result = score(ans.text, q.gold)
    correct, method = result.correct, result.method
    if correct is None:
        correct, _ = LLMJudge(clients[cfg.judge]).judge(q.question, str(q.gold.value), extract_final(ans.text))
    parse_failure = any(not store.get(d).page(p).text.strip() for d, p in q.evidence)
    ledger = retrieval.ledger
    n = max(n_questions, 1)
    prices = {model: tuple(pair) for model, pair in cfg.prices.items()}
    return row | {
        "correct": correct, "judge": method,
        "evidence_recall": recall, "evidence_precision": precision, "wrong_doc": wrong_doc,
        "latency_s": ledger.latency_s, "llm_calls_sequential": ledger.sequential_calls,
        "tokens_query": ledger.input_tokens + ledger.output_tokens,
        "tokens_index_amortised": amortised_tokens,
        "cost_usd": _add(ledger.cost_usd(prices), index.cost_usd / n if index.cost_usd is not None else None),
        "gpu_s": _add(ledger.gpu_s, index.gpu_s / n if index.gpu_s is not None else None),
        "failure": classify_failure(correct=correct, prior=result.failure, recall=recall,
                                    wrong_doc=wrong_doc, parse_failure=parse_failure),
        "detail": detail | {"answer": extract_final(ans.text), "raw": ans.text, "truncated": ans.truncated},
    }


def run_experiment(cfg: RunConfig, *, clients: dict[str, ChatClient], store: Store,
                   questions: list[Question], checkpoint: Checkpoint) -> None:
    selected = [q for q in questions if q.split in cfg.splits]
    for arm_name in cfg.arms:
        probe = make_arm(arm_name)
        navigators = cfg.navigators if probe.uses_navigator else ["none"]
        for navigator in navigators:
            arm = make_arm(arm_name, navigator=clients.get(navigator)) if probe.uses_navigator else probe
            index = arm.index(store)
            for q in selected:
                for answerer_id in cfg.answerers:
                    key = row_key(q.qid, arm_name, navigator, answerer_id)
                    existing = checkpoint.get(key)
                    if existing is not None and existing.get("failure") != "error":
                        continue
                    try:
                        row = _evaluate_one(cfg, q, arm, navigator, answerer_id, clients, store,
                                            index, len(selected))
                    except LLMError as exc:
                        row = _error_row(q, arm_name, navigator, answerer_id, str(exc))
                    checkpoint.put(key, row)


def _summary(rows: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["navigator"], r["answerer"])].append(r)
    lines = []
    for (arm, nav, ans), rs in sorted(groups.items()):
        scored = [r for r in rs if r["correct"] is not None]
        errors = sum(r["failure"] == "error" for r in rs)
        acc = sum(r["correct"] for r in scored) / len(scored) if scored else float("nan")
        lines.append(f"{arm:16} nav={nav:20} ans={ans:20} acc={acc:.3f} n={len(scored)} errors={errors}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    cfg = RunConfig.from_yaml(argv[1])
    questions = load_questions(cfg.gold, seed=cfg.seed, dev_fraction=cfg.dev_fraction)
    store = Store(ParsedDoc.from_dict(json.loads(p.read_text()))
                  for p in sorted(Path(cfg.parsed_dir).glob("*.json")))
    used = set(cfg.answerers) | {cfg.judge} | {n for n in cfg.navigators if n != "none"}
    clients = {mid: build_client(mid, cfg.models[mid]) for mid in used}
    manifest = build_manifest(asdict(cfg), models={mid: cfg.models[mid].get("revision", "unknown") for mid in used})
    Path(cfg.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{cfg.checkpoint}.manifest.json").write_text(
        json.dumps(manifest | {"manifest_hash": manifest_hash(manifest)}, indent=2))
    checkpoint = Checkpoint(cfg.checkpoint)
    run_experiment(cfg, clients=clients, store=store, questions=questions, checkpoint=checkpoint)
    print(_summary(checkpoint.rows()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

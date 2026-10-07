"""Load verified gold questions. Only reviewed items (`verified_by` set) may enter a run (spec §5.5; see
the pre-registration for the model-verified deviation)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import jsonschema

from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.types import Question

_SCHEMA = json.loads((Path(__file__).parent / "schema" / "gold.v1.schema.json").read_text())
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


class GoldError(ValueError):
    pass


def assign_split(qid: str, seed: int, dev_fraction: float = 0.2) -> Literal["dev", "test"]:
    digest = hashlib.sha256(f"{seed}:{qid}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "dev" if bucket < dev_fraction else "test"


def _check_evidence(record: dict, where: str) -> None:
    unanswerable = record["gold"]["kind"] == "unanswerable"
    if unanswerable and record["evidence"]:
        raise GoldError(f"{where}: unanswerable question must have empty evidence")
    if not unanswerable and not record["evidence"]:
        raise GoldError(f"{where}: answerable question needs at least one evidence page")


def load_questions(path: str | Path, *, seed: int, dev_fraction: float = 0.2) -> list[Question]:
    path = Path(path)
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldError(f"{where}: invalid JSON ({exc})") from exc
        errors = sorted(_VALIDATOR.iter_errors(record), key=lambda e: e.path)
        if errors:
            raise GoldError(f"{where}: {errors[0].message}")
        if record["qid"] in seen:
            raise GoldError(f"{where}: duplicate qid {record['qid']!r}")
        seen.add(record["qid"])
        _check_evidence(record, where)
        questions.append(Question(
            qid=record["qid"], dataset=record["dataset"], regime=record["regime"],
            qtype=record["qtype"], question=record["question"],
            doc_ids=tuple(record["doc_ids"]),
            gold=GoldAnswer(record["gold"]["kind"], record["gold"]["value"]),
            evidence=tuple((e["doc_id"], e["page"]) for e in record["evidence"]),
            # Both forms of a regime-A pair land in the same split, or dev tuning would see test answers.
            split=assign_split(record.get("pair_id", record["qid"]), seed, dev_fraction),
            pair_id=record.get("pair_id"), form=record.get("form"),
        ))
    return questions

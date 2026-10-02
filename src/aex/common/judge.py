"""LLM judge for free-text answers, validated against human grades via Cohen's kappa."""
from __future__ import annotations

from aex.common.llm import ChatClient, Completion

JUDGE_PROMPT = """You are grading an answer against a reference answer.

Question: {question}
Reference answer: {gold}
Candidate answer: {answer}

Reply with exactly one word: CORRECT if the candidate states the same fact as the
reference (wording may differ), otherwise INCORRECT."""


class LLMJudge:
    def __init__(self, client: ChatClient) -> None:
        self._client = client

    def judge(self, question: str, gold: str, answer: str) -> tuple[bool, Completion]:
        prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=answer)
        completion = self._client.complete([{"role": "user", "content": prompt}], max_tokens=5)
        return completion.text.strip().upper().startswith("CORRECT"), completion


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("cohen_kappa needs two non-empty label lists of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)

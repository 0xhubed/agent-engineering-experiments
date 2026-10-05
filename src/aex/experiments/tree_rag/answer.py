"""The one answer step every arm shares: same prompt, same model, same evidence budget."""
from __future__ import annotations

from aex.common.llm import ChatClient, LLMError
from aex.experiments.tree_rag.types import Answer, EvidencePage, Question, Retrieval

ANSWER_PROMPT = """Answer the question using only the document excerpts below.
If the excerpts do not contain the answer, reply with: ANSWER: NOT_STATED
Show any calculation briefly, then give the final answer on its own last line as:
ANSWER: <answer>

Question: {question}

Excerpts:
{excerpts}"""


def format_excerpts(evidence: list[EvidencePage], *, max_words: int) -> tuple[str, bool]:
    if not evidence:
        return "(no excerpts retrieved)", False
    blocks: list[str] = []
    budget = max_words
    truncated = False
    for ev in evidence:
        words = ev.text.split()
        if len(words) > budget:
            words, truncated = words[:budget], True
        blocks.append(f"[{ev.doc_id} p.{ev.page}]\n{' '.join(words)}")
        budget -= len(words)
        if budget <= 0:
            truncated = truncated or ev is not evidence[-1]
            break
    return "\n\n".join(blocks), truncated


def answer(q: Question, retrieval: Retrieval, client: ChatClient, *,
           max_evidence_words: int = 9000, max_tokens: int = 1024, seed: int = 0) -> Answer:
    excerpts, truncated = format_excerpts(retrieval.evidence, max_words=max_evidence_words)
    prompt = ANSWER_PROMPT.format(question=q.question, excerpts=excerpts)
    completion = client.complete([{"role": "user", "content": prompt}], max_tokens=max_tokens, seed=seed)
    retrieval.ledger.record(completion, local=client.local)
    if completion.finish_reason == "length":
        # Thinking models spend the budget on reasoning; a cut-off answer must not be scored as a wrong one.
        raise LLMError(f"{client.model}: answer truncated at max_tokens={max_tokens}")
    return Answer(completion.text, completion, truncated)

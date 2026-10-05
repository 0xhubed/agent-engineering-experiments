from aex.common.accounting import Ledger
from aex.common.llm import MockClient
from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.answer import answer, format_excerpts
from aex.experiments.tree_rag.types import EvidencePage, Question, Retrieval


def _q():
    return Question("q1", "fixture", "A", "lookup", "What is the barrier?", ("d1",),
                    GoldAnswer("numeric", "60%"), (("d1", 2),), "test")


def test_format_excerpts_labels_and_truncates():
    ev = [EvidencePage("d1", 2, "one two three"), EvidencePage("d1", 3, "four five six")]
    text, truncated = format_excerpts(ev, max_words=100)
    assert text == "[d1 p.2]\none two three\n\n[d1 p.3]\nfour five six" and not truncated
    short, truncated = format_excerpts(ev, max_words=4)
    assert truncated and "five" not in short and "four" in short


def test_format_excerpts_empty():
    assert format_excerpts([], max_words=10) == ("(no excerpts retrieved)", False)


def test_answer_sends_question_and_excerpts_and_records_ledger():
    seen = {}

    def responder(messages):
        seen["prompt"] = messages[-1]["content"]
        return "The barrier is stated on page 2.\nANSWER: 60%"

    retrieval = Retrieval([EvidencePage("d1", 2, "Barrier: 60% of initial")], {}, Ledger())
    out = answer(_q(), retrieval, MockClient(responder))
    assert "What is the barrier?" in seen["prompt"] and "Barrier: 60% of initial" in seen["prompt"]
    assert "NOT_STATED" in seen["prompt"]
    assert out.text.endswith("ANSWER: 60%") and not out.truncated
    assert retrieval.ledger.sequential_calls == 1


class _TruncatingClient:
    model, local = "t", True

    def complete(self, messages, **kw):
        from aex.common.llm import Completion
        return Completion("still thinking", 10, kw["max_tokens"], 0.0, "t", finish_reason="length")


def test_truncated_answer_raises_instead_of_scoring_wrong():
    import pytest
    from aex.common.llm import LLMError
    retrieval = Retrieval([EvidencePage("d1", 2, "Barrier: 60%")], {}, Ledger())
    with pytest.raises(LLMError, match="max_tokens=64"):
        answer(_q(), retrieval, _TruncatingClient(), max_tokens=64)
    assert retrieval.ledger.output_tokens == 64   # the spent tokens are still accounted

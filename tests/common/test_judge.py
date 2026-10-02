import pytest

from aex.common.judge import LLMJudge, cohen_kappa
from aex.common.llm import MockClient


def test_judge_parses_verdict():
    assert LLMJudge(MockClient(lambda m: "CORRECT")).judge("q", "a", "a")[0] is True
    assert LLMJudge(MockClient(lambda m: "INCORRECT")).judge("q", "a", "b")[0] is False
    assert LLMJudge(MockClient(lambda m: " correct\n")).judge("q", "a", "a")[0] is True


def test_judge_prompt_contains_all_three_fields():
    seen = {}

    def responder(messages):
        seen["prompt"] = messages[-1]["content"]
        return "CORRECT"

    LLMJudge(MockClient(responder)).judge("What barrier?", "60%", "sixty percent")
    assert "What barrier?" in seen["prompt"] and "60%" in seen["prompt"] and "sixty percent" in seen["prompt"]


def test_cohen_kappa_known_value():
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)


def test_cohen_kappa_perfect_agreement_constant_labels():
    assert cohen_kappa([True, True], [True, True]) == 1.0


def test_cohen_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])
    with pytest.raises(ValueError):
        cohen_kappa([], [])

from datetime import date

import pytest

from aex.common.scoring import (GoldAnswer, extract_final, is_refusal, parse_dates,
                                parse_numbers, score)


def test_extract_final_prefers_last_answer_tag():
    assert extract_final("reasoning...\nANSWER: 65%\nmore\nANSWER: 60%") == "60%"
    assert extract_final("line one\n\n  the barrier is 60%  \n") == "the barrier is 60%"


@pytest.mark.parametrize("text", ["NOT_STATED", "The document does not state this.",
                                  "This is not specified in the termsheet", "cannot be determined"])
def test_refusals(text):
    assert is_refusal(text)


def test_non_refusal():
    assert not is_refusal("The barrier is 60% of the initial level")


@pytest.mark.parametrize("answer,gold", [
    ("CHF 1,000.00", "1000"),
    ("1'000.00 CHF", "1000"),
    ("EUR 1.000,50", "1000.5"),
    ("65 %", "0.65"),
    ("65%", "65%"),
    ("0.65", "65%"),
    ("USD 99.6", "100"),          # within 0.5% relative
])
def test_numeric_equivalences(answer, gold):
    assert score(answer, GoldAnswer("numeric", gold)).correct is True


def test_numeric_outside_tolerance():
    assert score("98", GoldAnswer("numeric", "100")).correct is False


def test_parse_numbers_none_when_no_digits():
    assert parse_numbers("no numbers here") is None


def test_dates_all_formats():
    want = date(2027, 6, 15)
    for s in ["2027-06-15", "15.06.2027", "15/06/2027", "15 June 2027", "June 15, 2027", "15-Jun-2027"]:
        assert parse_dates(s) == [want], s


def test_date_and_date_list():
    assert score("15 June 2027", GoldAnswer("date", "2027-06-15")).correct is True
    gold = GoldAnswer("date_list", ["2026-06-15", "2026-12-15", "2027-06-15"])
    assert score("15.06.2026; 15.12.2026; 15.06.2027", gold).correct is True
    assert score("15.06.2026; 15.12.2026", gold).correct is False   # missing one date


def test_exact_is_case_and_whitespace_insensitive():
    assert score("  barrier  reverse convertible.", GoldAnswer("exact", "Barrier Reverse Convertible")).correct


def test_unanswerable_refusal_correct_and_concrete_answer_flagged():
    gold = GoldAnswer("unanswerable", None)
    assert score("ANSWER: NOT_STATED", gold).correct is True
    wrong = score("ANSWER: 4.5%", gold)
    assert wrong.correct is False and wrong.failure == "answered_unanswerable"


def test_refusal_on_answerable_question_is_wrong_without_failure_label():
    r = score("NOT_STATED", GoldAnswer("numeric", "60"))
    assert r.correct is False and r.failure is None


def test_free_text_defers_to_judge():
    r = score("some prose", GoldAnswer("free", "reference prose"))
    assert r.correct is None and r.method == "llm"

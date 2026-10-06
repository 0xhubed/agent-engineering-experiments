import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.types import Question
from aex.gold.judge_review import Grading, build_queue, kappa_report, latest_grades, make_handler


def _q(qid, split="dev", kind="free"):
    return Question(qid, "d", "B", "lookup", f"Question {qid}?", ("d",), GoldAnswer(kind, "the gold"), (("d", 1),), split)


def _row(qid, arm, correct=True, judge="llm", failure=None, answer="an answer"):
    return {"qid": qid, "arm": arm, "navigator": "m", "answerer": "m", "judge": judge, "correct": correct,
            "failure": failure, "detail": {"answer": answer}}


def test_queue_takes_only_judged_dev_answers_round_robin_across_arms():
    qs = [_q(f"q{i}") for i in range(10)] + [_q("t1", split="test")]
    rows = ([_row(f"q{i}", "a") for i in range(10)] + [_row(f"q{i}", "b") for i in range(3)]
            + [_row("t1", "a"), _row("q0", "c", judge="numeric"), _row("q1", "c", failure="error"),
               _row("q2", "c", answer=None), _row("q3", "c", correct=None)])
    queue = build_queue(rows, qs, n=8, seed=1)
    assert len(queue) == 8 and {it["arm"] for it in queue} == {"a", "b"}
    assert sum(it["arm"] == "b" for it in queue) == 3          # b exhausted, a fills the rest
    assert all(it["qid"] != "t1" for it in queue)
    assert queue == build_queue(rows, qs, n=8, seed=1)
    assert "judge_correct" in queue[0] and len({it["item_id"] for it in queue}) == 8


def test_grades_undo_and_kappa(tmp_path):
    queue = [{"item_id": f"i{k}", "judge_correct": k < 2} for k in range(4)]
    g = Grading(queue, tmp_path / "grades.jsonl")
    for item, action in [("i0", "correct"), ("i1", "incorrect"), ("i2", "incorrect"), ("i3", "correct"), ("", "undo"),
                         ("i3", "incorrect")]:
        g.record(item, action)
    grades = latest_grades(tmp_path / "grades.jsonl")
    assert grades == {"i0": True, "i1": False, "i2": False, "i3": False}
    report = kappa_report(queue, grades)
    assert report["n"] == 4 and report["kappa"] == pytest.approx(0.5) and report["disagreements"] == ["i1"]
    assert report["passes"] is False                 # kappa < 0.7 and n < 100
    with pytest.raises(ValueError):
        kappa_report(queue, {})


def test_grading_page_is_blind_and_records(tmp_path):
    queue = [{"item_id": "i0", "qid": "q", "arm": "a", "question": "What?", "gold": "G", "answer": "A",
              "judge_correct": True}]
    g = Grading(queue, tmp_path / "grades.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(g))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "What?" in page and "judge_correct" not in page and "true" not in page.lower().split("<style>")[0]
        urllib.request.urlopen(base + "/grade", data=urllib.parse.urlencode({"item_id": "i0", "action": "incorrect"}).encode())
        assert latest_grades(tmp_path / "grades.jsonl") == {"i0": False}
        assert "All answers graded" in urllib.request.urlopen(base + "/").read().decode()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/grade", data=b"item_id=nope&action=correct")
    finally:
        server.shutdown()

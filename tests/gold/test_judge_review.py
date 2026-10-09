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


def test_a_model_grader_gets_its_own_file_and_is_named_in_the_report(tmp_path):
    from aex.gold.judge_review import main
    queue = [{"item_id": f"i{n}", "qid": f"q{n}", "arm": "a", "question": "Q?", "gold": "g", "answer": "x",
              "judge_correct": n % 2 == 0} for n in range(4)]
    (tmp_path / "judge_queue.jsonl").write_text("".join(json.dumps(it) + "\n" for it in queue))
    by = ["--by", "claude-opus-5.5", "--dir", str(tmp_path)]
    for n in range(4):
        assert main(["jr", "grade", f"i{n}", "correct" if n % 2 == 0 else "incorrect", "--reason", "r", *by]) == 0
    assert not (tmp_path / "judge_grades.jsonl").exists()
    lines = (tmp_path / "judge_grades.claude-opus-5.5.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["reason"] == "r"
    main(["jr", "kappa", *by, "--out", str(tmp_path / "k.json")])
    report = json.loads((tmp_path / "k.json").read_text())
    assert report["grader"] == "claude-opus-5.5" and report["kappa"] == 1.0 and not report["passes"]   # n < 100


def test_blind_view_never_shows_the_judge_verdict(tmp_path, capsys):
    from aex.gold.judge_review import main
    it = {"item_id": "i0", "qid": "q0", "arm": "a", "question": "Q?", "gold": "g", "answer": "x", "judge_correct": True}
    (tmp_path / "judge_queue.jsonl").write_text(json.dumps(it) + "\n")
    main(["jr", "blind", "--by", "claude-opus-5.5", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert '"i0"' in out and "judge_correct" not in out


def test_grade_refuses_to_write_daniels_grades(tmp_path):
    from aex.gold.judge_review import main
    (tmp_path / "judge_queue.jsonl").write_text(json.dumps({"item_id": "i0"}) + "\n")
    with pytest.raises(SystemExit):
        main(["jr", "grade", "i0", "correct", "--reason", "r", "--dir", str(tmp_path)])

import subprocess
import sys
import textwrap

from aex.common.checkpoint import Checkpoint, row_key


def test_put_done_rows(tmp_path):
    cp = Checkpoint(tmp_path / "run.sqlite")
    k = row_key("q1", "oracle", "mock", "mock")
    assert not cp.done(k)
    cp.put(k, {"qid": "q1", "correct": True})
    assert cp.done(k)
    assert cp.rows() == [{"qid": "q1", "correct": True}]


def test_get_returns_row_or_none(tmp_path):
    cp = Checkpoint(tmp_path / "run.sqlite")
    cp.put("k", {"v": 1})
    assert cp.get("k") == {"v": 1}
    assert cp.get("missing") is None


def test_put_same_key_replaces_not_duplicates(tmp_path):
    cp = Checkpoint(tmp_path / "run.sqlite")
    k = row_key("q1", "a", "n", "m")
    cp.put(k, {"v": 1})
    cp.put(k, {"v": 2})
    assert len(cp) == 1 and cp.rows() == [{"v": 2}]


def test_survives_hard_kill_and_resumes_without_duplicates(tmp_path):
    db = tmp_path / "run.sqlite"
    # Child writes rows 0..2 then dies abruptly (no close, no cleanup) while "working" on row 3.
    child = textwrap.dedent(f"""
        import os
        from aex.common.checkpoint import Checkpoint, row_key
        cp = Checkpoint({str(db)!r})
        for i in range(5):
            if i == 3:
                os._exit(9)
            cp.put(row_key(f"q{{i}}", "a", "n", "m"), {{"i": i}})
    """)
    result = subprocess.run([sys.executable, "-c", child])
    assert result.returncode == 9

    cp = Checkpoint(db)
    assert [r["i"] for r in cp.rows()] == [0, 1, 2]
    for i in range(5):
        k = row_key(f"q{i}", "a", "n", "m")
        if not cp.done(k):
            cp.put(k, {"i": i})
    assert [r["i"] for r in cp.rows()] == [0, 1, 2, 3, 4]

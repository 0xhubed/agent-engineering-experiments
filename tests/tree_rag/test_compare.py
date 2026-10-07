from aex.experiments.tree_rag.compare import compare


def _row(qid, arm="hybrid_rerank", raw="ANSWER: 60%", correct=True, pages=(("d", 2),)):
    return {"qid": qid, "arm": arm, "navigator": "none", "answerer": "m", "correct": correct,
            "detail": {"raw": raw, "evidence": [{"doc_id": d, "page": p} for d, p in pages]}}


def test_compare_rates_per_arm_and_overall():
    a = [_row("q1"), _row("q2"), _row("q3", arm="pageindex"), _row("q4")]
    b = [_row("q1"), _row("q2", raw="ANSWER: 60 %"), _row("q3", arm="pageindex", pages=(("d", 3),)), _row("q5")]
    r = compare(a, b)
    assert (r["shared_rows"], r["only_a"], r["only_b"]) == (3, 1, 1)
    assert r["rates"]["hybrid_rerank"] == {"n": 2, "raw": 0.5, "correct": 1.0, "evidence": 1.0,
                                           "correct_and_evidence": 1.0}
    assert r["rates"]["all"]["correct_and_evidence"] == round(2 / 3, 4)
    assert r["differing"] == ["q3|pageindex|none|m"]


def test_phase0_list_evidence_is_comparable():
    a = [_row("q1") | {"detail": {"raw": "x", "evidence": [["d", 2]]}}]
    assert compare(a, [_row("q1", raw="x")])["rates"]["all"]["evidence"] == 1.0

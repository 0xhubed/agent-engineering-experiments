import pytest

import aex.experiments.tree_rag.arms.pageindex_arm as pia
from aex.common.llm import LLMError, OpenAICompatClient
from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.arms import ARMS, Store, make_arm
from aex.experiments.tree_rag.types import Page, ParsedDoc, Question, TreeNode

TREE = (TreeNode("n1", None, "Terms", 1, 3, 1), TreeNode("n2", "n1", "Barrier", 2, 3, 2),
        TreeNode("n3", None, "Redemption", 7, 8, 1), TreeNode("n4", None, "Risks", 9, 10, 1))


def _store():
    doc = ParsedDoc("d1", "Doc", "sha256:d1", tuple(Page(n, f"our text of page {n}") for n in range(1, 11)), TREE)
    other = ParsedDoc("d2", "Other", "sha256:d2", (Page(1, "other"),), ())
    return Store([doc, other], groups={"d1": "ts", "d2": "ts"})


def _q():
    return Question("q1", "ts", "A", "lookup", "Shares per note after a barrier event?", ("d1",),
                    GoldAnswer("numeric", "4"), (("d1", 7),), "test")


class FakeClient:
    """Stands in for PageIndexClient; its chat() drives the real page-spec parser like the agent's tool does."""
    instances: list["FakeClient"] = []

    def __init__(self, reads=(("2-3", "d1.pdf"), ("7", "d1.pdf")), fail=False, shown=("d1",), structure=(),
                 **config):
        self.config, self.reads, self.fail = config, reads, fail
        self.shown, self.structure = shown, structure
        self.submitted, self.chats = [], []
        FakeClient.instances.append(self)

    def get_document(self, pid):
        return {"id": pid, "name": f"{pid[3:]}.pdf", "pageNum": 10,
                "description": f"Description of {pid[3:]}: barrier 60%, 4.0 shares per note."}

    def get_tree(self, pid, node_summary=False, include_text=True):
        return {"result": [{"title": "Terms", "node_id": "0000", "start_index": 1, "end_index": 3,
                            "summary": "Terms summary", "nodes": [
                                {"title": "Barrier", "node_id": "0001", "start_index": 2, "end_index": 3,
                                 "summary": "Barrier is 60%."}]},
                           {"title": "Redemption", "node_id": "0002", "start_index": 7, "end_index": 8,
                            "summary": "4.0 shares per note."}]}

    def submit_document(self, path, metadata=None):
        self.submitted.append((path, metadata))
        return {"doc_id": f"pi-{metadata['aex_doc_id']}"}

    def chat(self, question, doc_id=None, extra_body=None):
        import pageindex.agent_tools as at
        self.chats.append((question, doc_id, extra_body))
        if self.fail:
            raise RuntimeError("upstream exploded")
        for d in self.shown:            # the SDK's targeting block fetches each document's metadata
            self.get_document(f"pi-{d}")
        for d in self.structure:        # the get_document_structure tool
            self.get_tree(f"pi-{d}", node_summary=True)
        for pages, name in self.reads:
            at._parse_page_spec(pages, name)
        return "4.0 shares per note (page 7)."


@pytest.fixture
def pdfs(tmp_path):
    for d in ("d1", "d2"):
        (tmp_path / f"{d}.pdf").write_bytes(b"%PDF-1.4 " + d.encode())
    return tmp_path


def _arm(pdfs, tmp_path, monkeypatch, **fake):
    FakeClient.instances = []
    monkeypatch.setattr(pia, "_client_factory", lambda **config: FakeClient(**fake, **config))
    nav = OpenAICompatClient("http://upstream/v1", "qwen3.8-27b",
                             extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}})
    return make_arm("pageindex", navigator=nav, pdf_dir=str(pdfs), storage_dir=str(tmp_path / "pi"))


def test_registered_as_a_navigating_tree_arm():
    assert ARMS["pageindex"].family == "tree" and ARMS["pageindex"].uses_navigator


def test_index_submits_each_pdf_once_and_reuses_the_map(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch)
    arm.first_stats = arm.index(_store())
    assert sorted(m["aex_doc_id"] for _, m in FakeClient.instances[-1].submitted) == ["d1", "d2"]
    again = _arm(pdfs, tmp_path, monkeypatch)
    stats = again.index(_store())
    assert FakeClient.instances[-1].submitted == []      # same sha256 -> already indexed
    assert stats == arm.first_stats                      # ...but the build cost is still reported


def test_retrieve_returns_read_pages_in_our_parse_with_a_navigation_trace(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch)
    arm.index(_store())
    r = arm.retrieve(_q(), _store())
    assert [(e.doc_id, e.page, e.text) for e in r.evidence] == [
        ("d1", 2, "our text of page 2"), ("d1", 3, "our text of page 3"), ("d1", 7, "our text of page 7")]
    assert all(e.kind == "page" for e in r.evidence)
    t = r.trace
    assert [x["pages"] for x in t["reads"]] == [[2, 3], [7]]
    assert t["doc_id"] == "d1" and t["pages"] == [["d1", 2], ["d1", 3], ["d1", 7]]
    assert t["visited"] == ["n1", "n2", "n3"] and t["backtracks"] == ["n1", "n2"] and t["selected"] == "n3"
    assert t["native_answer"].startswith("4.0 shares")
    question, scope, extra = FakeClient.instances[-1].chats[-1]
    assert sorted(scope) == ["pi-d1", "pi-d2"]                     # the whole corpus, as released
    assert extra == {"chat_template_kwargs": {"reasoning_effort": "medium"}}


def test_no_page_read_means_no_evidence(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch, reads=())
    arm.index(_store())
    r = arm.retrieve(_q(), _store())
    assert r.trace["reads"] == [] and r.trace["selected"] is None
    assert all(e.kind == "summary" for e in r.evidence)            # falls back to what it was shown


def test_sdk_failures_become_llm_errors_so_the_row_is_retried(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch, fail=True)
    arm.index(_store())
    with pytest.raises(LLMError, match="exploded"):
        arm.retrieve(_q(), _store())


def test_missing_pdf_fails_at_index_time(pdfs, tmp_path, monkeypatch):
    (pdfs / "d2.pdf").unlink()
    with pytest.raises(FileNotFoundError, match="d2"):
        _arm(pdfs, tmp_path, monkeypatch).index(_store())


LIVE = "http://hubed-dgx:8000/v1"


def _live_up() -> bool:
    import httpx
    try:
        return httpx.get(f"{LIVE}/models", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _live_up(), reason="hubed-dgx endpoint not reachable")
def test_live_pageindex_reads_the_redemption_page(tmp_path):
    from aex.experiments.tree_rag.parse import parse_pdf
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    from aex.experiments.tree_rag.synthetic_pdf import make_termsheet_pdf
    make_termsheet_pdf(pdf_dir / "ts1.pdf")
    store = Store([parse_pdf(pdf_dir / "ts1.pdf", "ts1")])
    nav = OpenAICompatClient(LIVE, "qwen3.8-27b", extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}})
    arm = make_arm("pageindex", navigator=nav, pdf_dir=str(pdf_dir), storage_dir=str(tmp_path / "pi"))
    stats = arm.index(store)
    q = Question("live-1", "ts", "A", "lookup", "How many shares are delivered per note after a barrier event?",
                 ("ts1",), GoldAnswer("numeric", "4"), (("ts1", 7),), "test")
    r = arm.retrieve(q, store)
    print("index", stats, "\ntrace", r.trace, "\nledger", r.ledger.input_tokens, r.ledger.output_tokens,
          r.ledger.sequential_calls, round(r.ledger.latency_s, 1))
    # Plumbing, not model quality: which pages the agent picks varies between fresh index builds (tree
    # summaries come from concurrent calls, which vLLM does not reproduce bit for bit). On one fixed tree,
    # repeated queries read identical pages. Whether it finds page 7 is what the experiment measures.
    # About one fresh tree in three, the agent answers from the document description / node summaries
    # without reading a page; then the evidence is what it was shown, as summaries.
    assert r.trace["answered_from"] in ("pages", "summaries")
    pages = [e for e in r.evidence if e.kind == "page"]
    assert all(e.text == store.get("ts1").page(e.page).text for e in pages)
    assert r.trace["native_answer"].strip()
    assert r.ledger.sequential_calls >= 2 and r.ledger.input_tokens > 0
    assert stats.input_tokens > 0 and stats.gpu_s


def test_proxy_supplies_temperature_seed_and_thinking(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch)
    arm.index(_store())
    assert arm._proxy.defaults == {"temperature": 0.0, "seed": 0,
                                   "chat_template_kwargs": {"reasoning_effort": "medium"}}


def test_no_page_read_falls_back_to_what_the_agent_was_shown(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch, reads=(), shown=("d1",), structure=("d1",))
    arm.index(_store())
    r = arm.retrieve(_q(), _store())
    assert r.trace["answered_from"] == "summaries"
    assert all(e.kind == "summary" for e in r.evidence)
    assert [(e.page, e.end_page, e.text) for e in r.evidence] == [
        (1, 3, "Terms: Terms summary"), (2, 3, "Barrier: Barrier is 60%."), (7, 8, "Redemption: 4.0 shares per note."),
        (1, 10, "Description of d1: barrier 60%, 4.0 shares per note.")]


def test_description_only_when_no_structure_was_fetched(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch, reads=(), shown=("d1",))
    arm.index(_store())
    r = arm.retrieve(_q(), _store())
    assert [(e.kind, e.page, e.end_page) for e in r.evidence] == [("summary", 1, 10)]


def test_pages_read_means_no_summaries_and_native_answer_is_kept(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch, structure=("d1",))
    arm.index(_store())
    r = arm.retrieve(_q(), _store())
    assert r.trace["answered_from"] == "pages" and all(e.kind == "page" for e in r.evidence)
    assert r.trace["native_answer"] == "4.0 shares per note (page 7)."


def test_nothing_seen_means_no_evidence(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch, reads=(), shown=())
    arm.index(_store())
    r = arm.retrieve(_q(), _store())
    assert r.evidence == [] and r.trace["answered_from"] == "nothing"


def test_runner_writes_a_judged_native_row_next_to_the_shared_answer(pdfs, tmp_path, monkeypatch):
    from aex.common.checkpoint import Checkpoint
    from aex.common.llm import MockClient
    from aex.experiments.tree_rag.run import RunConfig, run_experiment
    FakeClient.instances = []
    monkeypatch.setattr(pia, "_client_factory", lambda **config: FakeClient(**config))
    nav = OpenAICompatClient("http://upstream/v1", "nav")
    judge_prompts = []

    def judge(messages):
        judge_prompts.append(messages[-1]["content"])
        return "CORRECT"

    clients = {"nav": nav, "ans": MockClient(lambda m: "ANSWER: 4", model="ans"), "judge": MockClient(judge, model="judge")}
    cfg = RunConfig(seed=17, gold="", parsed_dir="", checkpoint="", arms=["pageindex"], navigators=["nav"],
                    answerers=["ans"], judge="judge", models={}, splits=("test",),
                    arm_options={"pageindex": {"pdf_dir": str(pdfs), "storage_dir": str(tmp_path / "pi")}})
    cp = Checkpoint(tmp_path / "cp.sqlite")
    run_experiment(cfg, clients=clients, store=_store(), questions=[_q()], checkpoint=cp)
    rows = {r["arm"]: r for r in cp.rows()}
    assert set(rows) == {"pageindex", "pageindex_native"}
    native = rows["pageindex_native"]
    assert native["answerer"] == "nav" and native["judge"] == "llm" and native["correct"] is True
    assert native["llm_calls_sequential"] == rows["pageindex"]["llm_calls_sequential"] - 1   # no shared answer call
    assert "4.0 shares per note (page 7)." in judge_prompts[0]
    run_experiment(cfg, clients=clients, store=_store(), questions=[_q()], checkpoint=cp)
    assert len(cp) == 2                                                   # resume: nothing redone


def test_native_arm_is_registered_for_export():
    from aex.experiments.tree_rag.export import ARM_LABELS
    assert ARMS["pageindex_native"].family == "tree"
    assert ARM_LABELS["pageindex_native"] == "PageIndex (its own answer)"


NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _indexer_arm(pdfs, tmp_path, monkeypatch, indexer_body):
    arm = _arm(pdfs, tmp_path, monkeypatch)
    arm.indexer = OpenAICompatClient("http://upstream/v1", "qwen3.8-27b", extra_body=indexer_body)
    return arm


def test_trees_are_built_under_the_indexer_settings_and_navigation_keeps_the_navigators(pdfs, tmp_path, monkeypatch):
    arm = _indexer_arm(pdfs, tmp_path, monkeypatch, NO_THINK)
    seen = []
    monkeypatch.setattr(FakeClient, "submit_document", lambda self, path, metadata=None: (
        seen.append(dict(arm._proxy.defaults)), {"doc_id": f"pi-{metadata['aex_doc_id']}"})[1])
    arm.index(_store())
    assert [d["chat_template_kwargs"] for d in seen] == [{"enable_thinking": False}] * 2
    assert arm._proxy.defaults["chat_template_kwargs"] == {"reasoning_effort": "medium"}


def test_a_different_indexer_rebuilds_the_trees(pdfs, tmp_path, monkeypatch):
    _indexer_arm(pdfs, tmp_path, monkeypatch, NO_THINK).index(_store())
    _indexer_arm(pdfs, tmp_path, monkeypatch, NO_THINK).index(_store())
    assert FakeClient.instances[-1].submitted == []
    _arm(pdfs, tmp_path, monkeypatch).index(_store())          # back to the navigator's settings
    assert len(FakeClient.instances[-1].submitted) == 2


def test_the_indexer_must_be_the_navigators_served_model(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch)
    arm.indexer = OpenAICompatClient("http://elsewhere/v1", "other-model")
    with pytest.raises(ValueError, match="served model"):
        arm.index(_store())


def test_a_failed_tree_does_not_stop_the_others_but_fails_the_index(pdfs, tmp_path, monkeypatch):
    arm = _arm(pdfs, tmp_path, monkeypatch)

    def submit(self, path, metadata=None):
        if metadata["aex_doc_id"] == "d1":
            raise RuntimeError("node dropped")
        self.submitted.append((path, metadata))
        return {"doc_id": f"pi-{metadata['aex_doc_id']}"}

    monkeypatch.setattr(FakeClient, "submit_document", submit)
    with pytest.raises(RuntimeError, match="1 tree\\(s\\) failed to build: d1: RuntimeError: node dropped"):
        arm.index(_store())
    assert [m["aex_doc_id"] for _, m in FakeClient.instances[-1].submitted] == ["d2"]

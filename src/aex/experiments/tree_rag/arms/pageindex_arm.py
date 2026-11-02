"""PageIndex as released (SDK local mode), pointed at our endpoint; we record which pages its agent reads.

The SDK builds its own tree from the PDF and runs its own agent — that is the method under
test. What it reads is captured at `pageindex.agent_tools._parse_page_spec`, the one place
every page request passes through at call time. Those pages, in OUR shared parse, go to the
shared answerer; the SDK's own answer is kept in the trace only. Its model calls go through a
MeteringProxy so tokens, latency and sequential calls land in the question's Ledger.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from aex.common.accounting import Ledger
from aex.common.llm import ChatClient, LLMError
from aex.common.meter import MeteringProxy
from aex.experiments.tree_rag.arms.base import IndexStats, Store, register
from aex.experiments.tree_rag.parse import file_sha256
from aex.experiments.tree_rag.types import EvidencePage, Question, Retrieval

ATTEMPTS = 3
_lock = threading.Lock()
_active: list[tuple[float, str, list[int]]] | None = None


def _install_capture() -> None:
    import pageindex.agent_tools as at
    if getattr(at._parse_page_spec, "_aex_capture", False):
        return
    original = at._parse_page_spec

    def capturing(pages, doc_name, *args, **kwargs):
        result = original(pages, doc_name, *args, **kwargs)
        requested = result[0] if isinstance(result, tuple) else None
        with _lock:
            if _active is not None and requested:
                _active.append((time.perf_counter(), str(doc_name), [int(p) for p in requested]))
        return result

    capturing._aex_capture = True
    at._parse_page_spec = capturing


def _client_factory(*, model: str, storage_path: str, base_url: str):
    from pageindex import PageIndexClient
    backend = {"base_url": base_url, "api_key": "EMPTY"}
    return PageIndexClient(index={"model": f"openai/{model}", "storage_path": storage_path, "backend": backend},
                           chat={"model": f"openai/{model}", "backend": backend})


def _flatten(nodes: list[dict]) -> list[dict]:
    out = []
    for node in nodes or []:
        out.append(node)
        out += _flatten(node.get("nodes") or [])
    return out


@register("pageindex_native")
class PageIndexNativeRows:
    """Not runnable: names the rows that score the SDK's own answer next to the shared answerer's."""
    family = "tree"
    uses_navigator = True

    def __init__(self, **options) -> None:
        raise ValueError("pageindex_native rows are written by the pageindex arm; list 'pageindex' in arms")


def _doc_id_from_name(name: str) -> str:
    return name[:-4] if name.lower().endswith(".pdf") else name


@register("pageindex")
class PageIndexArm:
    family = "tree"
    uses_navigator = True

    def __init__(self, *, navigator: ChatClient | None = None, pdf_dir: str | None = None,
                 storage_dir: str = "data/pageindex", scopes: dict | None = None, seed: int = 0,
                 indexer: ChatClient | None = None, **options) -> None:
        self.navigator, self.scopes, self.seed = navigator, scopes or {}, seed
        self.indexer = indexer or navigator   # its settings apply while trees are built
        self.pdf_dir = Path(pdf_dir) if pdf_dir else None
        self.storage_dir = Path(storage_dir)
        self._client = None
        self._proxy: MeteringProxy | None = None
        self._ids: dict[str, str] = {}
        self.failed: dict[str, str] = {}   # doc_id -> why its tree could not be built
        self._shown: list[tuple[str, str]] | None = None   # (what, pageindex id) the agent was shown, in order

    # ── setup ──

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.navigator is None or not hasattr(self.navigator, "base_url"):
            raise ValueError("arm 'pageindex' needs an OpenAI-compatible navigator (navigators in the run config)")
        if self.pdf_dir is None:
            raise ValueError("arm 'pageindex' needs pdf_dir (arm_options.pageindex.pdf_dir)")
        _install_capture()
        # The SDK sends no temperature or seed, and no thinking setting on its indexing calls: the proxy
        # supplies ours wherever it leaves them out, so PageIndex runs under the same settings as every arm.
        if self.indexer is not self.navigator and (self.indexer.base_url, self.indexer.model) != (
                self.navigator.base_url, self.navigator.model):
            raise ValueError("arm 'pageindex': the indexer must be the navigator's served model (other settings only)")
        self._proxy = MeteringProxy(self.navigator.base_url, model=self.navigator.model, local=self.navigator.local,
                                    defaults=self._defaults(self.navigator),
                                    timeout_s=getattr(self.navigator, "timeout_s", 600.0))
        self._proxy.start()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._client = _client_factory(model=self.navigator.model, storage_path=str(self.storage_dir),
                                       base_url=self._proxy.base_url)
        self._watch("get_document", "description")
        self._watch("get_tree", "structure")

    def _watch(self, method: str, what: str) -> None:
        """Record which documents' descriptions and structures the SDK puts in front of its agent."""
        original = getattr(self._client, method)

        def watched(pid, *args, **kwargs):
            if self._shown is not None:
                self._shown.append((what, pid))
            return original(pid, *args, **kwargs)

        setattr(self._client, method, watched)

    def _defaults(self, client: ChatClient) -> dict:
        return {"temperature": 0.0, "seed": self.seed, **(getattr(client, "extra_body", None) or {})}

    @property
    def _map_file(self) -> Path:
        return self.storage_dir / "aex-index.json"

    @property
    def _failed_file(self) -> Path:
        return self.storage_dir / "aex-failed.json"

    def index(self, store: Store) -> IndexStats:
        self._ensure_client()
        known = json.loads(self._map_file.read_text()) if self._map_file.exists() else {}
        failed = json.loads(self._failed_file.read_text()) if self._failed_file.exists() else {}
        total = IndexStats()
        indexer = json.dumps(self._defaults(self.indexer), sort_keys=True)
        try:
            self._proxy.defaults = self._defaults(self.indexer)
            self._build_trees(store, known, failed, indexer, total)
        finally:
            self._proxy.defaults = self._defaults(self.navigator)
        self._ids = {d: v["pageindex_id"] for d, v in known.items()}
        self.failed = {d: v["error"] for d, v in failed.items() if d in store.docs and d not in self._ids}
        if self.failed:
            # PageIndex runs its released defaults only (pre-registration §5): a tree that does not build is
            # not retried with other settings. Questions that need it are scored incorrect (`index_failed`).
            print(f"pageindex: {len(self.failed)} tree(s) failed to build: "
                  + "; ".join(f"{d}: {e}" for d, e in sorted(self.failed.items())), file=sys.stderr, flush=True)
        return total

    def _build_trees(self, store: Store, known: dict, failed: dict, indexer: str, total: IndexStats) -> None:
        for doc_id in sorted(store.docs):
            pdf = self.pdf_dir / f"{doc_id}.pdf"
            if not pdf.exists():
                raise FileNotFoundError(f"pageindex: no PDF for document {doc_id!r} at {pdf}")
            sha = file_sha256(pdf)
            # Trees built before the indexer setting existed carry none; they were built with the navigator.
            built_with = known.get(doc_id, {}).get("indexer", json.dumps(self._defaults(self.navigator), sort_keys=True))
            if known.get(doc_id, {}).get("sha256") != sha or built_with != indexer:
                # The same settings can fail and then succeed (fb-johnson-johnson-2022-10k failed twice, then
                # built): each build gets ATTEMPTS tries before the document counts as failed.
                errors = []
                for _ in range(ATTEMPTS):
                    ledger = Ledger()
                    try:
                        with self._proxy.recording(ledger):
                            submitted = self._client.submit_document(str(pdf), metadata={"aex_doc_id": doc_id, "sha256": sha})
                        break
                    except Exception as exc:   # the SDK's own error types; every other tree is still built
                        errors.append(f"{type(exc).__name__}: {exc}")
                else:
                    failed[doc_id] = {"sha256": sha, "indexer": indexer, "error": errors[-1], "attempts": len(errors)}
                    self._failed_file.write_text(json.dumps(failed, indent=1, sort_keys=True))
                    continue
                failed.pop(doc_id, None)
                known[doc_id] = {"sha256": sha, "pageindex_id": submitted["doc_id"], "indexer": indexer,
                                 "cost": {"input_tokens": ledger.input_tokens, "output_tokens": ledger.output_tokens,
                                          "gpu_s": ledger.gpu_s}}
                self._map_file.write_text(json.dumps(known, indent=1, sort_keys=True))
            # The tree's build cost is reported even when it already exists (vec_tree shares these trees).
            cost = known[doc_id].get("cost", {})
            total.input_tokens += cost.get("input_tokens", 0)
            total.output_tokens += cost.get("output_tokens", 0)
            if cost.get("gpu_s") is not None:
                total.gpu_s = (total.gpu_s or 0.0) + cost["gpu_s"]
        self._failed_file.parent.mkdir(parents=True, exist_ok=True)
        self._failed_file.write_text(json.dumps(failed, indent=1, sort_keys=True))

    # ── query ──

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        self._ensure_client()
        return self._navigate(q, store, store.scope(q, self.scopes.get(q.dataset, "corpus")), Ledger())

    def _navigate(self, q: Question, store: Store, doc_ids: list[str], ledger: Ledger) -> Retrieval:
        """Run the SDK's agent over `doc_ids` (ours) and turn what it read or was shown into evidence."""
        global _active
        scope = [self._ids[d] for d in doc_ids if d in self._ids]
        missing = [d for d in doc_ids if d in self.failed]
        if not scope and missing:
            trace = {"index_failed": missing, "reads": [], "pages": [], "doc_id": None, "visited": [],
                     "backtracks": [], "selected": None, "native_answer": None}
            return Retrieval([], trace, ledger, failure="index_failed")
        target = scope[0] if len(scope) == 1 else scope
        extra = getattr(self.navigator, "extra_body", None) or None
        reads: list[tuple[float, str, list[int]]] = []
        started = time.perf_counter()
        self._shown = []
        with _lock:
            _active = reads
        gave_up = []
        try:
            # The agent can run out of its default turn budget; the same settings can then finish (as a tree
            # build can), so it gets ATTEMPTS tries. After that the method failed: scored, not retried.
            for _ in range(ATTEMPTS):
                reads.clear()
                self._shown = []
                try:
                    with self._proxy.recording(ledger):
                        sdk_answer = self._client.chat(q.question, doc_id=target, extra_body=extra)
                    break
                except Exception as exc:  # the SDK raises its own error types; the runner retries LLMError rows
                    if "max_turns" not in str(exc):
                        raise LLMError(f"pageindex: {type(exc).__name__}: {exc}") from exc
                    gave_up.append(f"{type(exc).__name__}: {exc}")
        finally:
            with _lock:
                _active = None
            shown, self._shown = self._shown, None
        if len(gave_up) == ATTEMPTS:
            trace = {"max_turns": gave_up, "reads": [], "pages": [], "doc_id": None, "visited": [],
                     "backtracks": [], "selected": None, "native_answer": None}
            return Retrieval([], trace, ledger, failure="max_turns")
        evidence, trace = self._evidence_and_trace(reads, store, started, str(sdk_answer))
        if missing:
            trace["index_failed"] = missing   # the agent searched the other documents in scope
        if gave_up:
            trace["max_turns"] = gave_up      # earlier attempts that ran out of turns
        if evidence:
            trace["answered_from"] = "pages"
        else:
            # Decided 2026-10-05: when the agent read no page, the shared answerer gets what the agent
            # was shown instead — document descriptions and structure summaries, marked as summaries
            # so they never count as retrieved pages.
            evidence = self._shown_as_evidence(shown or [], store)
            trace["answered_from"] = "summaries" if evidence else "nothing"
        return Retrieval(evidence, trace, ledger)

    def _shown_as_evidence(self, shown: list[tuple[str, str]], store: Store) -> list[EvidencePage]:
        by_pid = {pid: doc_id for doc_id, pid in self._ids.items()}
        evidence: list[EvidencePage] = []
        done: set[tuple[str, str]] = set()
        # Structures first (the more specific view), then descriptions, each in the order shown.
        for what in ("structure", "description"):
            for kind, pid in shown:
                if kind != what or (kind, pid) in done or pid not in by_pid:
                    continue
                done.add((kind, pid))
                doc_id = by_pid[pid]
                if kind == "structure":
                    nodes = self._client.get_document_structure(pid) if hasattr(self._client, "get_document_structure") \
                        else self._client.get_tree(pid, node_summary=True).get("result", [])
                    for node in _flatten(nodes):
                        if node.get("summary"):
                            evidence.append(EvidencePage(doc_id, int(node["start_index"]), f"{node['title']}: {node['summary']}",
                                                         kind="summary", end_page=int(node["end_index"])))
                else:
                    meta = self._client.get_document(pid)
                    if meta.get("description", "").strip():
                        evidence.append(EvidencePage(doc_id, 1, meta["description"].strip(), kind="summary",
                                                     end_page=int(meta.get("pageNum") or store.get(doc_id).n_pages)))
        return evidence

    def _evidence_and_trace(self, reads, store: Store, started: float, sdk_answer: str):
        evidence: list[EvidencePage] = []
        seen: set[tuple[str, int]] = set()
        trace_reads = []
        for t, name, pages in reads:
            doc_id = _doc_id_from_name(name)
            trace_reads.append({"doc_id": doc_id, "pages": pages, "t": round(t - started, 2)})
            if doc_id not in store.docs:
                continue
            doc = store.get(doc_id)
            for p in pages:
                if 1 <= p <= doc.n_pages and (doc_id, p) not in seen:
                    seen.add((doc_id, p))
                    evidence.append(EvidencePage(doc_id, p, doc.page(p).text))
        trace = {"reads": trace_reads, "pages": [[e.doc_id, e.page] for e in evidence],
                 "doc_id": None, "visited": [], "backtracks": [], "selected": None,
                 "native_answer": sdk_answer}
        if not trace_reads or trace_reads[-1]["doc_id"] not in store.docs:
            return evidence, trace
        # Map reads onto our tree of the document the agent ended in (node ids are per document).
        last = trace_reads[-1]
        doc = store.get(last["doc_id"])
        last_pages = set(last["pages"])

        def nodes_for(pages: list[int]) -> list[str]:
            return [n.id for n in doc.tree if any(n.start_page <= p <= n.end_page for p in pages)]

        visited, backtracks = [], []
        for read in trace_reads:
            if read["doc_id"] != doc.doc_id:
                continue
            ids = nodes_for(read["pages"])
            visited += [i for i in ids if i not in visited]
            if not last_pages & set(read["pages"]):
                backtracks += [i for i in ids if i not in backtracks]
        last_nodes = set(nodes_for(last["pages"]))
        backtracks = [b for b in backtracks if b not in last_nodes]
        containing = [n for n in doc.tree if n.start_page <= last["pages"][0] <= n.end_page]
        selected = max(containing, key=lambda n: (n.level, n.start_page)).id if containing else None
        trace |= {"doc_id": doc.doc_id, "visited": visited, "backtracks": backtracks, "selected": selected}
        return evidence, trace

"""PageIndex as released (SDK local mode), pointed at our endpoint; we record which pages its agent reads.

The SDK builds its own tree from the PDF and runs its own agent — that is the method under
test. What it reads is captured at `pageindex.agent_tools._parse_page_spec`, the one place
every page request passes through at call time. Those pages, in OUR shared parse, go to the
shared answerer; the SDK's own answer is kept in the trace only. Its model calls go through a
MeteringProxy so tokens, latency and sequential calls land in the question's Ledger.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from aex.common.accounting import Ledger
from aex.common.llm import ChatClient, LLMError
from aex.common.meter import MeteringProxy
from aex.experiments.tree_rag.arms.base import IndexStats, Store, register
from aex.experiments.tree_rag.parse import file_sha256
from aex.experiments.tree_rag.types import EvidencePage, Question, Retrieval

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


def _doc_id_from_name(name: str) -> str:
    return name[:-4] if name.lower().endswith(".pdf") else name


@register("pageindex")
class PageIndexArm:
    family = "tree"
    uses_navigator = True

    def __init__(self, *, navigator: ChatClient | None = None, pdf_dir: str | None = None,
                 storage_dir: str = "data/pageindex", scopes: dict | None = None, seed: int = 0,
                 **options) -> None:
        self.navigator, self.scopes, self.seed = navigator, scopes or {}, seed
        self.pdf_dir = Path(pdf_dir) if pdf_dir else None
        self.storage_dir = Path(storage_dir)
        self._client = None
        self._proxy: MeteringProxy | None = None
        self._ids: dict[str, str] = {}

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
        defaults = {"temperature": 0.0, "seed": self.seed, **(getattr(self.navigator, "extra_body", None) or {})}
        self._proxy = MeteringProxy(self.navigator.base_url, model=self.navigator.model, local=self.navigator.local,
                                    defaults=defaults)
        self._proxy.start()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._client = _client_factory(model=self.navigator.model, storage_path=str(self.storage_dir),
                                       base_url=self._proxy.base_url)

    @property
    def _map_file(self) -> Path:
        return self.storage_dir / "aex-index.json"

    def index(self, store: Store) -> IndexStats:
        self._ensure_client()
        known = json.loads(self._map_file.read_text()) if self._map_file.exists() else {}
        ledger = Ledger()
        with self._proxy.recording(ledger):
            for doc_id in sorted(store.docs):
                pdf = self.pdf_dir / f"{doc_id}.pdf"
                if not pdf.exists():
                    raise FileNotFoundError(f"pageindex: no PDF for document {doc_id!r} at {pdf}")
                sha = file_sha256(pdf)
                if known.get(doc_id, {}).get("sha256") != sha:
                    submitted = self._client.submit_document(str(pdf), metadata={"aex_doc_id": doc_id, "sha256": sha})
                    known[doc_id] = {"sha256": sha, "pageindex_id": submitted["doc_id"]}
                    self._map_file.write_text(json.dumps(known, indent=1, sort_keys=True))
        self._ids = {d: v["pageindex_id"] for d, v in known.items()}
        return IndexStats(input_tokens=ledger.input_tokens, output_tokens=ledger.output_tokens, gpu_s=ledger.gpu_s)

    # ── query ──

    def retrieve(self, q: Question, store: Store) -> Retrieval:
        global _active
        self._ensure_client()
        scope = [self._ids[d] for d in store.scope(q, self.scopes.get(q.dataset, "corpus")) if d in self._ids]
        target = scope[0] if len(scope) == 1 else scope
        extra = getattr(self.navigator, "extra_body", None) or None
        ledger = Ledger()
        reads: list[tuple[float, str, list[int]]] = []
        started = time.perf_counter()
        with _lock:
            _active = reads
        try:
            with self._proxy.recording(ledger):
                sdk_answer = self._client.chat(q.question, doc_id=target, extra_body=extra)
        except Exception as exc:  # the SDK raises its own error types; the runner retries LLMError rows
            raise LLMError(f"pageindex: {type(exc).__name__}: {exc}") from exc
        finally:
            with _lock:
                _active = None
        return Retrieval(*self._evidence_and_trace(reads, store, started, str(sdk_answer)), ledger)

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
                 "pageindex_answer": sdk_answer[:500]}
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

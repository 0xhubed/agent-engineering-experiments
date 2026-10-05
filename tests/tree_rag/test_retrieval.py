import pytest

from aex.common.embed import HashEmbedder
from aex.common.scoring import GoldAnswer
from aex.experiments.tree_rag.arms import Store
from aex.experiments.tree_rag.retrieval import BM25, ChunkIndex, chunk_doc, rrf, top_k
from aex.experiments.tree_rag.types import Page, ParsedDoc, Question


def _doc(doc_id, pages):
    return ParsedDoc(doc_id, doc_id, f"sha256:{doc_id}", tuple(Page(i + 1, t) for i, t in enumerate(pages)), ())


def _q(doc_ids=("a",), text="barrier level"):
    return Question("q", "d", "A", "lookup", text, doc_ids, GoldAnswer("numeric", "1"), (), "test")


def test_chunks_never_cross_pages_and_overlap_within_a_page():
    doc = _doc("a", [" ".join(f"w{i}" for i in range(10)), "short page", "   "])
    chunks = chunk_doc(doc, max_words=4, overlap=1)
    assert [(c.page, c.text) for c in chunks] == [
        (1, "w0 w1 w2 w3"), (1, "w3 w4 w5 w6"), (1, "w6 w7 w8 w9"), (2, "short page")]
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_params_validated():
    with pytest.raises(ValueError):
        chunk_doc(_doc("a", ["x"]), max_words=4, overlap=4)


def test_bm25_ranks_rare_matching_terms_first():
    bm = BM25(["the barrier level is sixty percent", "the coupon is paid", "the the the the"])
    s = bm.scores("barrier level")
    assert s[0] > 0 and s[1] == 0 and s[2] == 0
    assert bm.scores("the")[2] > bm.scores("the")[1]


def test_rrf_fuses_rankings():
    assert rrf([[1, 2, 3], [3, 1, 4]], k=60)[:2] == [1, 3]
    assert rrf([], k=60) == []


def test_top_k_breaks_ties_by_position():
    assert top_k([0.5, 0.9, 0.5, 0.1], 3) == [1, 0, 2]


def test_store_scopes_by_corpus_group():
    store = Store([_doc("a", ["x"]), _doc("b", ["y"]), _doc("c", ["z"])],
                  groups={"a": "termsheets", "b": "termsheets", "c": "prospectuses"})
    assert store.scope(_q(("a",)), "corpus") == ["a", "b"]
    assert store.scope(_q(("a", "c")), "corpus") == ["a", "b", "c"]
    assert store.scope(_q(("a",)), "question_docs") == ["a"]
    assert Store([_doc("a", ["x"]), _doc("b", ["y"])]).scope(_q(("a",)), "corpus") == ["a", "b"]


def test_chunk_index_dense_and_lexical_search_within_scope(tmp_path):
    store = Store([_doc("a", ["The barrier level is 60% of initial.", "Coupon dates: 15 June."]),
                   _doc("b", ["The barrier level is 65% of initial."])])
    index = ChunkIndex(store, HashEmbedder(dim=128), max_words=50, overlap=0, cache_dir=tmp_path)
    stats = index.build()
    assert stats.input_tokens > 0 and len(index.chunks) == 3
    qv = HashEmbedder(dim=128).embed_query("barrier level").vectors[0]
    hits = [i for i, _ in index.dense(qv, ["a"], k=5)]
    assert [index.chunks[i].doc_id for i in hits] == ["a", "a"]
    assert index.chunks[hits[0]].page == 1
    lex = index.lexical("coupon dates", ["a", "b"], k=1)
    assert index.chunks[lex[0][0]].page == 2

    # second build reads the cache: no embedding tokens spent
    again = ChunkIndex(store, HashEmbedder(dim=128), max_words=50, overlap=0, cache_dir=tmp_path)
    assert again.build().input_tokens == 0
    assert [i for i, _ in again.dense(qv, ["a"], k=1)] == hits[:1]

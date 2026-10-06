import json
import math

import httpx
import pytest

from aex.common.embed import (HashEmbedder, OpenAICompatEmbedder, OpenAICompatReranker,
                              OverlapReranker, cosine)
from aex.common.llm import LLMError


def test_hash_embedder_is_deterministic_normalised_and_lexical():
    e = HashEmbedder(dim=64)
    a, b, c = e.embed(["barrier level 60%", "barrier level 60%", "coupon payment dates"]).vectors
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    assert cosine(a, b) > cosine(a, c)


def test_query_prefix_only_applies_to_queries():
    seen = []

    def handler(request):
        body = json.loads(request.read())
        seen.append(body["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))],
                                         "usage": {"prompt_tokens": 7}})

    e = OpenAICompatEmbedder("http://x/v1", "emb", query_prefix="Instruct: find it\nQuery: ",
                             transport=httpx.MockTransport(handler))
    docs = e.embed(["page text"])
    q = e.embed_query("what barrier?")
    assert seen == [["page text"], ["Instruct: find it\nQuery: what barrier?"]]
    assert docs.vectors == [[0.6, 0.8]] and docs.input_tokens == 7
    assert q.vectors == [[0.6, 0.8]]


def test_embedder_batches_and_keeps_order():
    def handler(request):
        body = json.loads(request.read())
        # return out of order: the client must sort by index
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])][::-1]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": len(body["input"])}})

    e = OpenAICompatEmbedder("http://x/v1", "emb", batch_size=2, transport=httpx.MockTransport(handler))
    out = e.embed(["a", "bbb", "cc"])
    assert [round(v[0] / v[1]) for v in out.vectors] == [1, 3, 2]
    assert out.input_tokens == 3


def test_embedder_raises_llmerror_after_5xx():
    e = OpenAICompatEmbedder("http://x/v1", "emb", retries=2, backoff_s=0,
                             transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(LLMError):
        e.embed(["x"])


def test_reranker_returns_scores_in_input_order():
    def handler(request):
        body = json.loads(request.read())
        assert body["query"] == "q" and body["documents"] == ["d0", "d1", "d2"]
        return httpx.Response(200, json={"results": [{"index": 2, "relevance_score": 0.9},
                                                     {"index": 0, "relevance_score": 0.5},
                                                     {"index": 1, "relevance_score": 0.1}],
                                         "usage": {"total_tokens": 30}})

    r = OpenAICompatReranker("http://x/v1", "rr", transport=httpx.MockTransport(handler))
    out = r.rerank("q", ["d0", "d1", "d2"])
    assert out.scores == [0.5, 0.1, 0.9] and out.input_tokens == 30


def test_overlap_reranker_prefers_shared_words():
    out = OverlapReranker().rerank("barrier level", ["coupon dates", "the barrier level is 60%"])
    assert out.scores[1] > out.scores[0]


def test_reranker_applies_query_and_document_templates():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}], "usage": {"total_tokens": 5}})

    r = OpenAICompatReranker("http://x/v1", "rr", transport=httpx.MockTransport(handler),
                             query_template="<Q>{query}</Q>", document_template="<D>{document}</D>")
    r.rerank("barrier?", ["60%"])
    assert seen["query"] == "<Q>barrier?</Q>" and seen["documents"] == ["<D>60%</D>"]


def test_templates_tolerate_braces_in_text():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.1}]})

    r = OpenAICompatReranker("http://x/v1", "rr", transport=httpx.MockTransport(handler),
                             document_template="<D>{document}</D>")
    r.rerank("q", ["a {weird} value"])
    assert seen["documents"] == ["<D>a {weird} value</D>"]

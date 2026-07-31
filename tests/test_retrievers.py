"""Tests for the retrievers: BM25 relevance and hybrid RRF fusion order."""

from __future__ import annotations

from ragkit.retrievers.hybrid import HybridRetriever
from ragkit.types import Chunk, ScoredChunk


def _doc_ids(hits: list[ScoredChunk]) -> list[str]:
    out: list[str] = []
    for sc in hits:
        if sc.chunk.doc_id not in out:
            out.append(sc.chunk.doc_id)
    return out


# -- BM25 (via the offline corpus fixture) ------------------------------

def test_bm25_ranks_keyword_document_first(corpus):
    hits = corpus.bm25.retrieve("how many hours can I work each week", k=5)
    assert hits, "expected some BM25 hits"
    assert _doc_ids(hits)[0] == "working"


def test_bm25_finds_graduate_route(corpus):
    hits = corpus.bm25.retrieve("stay in the UK after graduating PhD", k=5)
    assert _doc_ids(hits)[0] == "graduate_route"


def test_bm25_returns_nothing_for_unrelated_terms(corpus):
    assert corpus.bm25.retrieve("xylophone quantum banana", k=5) == []


# -- Hybrid RRF fusion (deterministic, with fake retrievers) ------------

def _mk(cid: str) -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("#")[0], ordinal=0, text="")


class _FakeRetriever:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=_mk(i), score=1.0) for i in self._ids[:k]]


def test_rrf_fuses_rankings_correctly():
    sparse = _FakeRetriever(["a", "b", "c"])
    dense = _FakeRetriever(["b", "c", "d"])
    hybrid = HybridRetriever(sparse, dense, rrf_k=60, candidate_k=10)

    result = [sc.chunk.id for sc in hybrid.retrieve("q", k=4)]
    # RRF: b appears in both (ranks 2 & 1) -> highest; then c (3 & 2);
    # then a (1/61) edges out d (1/63).
    assert result == ["b", "c", "a", "d"]


def test_rrf_respects_top_k():
    sparse = _FakeRetriever(["a", "b", "c"])
    dense = _FakeRetriever(["d", "e", "f"])
    hybrid = HybridRetriever(sparse, dense, rrf_k=60, candidate_k=10)
    assert len(hybrid.retrieve("q", k=3)) == 3

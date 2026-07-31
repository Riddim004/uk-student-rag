"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

BM25 (lexical) and dense (semantic) retrievers have complementary failure
modes: BM25 nails exact keywords but misses paraphrases; dense catches meaning
but drifts on rare terms and numbers. RRF fuses their *rankings* (not their
raw scores, which live on different scales) into one list.

RRF score for a chunk = sum over retrievers of  weight / (rrf_k + rank),
where rank is 1-based within that retriever's list.
"""

from __future__ import annotations

from ..types import Chunk, ScoredChunk
from .base import Retriever


class HybridRetriever:
    """Fuses several retrievers' rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self._sparse = sparse
        self._dense = dense
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k
        self._sparse_weight = sparse_weight
        self._dense_weight = dense_weight

    def _rrf_contribution(
        self,
        results: list[ScoredChunk],
        weight: float,
        fused: dict[str, float],
        chunks: dict[str, Chunk],
    ) -> None:
        for rank, sc in enumerate(results, start=1):
            cid = sc.chunk.id
            chunks[cid] = sc.chunk
            fused[cid] = fused.get(cid, 0.0) + weight / (self._rrf_k + rank)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        sparse_hits = self._sparse.retrieve(query, self._candidate_k)
        dense_hits = self._dense.retrieve(query, self._candidate_k)

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        self._rrf_contribution(sparse_hits, self._sparse_weight, fused, chunks)
        self._rrf_contribution(dense_hits, self._dense_weight, fused, chunks)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [ScoredChunk(chunk=chunks[cid], score=score) for cid, score in ranked[:k]]

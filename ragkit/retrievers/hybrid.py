"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

BM25 (lexical) and dense (semantic) retrievers have complementary failure
modes: BM25 nails exact keywords but misses paraphrases; dense catches meaning
but drifts on rare terms and numbers. RRF fuses their *rankings* (not their
raw scores, which live on different scales) into one list.

RRF score for a chunk = sum over retrievers of weight / (rrf_k + rank),
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
        candidate_k: int = 50,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        self._sparse = sparse
        self._dense = dense
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k
        self._sparse_weight = sparse_weight
        self._dense_weight = dense_weight

    def _contribute(
        self,
        results: list[ScoredChunk],
        name: str,
        weight: float,
        fused: dict[str, float],
        chunks: dict[str, Chunk],
        components: dict[str, dict[str, float]],
    ) -> None:
        for rank, sc in enumerate(results, start=1):
            cid = sc.chunk.id
            chunks[cid] = sc.chunk
            item = components.setdefault(cid, {})
            item[name] = sc.components.get(name, sc.score)
            item[f"{name}_rank"] = float(rank)
            fused[cid] = fused.get(cid, 0.0) + weight / (self._rrf_k + rank)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        sparse_hits = self._sparse.retrieve(query, self._candidate_k)
        dense_hits = self._dense.retrieve(query, self._candidate_k)

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        components: dict[str, dict[str, float]] = {}
        self._contribute(
            sparse_hits, "sparse", self._sparse_weight, fused, chunks, components
        )
        self._contribute(
            dense_hits, "dense", self._dense_weight, fused, chunks, components
        )

        for cid, score in fused.items():
            components[cid]["rrf"] = score
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [
            ScoredChunk(chunk=chunks[cid], score=score, components=components[cid])
            for cid, score in ranked[:k]
        ]

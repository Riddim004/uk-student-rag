"""Cross-encoder reranking.

Fusion optimises for recall; a cross-encoder then reads each (query, chunk)
pair and reorders the candidate pool for precision. The reranker is optional so
the CPU-only path stays usable.
"""

from __future__ import annotations

from typing import Protocol

from ..errors import ModelError
from ..types import ScoredChunk
from .base import Retriever


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        ...


class CrossEncoderReranker:
    """Rerank with a sentence-transformers CrossEncoder (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ModelError(
                "sentence-transformers is not installed; install "
                "requirements-local.txt or disable reranking."
            ) from exc
        self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(self, query: str, chunks: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        model = self._ensure_model()
        pairs = [(query, sc.chunk.text) for sc in chunks]
        scores = model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        if len(scores) != len(chunks):
            raise ModelError(
                f"reranker returned {len(scores)} scores for {len(chunks)} pairs"
            )
        reranked = [
            ScoredChunk(
                chunk=sc.chunk,
                score=float(score),
                components={**sc.components, "rerank": float(score)},
            )
            for sc, score in zip(chunks, scores)
        ]
        reranked.sort(key=lambda sc: sc.score, reverse=True)
        return reranked[:k]


class NoOpReranker:
    """Identity reranker, useful in tests and when reranking is disabled."""

    def rerank(self, query: str, chunks: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        return chunks[:k]


class RerankedRetriever:
    """Retriever wrapper: fetch a candidate pool, then rerank it."""

    def __init__(self, base: Retriever, reranker: Reranker, candidate_k: int = 30) -> None:
        self._base = base
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        candidates = self._base.retrieve(query, max(k, self._candidate_k))
        return self._reranker.rerank(query, candidates, k)

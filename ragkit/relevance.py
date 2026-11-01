"""Relevance gating: decide whether retrieval is good enough to answer.

RRF scores are rank-based and always positive, so they cannot answer "is this
query in the corpus at all?". The gate therefore reads the underlying component
scores (dense cosine, BM25, or reranker logit) instead of the fused score.

Two gates ship:

* ``RelevanceGate`` — threshold on the strongest dense/BM25 score. Cheap, but
  the calibration report shows it cannot separate out-of-domain questions.
* ``RerankGate`` — run a cross-encoder over the hybrid candidates and use the
  strongest rerank score only as an answer/refuse decision. It deliberately
  keeps the hybrid order, because reranking the output hurt MRR in the eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .retrievers.rerank import Reranker
from .types import ScoredChunk


@dataclass(frozen=True)
class GateConfig:
    min_dense_score: float = 0.30
    min_sparse_score: float = 0.0
    min_rerank_score: float = -3.0


class Gate(Protocol):
    def filter(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        ...

    def would_refuse(self, query: str, chunks: list[ScoredChunk]) -> bool:
        ...


class RelevanceGate:
    """Keep chunks whose underlying retriever score clears its threshold."""

    def __init__(self, config: GateConfig | None = None) -> None:
        self._config = config or GateConfig()

    @classmethod
    def from_settings(cls, settings: Settings) -> "RelevanceGate":
        return cls(
            GateConfig(
                min_dense_score=settings.min_dense_score,
                min_sparse_score=settings.min_sparse_score,
                min_rerank_score=settings.min_rerank_score,
            )
        )

    def keep(self, sc: ScoredChunk) -> bool:
        components = sc.components
        if "rerank" in components:
            return components["rerank"] >= self._config.min_rerank_score
        if not components:
            # Fake retriever in unit tests: trust its explicit score.
            return sc.score > 0.0
        sparse = components.get("sparse", 0.0)
        dense = components.get("dense", 0.0)
        return sparse > self._config.min_sparse_score or dense >= self._config.min_dense_score

    def _max_component(
        self, chunks: list[ScoredChunk], name: str, default: float
    ) -> float:
        return max((sc.components.get(name, default) for sc in chunks), default=default)

    def should_answer(self, query: str, chunks: list[ScoredChunk]) -> bool:
        """Query-level gate: use the strongest evidence, not any weak hit."""
        if not chunks:
            return False
        if any("rerank" in sc.components for sc in chunks):
            return (
                self._max_component(chunks, "rerank", float("-inf"))
                >= self._config.min_rerank_score
            )
        dense = self._max_component(chunks, "dense", float("-inf"))
        sparse = self._max_component(chunks, "sparse", float("-inf"))
        return (
            dense >= self._config.min_dense_score
            or sparse > self._config.min_sparse_score
        )

    def filter(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        if not self.should_answer(query, chunks):
            return []
        return [sc for sc in chunks if self.keep(sc)]

    def would_refuse(self, query: str, chunks: list[ScoredChunk]) -> bool:
        return not self.should_answer(query, chunks)


class RerankGate:
    """Use a cross-encoder's best score as a refusal decision, not a ranking."""

    def __init__(
        self,
        reranker: Reranker,
        min_score: float = -3.0,
        candidate_k: int = 15,
    ) -> None:
        self._reranker = reranker
        self._min_score = min_score
        self._candidate_k = candidate_k

    def _annotate(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        if not chunks:
            return []
        candidates = chunks[: self._candidate_k]
        reranked = self._reranker.rerank(query, candidates, len(candidates))
        scores = {
            sc.chunk.id: sc.components.get("rerank", float("-inf"))
            for sc in reranked
        }
        return [
            ScoredChunk(
                chunk=sc.chunk,
                score=sc.score,
                components={
                    **sc.components,
                    "rerank": scores.get(sc.chunk.id, float("-inf")),
                },
            )
            for sc in chunks
        ]

    @property
    def threshold(self) -> float:
        return self._min_score

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    def annotate(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Attach rerank scores without reordering; used by the pipeline trace."""
        return self._annotate(query, chunks)

    def decide(self, annotated: list[ScoredChunk]) -> tuple[bool, float]:
        best = max(
            (sc.components.get("rerank", float("-inf")) for sc in annotated),
            default=float("-inf"),
        )
        return best >= self._min_score, best

    def should_answer(self, query: str, chunks: list[ScoredChunk]) -> bool:
        accepted, _best = self.decide(self._annotate(query, chunks))
        return accepted

    def filter(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        annotated = self._annotate(query, chunks)
        accepted, _best = self.decide(annotated)
        return annotated if accepted else []

    def would_refuse(self, query: str, chunks: list[ScoredChunk]) -> bool:
        return not self.should_answer(query, chunks)

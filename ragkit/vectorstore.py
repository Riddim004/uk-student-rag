"""A minimal in-memory vector store with cosine similarity.

Vectors are L2-normalised on the way in, so cosine similarity is a plain dot
product. This is intentionally simple — swapping in FAISS/Chroma later only
means implementing the same tiny surface. Kept dependency-free (no numpy) so
tests need nothing extra.
"""

from __future__ import annotations

from .types import Chunk, ScoredChunk


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore:
    """Stores chunks and their embeddings; searches by cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=_dot(query_vector, vec))
            for chunk, vec in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:k]

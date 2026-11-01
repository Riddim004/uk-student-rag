"""Dense (embedding-based) retriever backed by the vector store."""

from __future__ import annotations

from ..embeddings import Embedder
from ..types import ScoredChunk
from ..vectorstore import InMemoryVectorStore


class DenseRetriever:
    """Embeds the query and searches the vector store by cosine similarity."""

    def __init__(self, embedder: Embedder, store: InMemoryVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        if len(self._store) == 0:
            return []
        query_vec = self._embedder.embed_query(query)
        hits = self._store.search(query_vec, k)
        return [
            ScoredChunk(
                chunk=hit.chunk,
                score=hit.score,
                components={"dense": hit.score},
            )
            for hit in hits
        ]

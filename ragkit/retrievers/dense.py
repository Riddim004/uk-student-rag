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
        query_vec = self._embedder.embed([query])[0]
        return self._store.search(query_vec, k)

"""Ingestion: load markdown documents from a directory and chunk them.

A ``Corpus`` bundles the chunks plus the embedded vector store, and builds the
three retrievers off the same chunk set so they are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_document
from .config import Settings
from .embeddings import Embedder
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from .types import Chunk, Document
from .vectorstore import InMemoryVectorStore


def load_documents(directory: str | Path) -> list[Document]:
    """Load every ``*.md`` file in a directory as a Document.

    The first ``# `` heading (if any) becomes the title; the file name is the id.
    """
    directory = Path(directory)
    docs: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = ""
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(Document(id=path.stem, text=text, title=title, source=str(path)))
    return docs


@dataclass
class Corpus:
    """Chunked, embedded corpus with the three retrievers wired up."""

    chunks: list[Chunk]
    store: InMemoryVectorStore
    bm25: BM25Retriever
    dense: DenseRetriever
    hybrid: HybridRetriever


def build_corpus(
    documents: list[Document],
    embedder: Embedder,
    settings: Settings | None = None,
) -> Corpus:
    """Chunk → embed → index, and construct sparse/dense/hybrid retrievers."""
    settings = settings or Settings()

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))

    store = InMemoryVectorStore()
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.add(chunks, vectors)

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(embedder, store)
    hybrid = HybridRetriever(
        sparse=bm25,
        dense=dense,
        rrf_k=settings.rrf_k,
        candidate_k=settings.candidate_k,
        sparse_weight=settings.sparse_weight,
        dense_weight=settings.dense_weight,
    )
    return Corpus(chunks=chunks, store=store, bm25=bm25, dense=dense, hybrid=hybrid)

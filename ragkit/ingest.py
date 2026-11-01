"""Ingestion: load sections, chunk them, embed them, build retrievers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import chunk_documents
from .config import Settings
from .embeddings import Embedder
from .errors import ConfigurationError, DataError, EmbeddingError
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from .types import Chunk, Document
from .vectorstore import InMemoryVectorStore


def _document_from_dict(data: dict[str, Any]) -> Document:
    try:
        return Document(
            id=data["id"],
            text=data["text"],
            title=data.get("title", ""),
            source=data.get("source", ""),
            doc_id=data.get("doc_id", ""),
            section_id=data.get("section_id", ""),
            section_path=tuple(data.get("section_path") or ()),
            updated_at=data.get("updated_at", ""),
            fetched_at=data.get("fetched_at", ""),
            source_type=data.get("source_type", ""),
            ordinal=int(data.get("ordinal", 0)),
        )
    except KeyError as exc:
        raise DataError(f"document is missing required field {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise DataError(f"document has an invalid field: {exc}") from exc


def load_documents_jsonl(path: str | Path) -> list[Document]:
    """Load the section-level corpus produced by ``scripts/build_data.py``.

    Malformed input fails loudly with the file and line number; an empty
    corpus is an error, because answering from nothing would look like a
    relevance-gate refusal rather than a data problem.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"corpus not found: {path}. Run `python scripts/build_data.py --offline` "
            "or point --data at data/corpus/documents.jsonl."
        )
    documents: list[Document] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
            if not isinstance(data, dict):
                raise DataError(f"{path}:{lineno}: expected a JSON object")
            document = _document_from_dict(data)
            if not document.id.strip():
                raise DataError(f"{path}:{lineno}: empty document id")
            if not document.text.strip():
                raise DataError(f"{path}:{lineno}: document {document.id!r} has no text")
            if document.id in seen:
                raise DataError(f"{path}:{lineno}: duplicate document id {document.id!r}")
            seen.add(document.id)
            documents.append(document)
    if not documents:
        raise DataError(f"corpus is empty: {path}")
    return documents


def load_documents(directory: str | Path) -> list[Document]:
    """Legacy loader for a directory of markdown files.

    Kept for small fixtures; the real corpus goes through ``load_documents_jsonl``.
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
        docs.append(
            Document(
                id=path.stem,
                text=text,
                title=title,
                source=str(path),
                doc_id=path.stem,
                section_id=path.stem,
                section_path=(title or path.stem,),
            )
        )
    return docs


@dataclass
class Corpus:
    """Chunked, embedded corpus with the three retrievers wired up."""

    documents: list[Document]
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
    if not documents:
        raise DataError("cannot build a corpus from zero documents")
    if settings.chunk_size <= 0:
        raise ConfigurationError("chunk_size must be positive")
    if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
        raise ConfigurationError("chunk_overlap must be in [0, chunk_size)")

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise DataError("chunking produced zero chunks")

    store = InMemoryVectorStore()
    vectors = embedder.embed_documents([c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise EmbeddingError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        )
    dims = {len(vector) for vector in vectors}
    if len(dims) != 1:
        raise EmbeddingError(f"embedder returned inconsistent dimensions: {sorted(dims)}")
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
    return Corpus(
        documents=documents,
        chunks=chunks,
        store=store,
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
    )

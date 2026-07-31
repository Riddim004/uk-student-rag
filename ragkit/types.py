"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    id: str
    text: str
    title: str = ""
    source: str = ""  # URL or file path


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document."""

    id: str  # e.g. "doc1#3"
    doc_id: str
    ordinal: int
    text: str
    title: str = ""
    source: str = ""


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk paired with a retrieval score (higher = more relevant)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    """A reference surfaced alongside a generated answer."""

    marker: int  # the [n] used inline
    chunk_id: str
    doc_id: str
    title: str
    source: str


@dataclass(frozen=True)
class Answer:
    """The final RAG output."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    contexts: list[ScoredChunk] = field(default_factory=list)

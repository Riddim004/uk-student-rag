"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A retrievable source section before chunking.

    ``id`` is the stable section id (one part of a gov.uk guide, or one H2
    section of a detailed guide). ``doc_id`` is the parent source page.
    """

    id: str
    text: str
    title: str = ""
    source: str = ""  # canonical URL or file path
    doc_id: str = ""
    section_id: str = ""
    section_path: tuple[str, ...] = ()
    updated_at: str = ""
    fetched_at: str = ""
    source_type: str = ""
    ordinal: int = 0


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document."""

    id: str  # e.g. "student-visa__money#3"
    doc_id: str
    ordinal: int
    text: str
    title: str = ""
    source: str = ""
    section_id: str = ""
    section_ids: tuple[str, ...] = ()
    section_path: tuple[str, ...] = ()
    updated_at: str = ""
    fetched_at: str = ""
    source_type: str = ""


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk paired with a retrieval score (higher = more relevant).

    ``components`` keeps the per-retriever scores that produced the final
    ranking (sparse, dense, rrf, rerank). Keeping them makes the relevance gate
    and the "why this answer" UI possible without re-running retrievers.
    """

    chunk: Chunk
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """A reference surfaced alongside a generated answer."""

    marker: int  # the [n] used inline
    chunk_id: str
    doc_id: str
    title: str
    source: str
    section_path: tuple[str, ...] = ()
    updated_at: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class Answer:
    """The final RAG output."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    contexts: list[ScoredChunk] = field(default_factory=list)
    refused_reason: str = ""
    mode: str = ""
    degraded: bool = False
    degraded_reason: str = ""

"""Section-aware, overlapping word-window chunking.

Each ``Document`` is already a meaningful section (a gov.uk guide part or an H2
section of a detailed guide). The chunker keeps that section boundary and adds
an explicit context header, so a chunk never loses the page/section it came
from. Word-window overlap preserves context across boundaries.
"""

from __future__ import annotations

import re

from .types import Chunk, Document

_WORD = re.compile(r"\S+")


def _tokenize_words(text: str) -> list[str]:
    return _WORD.findall(text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split raw text into overlapping word windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = _tokenize_words(text)
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _context_header(doc: Document) -> str:
    section = doc.section_path[-1] if doc.section_path else (doc.title or doc.id)
    document_title = doc.title or doc.doc_id or doc.id
    return f"[Document: {document_title} | Section: {section}]"


def chunk_document(
    doc: Document,
    chunk_size: int,
    overlap: int,
    include_context: bool = True,
) -> list[Chunk]:
    """Chunk one section, carrying its metadata onto every chunk."""
    pieces = chunk_text(doc.text, chunk_size, overlap)
    header = _context_header(doc) if include_context else ""
    section_id = doc.section_id or doc.id
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        text = f"{header}\n{piece}" if header else piece
        chunks.append(
            Chunk(
                id=f"{doc.id}#{i}",
                doc_id=doc.doc_id or doc.id,
                ordinal=i,
                text=text,
                title=doc.title,
                source=doc.source,
                section_id=section_id,
                section_ids=(section_id,),
                section_path=doc.section_path,
                updated_at=doc.updated_at,
                fetched_at=doc.fetched_at,
                source_type=doc.source_type,
            )
        )
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    overlap: int,
    include_context: bool = True,
) -> list[Chunk]:
    """Chunk every section, preserving corpus order."""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunk_size, overlap, include_context))
    return chunks

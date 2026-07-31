"""Word-window chunking with overlap.

Deliberately simple and pure: given a document and a (size, overlap) in words,
produce deterministic, overlapping chunks. Overlap preserves context that would
otherwise be split across a boundary. A paragraph-aware splitter is noted as
future work in the design doc.
"""

from __future__ import annotations

import re

from .types import Chunk, Document

_WORD = re.compile(r"\S+")


def _tokenize_words(text: str) -> list[str]:
    return _WORD.findall(text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split raw text into overlapping word windows.

    ``overlap`` must be smaller than ``chunk_size`` so the window always
    advances; this is validated to avoid an infinite loop.
    """
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


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a document, carrying its metadata onto every chunk."""
    pieces = chunk_text(doc.text, chunk_size, overlap)
    return [
        Chunk(
            id=f"{doc.id}#{i}",
            doc_id=doc.id,
            ordinal=i,
            text=piece,
            title=doc.title,
            source=doc.source,
        )
        for i, piece in enumerate(pieces)
    ]

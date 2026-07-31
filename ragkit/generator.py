"""Answer generation: grounded, cited, and willing to refuse.

The generator turns retrieved chunks into an ``Answer``. Two guarantees matter
for a trustworthy RAG system and both are enforced here:

* **Grounding + citations** — the prompt forbids using outside knowledge and
  requires inline ``[n]`` markers; we then map those markers back to the exact
  chunks to build verifiable citations.
* **Refusal** — if retrieval returns nothing usable, we refuse *before* calling
  the model, so the system says "I don't know" instead of hallucinating.

The chat model is injected via the tiny ``ChatModel`` protocol, so tests run
with a fake and need no API.
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import Settings
from .types import Answer, Citation, ScoredChunk

_CITATION = re.compile(r"\[(\d+)\]")

REFUSAL_TEXT = (
    "I don't have enough information in the provided sources to answer that. "
    "Please check the official guidance."
)

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions strictly from the numbered "
    "context passages provided. Rules:\n"
    "- Use ONLY the context; never rely on outside knowledge.\n"
    "- Cite every claim inline with the passage number, e.g. [1].\n"
    "- If the context does not contain the answer, reply exactly: "
    f"\"{REFUSAL_TEXT}\"\n"
    "- Be concise and factual."
)


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the assistant reply for a system + user message."""
        ...


def build_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    blocks = []
    for i, sc in enumerate(chunks, start=1):
        header = sc.chunk.title or sc.chunk.doc_id
        blocks.append(f"[{i}] {header}\n{sc.chunk.text}")
    return "\n\n".join(blocks)


def _citations_from_text(text: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Build Citation objects for the markers actually used in ``text``."""
    used = sorted({int(n) for n in _CITATION.findall(text)})
    citations = []
    for marker in used:
        if 1 <= marker <= len(chunks):
            chunk = chunks[marker - 1].chunk
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    source=chunk.source,
                )
            )
    return citations


class LLMGenerator:
    """Generates grounded, cited answers with a refusal fallback."""

    def __init__(self, chat_model: ChatModel, settings: Settings | None = None) -> None:
        self._chat = chat_model
        self._settings = settings or Settings()

    def _usable(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        return [c for c in chunks if c.score >= self._settings.min_context_score]

    def generate(self, query: str, chunks: list[ScoredChunk]) -> Answer:
        usable = self._usable(chunks)
        if not usable and self._settings.refuse_when_empty:
            return Answer(text=REFUSAL_TEXT, citations=[], refused=True, contexts=[])

        context = build_context(usable)
        user = f"Context:\n{context}\n\nQuestion: {query}"
        text = self._chat.complete(SYSTEM_PROMPT, user).strip()

        refused = text.strip() == REFUSAL_TEXT
        citations = [] if refused else _citations_from_text(text, usable)
        return Answer(text=text, citations=citations, refused=refused, contexts=usable)

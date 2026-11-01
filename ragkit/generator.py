"""Answer generation: grounded, cited, and willing to refuse.

The generator receives only contexts that already passed the relevance gate. It
then enforces two guarantees:

* **Grounding + citations** — the prompt forbids outside knowledge and requires
  inline ``[n]`` markers; markers are mapped back to the exact chunks, and
  out-of-range markers are dropped.
* **Refusal** — the pipeline refuses before this stage when retrieval is empty
  or irrelevant; the model is also instructed to refuse when the context does
  not answer the question.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from collections import Counter
from typing import Protocol

from .config import Settings
from .errors import GenerationError
from .tokenization import tokenize
from .types import Answer, Citation, ScoredChunk

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=[。！？])")
_STOPWORDS = {
    "a", "about", "after", "also", "am", "an", "and", "are", "as", "at",
    "be", "been", "before", "being", "by", "can", "could", "did", "do",
    "does", "during", "for", "from", "had", "has", "have", "he", "her",
    "his", "how", "i", "if", "in", "is", "it", "its", "many", "may", "me",
    "might", "much", "must", "my", "no", "not", "of", "on", "or", "our",
    "per", "should", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "yes", "you", "your",
}
_NUMERIC_QUESTION = re.compile(r"\b(how (much|many|long)|cost|fee|price|salary)\b", re.I)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_UNIT_VALUE = re.compile(
    r"(£|\$|\b\d+(?:\.\d+)?\s*(?:hours?|days?|weeks?|months?|years?|%))",
    re.I,
)


def _loose_terms(text: str) -> set[str]:
    """Tokenize, and split hyphenated words so "term-time" matches "term"."""
    terms: set[str] = set()
    for token in tokenize(text):
        terms.add(token)
        if "-" in token:
            terms.update(part for part in token.split("-") if part)
    return terms


def clean_quote(text: str) -> str:
    """Turn markdown/table soup into a readable quote."""
    text = _MD_LINK.sub(r"\1", text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
    text = re.sub(r"\|\s*:?-{3,}:?\s*", " ", text)
    text = text.replace("**", "").replace("##", "").replace("•", "")
    text = re.sub(r"\s*\|\s*", "; ", text)
    text = re.sub(r"(;\s*)+", "; ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;")


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
    "- Answer in the same language as the question.\n"
    "- Be concise and factual."
)


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the assistant reply for a system + user message."""
        ...


class Generator(Protocol):
    def generate(self, query: str, chunks: list[ScoredChunk]) -> Answer: ...

    def refuse(self, reason: str) -> Answer: ...


def body_text(text: str) -> str:
    """Drop the chunk's retrieval header before showing text to a reader/model."""
    if text.startswith("[Document:"):
        _header, _, rest = text.partition("]\n")
        return rest.strip()
    return text


def build_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    blocks = []
    for i, sc in enumerate(chunks, start=1):
        chunk = sc.chunk
        title = chunk.title or chunk.doc_id
        section = chunk.section_path[-1] if chunk.section_path else ""
        header = f"{title} — {section}" if section and section != title else title
        updated = f" | updated {chunk.updated_at}" if chunk.updated_at else ""
        blocks.append(f"[{i}] {header}{updated}\n{body_text(chunk.text)}")
    return "\n\n".join(blocks)


def _normalise(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _is_refusal(text: str) -> bool:
    normalised = _normalise(text)
    target = _normalise(REFUSAL_TEXT)
    return normalised == target or normalised.startswith(target)


def _citations_from_text(text: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Build Citation objects for the markers actually used in ``text``."""
    used = sorted({int(n) for n in _CITATION.findall(text)})
    citations = []
    for marker in used:
        if 1 <= marker <= len(chunks):
            chunk = chunks[marker - 1].chunk
            snippet = " ".join(body_text(chunk.text).split())[:240]
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    source=chunk.source,
                    section_path=chunk.section_path,
                    updated_at=chunk.updated_at,
                    snippet=snippet,
                )
            )
    return citations


class LLMGenerator:
    """Generates grounded, cited answers with a refusal fallback."""

    def __init__(self, chat_model: ChatModel, settings: Settings | None = None) -> None:
        self._chat = chat_model
        self._settings = settings or Settings()

    def refuse(self, reason: str) -> Answer:
        return Answer(
            text=REFUSAL_TEXT,
            citations=[],
            refused=True,
            contexts=[],
            refused_reason=reason,
            mode="refusal",
        )

    def generate(self, query: str, chunks: list[ScoredChunk]) -> Answer:
        if not chunks:
            if self._settings.refuse_when_empty:
                return self.refuse("no_relevant_context")
            return self.refuse("empty_context")

        context = build_context(chunks)
        user = f"Context:\n{context}\n\nQuestion: {query}"
        text = self._chat.complete(SYSTEM_PROMPT, user).strip()

        if not text:
            return self.refuse("empty_model_response")
        if _is_refusal(text):
            return Answer(
                text=REFUSAL_TEXT,
                citations=[],
                refused=True,
                contexts=chunks,
                refused_reason="model_refusal",
                mode="refusal",
            )

        citations = _citations_from_text(text, chunks)
        return Answer(
            text=text,
            citations=citations,
            refused=False,
            contexts=chunks,
            mode="llm",
        )


class ExtractiveGenerator:
    """Deterministic, grounded fallback: quote the best-matching sentences.

    It is not a language model and does not claim to be; every sentence is
    copied from a retrieved chunk and carries that chunk's citation marker.
    Used for the no-API-key demo and as a degraded path when the chat model
    fails after retries.
    """

    def __init__(
        self,
        max_sentences: int = 1,
        max_words: int = 110,
        max_sentence_words: int = 35,
    ) -> None:
        self._max_sentences = max_sentences
        self._max_words = max_words
        self._max_sentence_words = max_sentence_words

    def refuse(self, reason: str) -> Answer:
        return Answer(
            text=REFUSAL_TEXT,
            citations=[],
            refused=True,
            contexts=[],
            refused_reason=reason,
            mode="refusal",
        )

    @staticmethod
    def _sentences(text: str) -> list[str]:
        flat = " ".join(clean_quote(body_text(text)).split())
        return [part.strip() for part in _SENTENCE.split(flat) if part.strip()]

    def _trim(self, sentence: str, terms: set[str]) -> str:
        """Keep the densest window of a very long (often table-shaped) sentence."""
        words = sentence.split()
        if len(words) <= self._max_sentence_words:
            return sentence
        limit = self._max_sentence_words
        best_start, best_count = 0, -1
        for start in range(0, len(words) - limit + 1):
            window = " ".join(words[start : start + limit])
            count = len(terms & _loose_terms(window))
            if count > best_count:
                best_start, best_count = start, count
            if count == len(terms):
                break
        if best_count <= 0:
            # No lexical overlap (common for a Chinese question over English
            # sources): centre the excerpt on a concrete value with a unit.
            match = _UNIT_VALUE.search(sentence) or re.search(r"\d", sentence)
            if match:
                anchor = len(sentence[: match.start()].split())
                best_start = max(0, min(anchor - limit // 3, len(words) - limit))
        trimmed = " ".join(words[best_start : best_start + limit])
        if best_start > 0:
            trimmed = "…" + trimmed
        if best_start + limit < len(words):
            trimmed = trimmed + "…"

        # Table rows arrive as semicolon-joined fragments. Keep the fragment
        # with the strongest query overlap instead of quoting half a clause.
        segments = [part.strip(" …") for part in re.split(r";\s*", trimmed) if part.strip()]
        if len(segments) > 1:
            best = max(
                segments,
                key=lambda part: (
                    len(terms & _loose_terms(part)),
                    bool(_UNIT_VALUE.search(part)),
                    bool(re.search(r"\d", part)),
                ),
            )
            best = best.strip(" ;,")
            if best.count("(") > best.count(")"):
                best = best[: best.rfind("(")].strip(" ;,")
            if len(best.split()) >= 5:
                return best
        return trimmed

    def _select(self, query: str, chunks: list[ScoredChunk]) -> list[tuple[str, int]]:
        all_terms = _loose_terms(query)
        terms = {term for term in all_terms if term not in _STOPWORDS} or set(all_terms)
        numeric_question = bool(_NUMERIC_QUESTION.search(query))

        sentences: list[tuple[int, int, str]] = []
        for marker, sc in enumerate(chunks[:5], start=1):
            for position, sentence in enumerate(self._sentences(sc.chunk.text)):
                if len(sentence.split()) >= 6:
                    sentences.append((marker, position, sentence))

        # Rare content terms carry more weight: "hours" should beat "work" for
        # a question about working hours.
        document_frequency = Counter()
        for _marker, _position, sentence in sentences:
            document_frequency.update(_loose_terms(sentence))
        idf = {term: 1.0 / math.log(2 + document_frequency[term]) for term in terms}
        total_weight = sum(idf.values()) or 1.0

        ranked: list[tuple[float, int, int, str]] = []
        for marker, position, sentence in sentences:
            trimmed = self._trim(sentence, terms)
            if len(trimmed.split()) < 5:
                continue
            matched = terms & _loose_terms(trimmed)
            overlap = sum(idf[term] for term in matched) / total_weight
            if numeric_question and re.search(r"\d", trimmed):
                overlap += 0.05
            score = overlap - 0.005 * position - 0.01 * marker
            ranked.append((score, marker, position, trimmed))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        best_score = ranked[0][0] if ranked else 0.0
        cutoff = max(0.0, best_score * 0.75)

        selected: list[tuple[str, int]] = []
        seen: set[str] = set()
        per_chunk: dict[int, int] = {}
        words = 0
        for score, marker, _position, sentence in ranked:
            if len(selected) >= self._max_sentences:
                break
            if selected and score < cutoff:
                break
            key = re.sub(r"\W+", "", sentence.lower())[:80]
            if key in seen or per_chunk.get(marker, 0) >= 2:
                continue
            if words + len(sentence.split()) > self._max_words:
                continue
            selected.append((sentence, marker))
            seen.add(key)
            per_chunk[marker] = per_chunk.get(marker, 0) + 1
            words += len(sentence.split())

        if best_score <= 0:
            selected = []

        if not selected:
            # No query terms matched (e.g. a Chinese question over English
            # sources). Quote the top passage's most concrete sentence rather
            # than refusing a question the gate just accepted.
            for sc in chunks[:1]:
                candidates = [
                    sentence
                    for sentence in self._sentences(sc.chunk.text)
                    if len(sentence.split()) >= 8
                ]
                numeric = [s for s in candidates if re.search(r"\d", s)]
                pool = numeric or candidates
                if pool:
                    selected = [(self._trim(pool[0], terms), 1)]
                break

        selected.sort(key=lambda item: item[1])
        return selected

    def generate(self, query: str, chunks: list[ScoredChunk]) -> Answer:
        if not chunks:
            return self.refuse("no_relevant_context")
        selected = self._select(query, chunks)
        if not selected:
            return self.refuse("no_extractable_sentence")
        text = " ".join(f"{sentence} [{marker}]" for sentence, marker in selected)
        return Answer(
            text=text,
            citations=_citations_from_text(text, chunks),
            refused=False,
            contexts=chunks,
            mode="extractive",
        )


class FallbackGenerator:
    """Try the primary generator, then degrade to extractive on model failure.

    The fallback answer is marked ``degraded`` so the API/UI can show it
    explicitly. A backend outage never masquerades as a relevance refusal.
    """

    def __init__(
        self,
        primary: Generator,
        fallback: ExtractiveGenerator | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback or ExtractiveGenerator()

    def refuse(self, reason: str) -> Answer:
        return self._primary.refuse(reason)

    def generate(self, query: str, chunks: list[ScoredChunk]) -> Answer:
        try:
            return self._primary.generate(query, chunks)
        except GenerationError as exc:
            answer = self._fallback.generate(query, chunks)
            return replace(
                answer,
                degraded=True,
                degraded_reason=f"{type(exc).__name__}: {exc}",
            )

"""A hand-written BM25 (Okapi) sparse retriever.

Implemented from the formula rather than pulling a library, both to keep the
dependency surface tiny and because being able to explain BM25 — term
frequency saturation (k1) and length normalisation (b) — is exactly the kind of
thing an interviewer probes. No numpy, no rank_bm25.
"""

from __future__ import annotations

import math
from collections import Counter

from ..tokenization import tokenize
from ..types import Chunk, ScoredChunk


class BM25Retriever:
    """Okapi BM25 over an in-memory corpus of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks = chunks
        self._doc_tokens = [tokenize(c.text) for c in chunks]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._term_freqs = [Counter(toks) for toks in self._doc_tokens]
        self._idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self._chunks)
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        # BM25 idf with +1 inside the log to keep it non-negative.
        return {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        tf = self._term_freqs[doc_idx]
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = self._idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1)
        return score

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        scored: list[ScoredChunk] = []
        for i, chunk in enumerate(self._chunks):
            score = self._score(query_terms, i)
            if score > 0.0:
                scored.append(
                    ScoredChunk(chunk=chunk, score=score, components={"sparse": score})
                )
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:k]

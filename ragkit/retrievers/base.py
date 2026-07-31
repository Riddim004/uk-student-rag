"""Retriever protocol shared by all retrieval strategies."""

from __future__ import annotations

from typing import Protocol

from ..types import ScoredChunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to ``k`` chunks ranked most-relevant first."""
        ...

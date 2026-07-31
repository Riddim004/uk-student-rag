"""The end-to-end RAG pipeline: retrieve then generate."""

from __future__ import annotations

from .generator import LLMGenerator
from .retrievers.base import Retriever
from .types import Answer


class RAGPipeline:
    """Wires a retriever to a generator. Retrieve top-k, then answer."""

    def __init__(self, retriever: Retriever, generator: LLMGenerator, top_k: int = 5) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def answer(self, question: str) -> Answer:
        chunks = self._retriever.retrieve(question, self._top_k)
        return self._generator.generate(question, chunks)

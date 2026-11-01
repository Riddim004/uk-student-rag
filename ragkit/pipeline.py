"""The end-to-end RAG pipeline: retrieve, gate, then generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .generator import Generator
from .relevance import Gate, RerankGate
from .retrievers.base import Retriever
from .types import Answer, ScoredChunk


@dataclass(frozen=True)
class PipelineTrace:
    """Observable intermediate state for the UI and benchmarks.

    When the gate is a ``RerankGate``, ``candidates`` are annotated with their
    rerank scores even if the answer was refused, so the refusal can be
    explained.
    """

    candidates: list[ScoredChunk] = field(default_factory=list)
    contexts: list[ScoredChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    gate_accepted: bool | None = None
    gate_best: float | None = None
    gate_threshold: float | None = None
    gate_candidate_k: int | None = None


class RAGPipeline:
    """Wires a retriever to a relevance gate and a generator."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 5,
        candidate_k: int = 50,
        gate: Gate | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._candidate_k = max(candidate_k, top_k)
        self._gate = gate

    def retrieve(self, question: str) -> list[ScoredChunk]:
        """Return the pre-gate candidate pool (for the evidence UI)."""
        return self._retriever.retrieve(question, self._candidate_k)

    def answer_with_trace(
        self,
        question: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[Answer, PipelineTrace]:
        if not question.strip():
            raise ValueError("question must not be empty")
        notify = on_stage or (lambda _stage: None)
        started = time.perf_counter()

        notify("retrieve")
        candidates = self.retrieve(question)
        retrieved_at = time.perf_counter()

        notify("gate")
        gate_accepted: bool | None = None
        gate_best: float | None = None
        gate_threshold: float | None = None
        gate_candidate_k: int | None = None
        if self._gate is None:
            contexts = candidates[: self._top_k]
        elif isinstance(self._gate, RerankGate):
            annotated = self._gate.annotate(question, candidates)
            accepted, best = self._gate.decide(annotated)
            candidates = annotated
            gate_accepted = accepted
            gate_best = None if best == float("-inf") else best
            gate_threshold = self._gate.threshold
            gate_candidate_k = self._gate.candidate_k
            contexts = annotated[: self._top_k] if accepted else []
        else:
            contexts = self._gate.filter(question, candidates)[: self._top_k]
            gate_accepted = bool(contexts)
        gated_at = time.perf_counter()

        notify("generate")
        if not contexts:
            answer = self._generator.refuse("no_relevant_context")
        else:
            answer = self._generator.generate(question, contexts)
        finished = time.perf_counter()
        notify("done")

        trace = PipelineTrace(
            candidates=candidates,
            contexts=contexts,
            timings_ms={
                "retrieve": round((retrieved_at - started) * 1000, 2),
                "gate": round((gated_at - retrieved_at) * 1000, 2),
                "generate": round((finished - gated_at) * 1000, 2),
                "total": round((finished - started) * 1000, 2),
            },
            gate_accepted=gate_accepted,
            gate_best=gate_best,
            gate_threshold=gate_threshold,
            gate_candidate_k=gate_candidate_k,
        )
        return answer, trace

    def answer(self, question: str) -> Answer:
        answer, _trace = self.answer_with_trace(question)
        return answer

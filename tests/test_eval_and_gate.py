"""Regression tests for the rebuilt eval, gate and multilingual retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.chunking import chunk_document
from ragkit.evals.dataset import EvalQuery, load_eval_set
from ragkit.evals.metrics import ndcg_at_k, precision_at_k
from ragkit.evals.runner import evaluate_retriever
from ragkit.generator import ExtractiveGenerator, LLMGenerator
from ragkit.pipeline import RAGPipeline
from ragkit.relevance import GateConfig, RelevanceGate, RerankGate
from ragkit.retrievers.rerank import NoOpReranker, RerankedRetriever
from ragkit.tokenization import tokenize
from ragkit.types import Chunk, Document, ScoredChunk

ROOT = Path(__file__).resolve().parents[1]


def _chunk(cid: str, section: str = "") -> Chunk:
    section = section or cid.split("#")[0]
    return Chunk(
        id=cid,
        doc_id=section,
        ordinal=0,
        text=f"text for {cid}",
        section_id=section,
        section_ids=(section,),
        section_path=(section, section),
    )


def _scored(cid: str, score: float, components: dict[str, float]) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid), score=score, components=components)


class _FakeRetriever:
    def __init__(self, mapping: dict[str, list[ScoredChunk]]) -> None:
        self._mapping = mapping

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self._mapping.get(query, [])[:k]


class _RecordingChat:
    def __init__(self, reply: str = "Grounded answer [1].") -> None:
        self.reply = reply
        self.called = False

    def complete(self, system: str, user: str) -> str:
        self.called = True
        return self.reply


# -- tokenisation -------------------------------------------------------


def test_chinese_tokenisation_produces_unigrams_and_bigrams():
    tokens = tokenize("学期期间 work")
    assert "学" in tokens
    assert "学期" in tokens
    assert "work" in tokens


def test_tokenisation_does_not_drop_chinese_only_queries():
    assert tokenize("毕业后能在英国停留多久？")


# -- metric definitions -------------------------------------------------


def test_precision_at_k_divides_by_k_not_by_returned_count():
    assert precision_at_k(["a"], {"a"}, k=5) == pytest.approx(0.2)


def test_ndcg_at_k_prefers_relevant_items_at_the_top():
    perfect = ndcg_at_k(["a", "b"], {"a"}, 2)
    second = ndcg_at_k(["b", "a"], {"a"}, 2)
    assert perfect == pytest.approx(1.0)
    assert second < perfect


# -- relevance gate -----------------------------------------------------


def test_gate_refuses_weak_dense_scores():
    gate = RelevanceGate(GateConfig(min_dense_score=0.80))
    weak = [_scored("a#0", 0.03, {"dense": 0.75, "rrf": 0.03})]
    strong = [_scored("a#0", 0.03, {"dense": 0.86, "rrf": 0.03})]
    assert gate.would_refuse("q", weak)
    assert not gate.would_refuse("q", strong)


def test_gate_cannot_be_fooled_by_positive_rrf_scores():
    gate = RelevanceGate(GateConfig(min_dense_score=0.80))
    # This is the original bug: RRF is always positive, so it must not be used
    # as evidence that the query is in scope.
    assert gate.would_refuse("q", [_scored("a#0", 0.03, {"rrf": 0.03})])


def test_gate_accepts_strong_lexical_match():
    gate = RelevanceGate(GateConfig(min_sparse_score=5.0))
    assert not gate.would_refuse(
        "q", [_scored("a#0", 0.02, {"sparse": 8.0, "rrf": 0.02})]
    )


def test_gate_uses_reranker_score_when_available():
    gate = RelevanceGate(GateConfig(min_rerank_score=0.10))
    weak = [_scored("a#0", 0.3, {"rerank": 0.01})]
    strong = [_scored("a#0", 0.3, {"rerank": 0.30})]
    assert gate.would_refuse("q", weak)
    assert not gate.would_refuse("q", strong)


class _FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def rerank(self, query: str, chunks: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=sc.chunk,
                score=score,
                components={**sc.components, "rerank": score},
            )
            for sc, score in zip(chunks[:k], self._scores[:k])
        ]


def test_rerank_gate_refuses_when_best_score_is_too_low():
    gate = RerankGate(_FakeReranker([-4.0, -5.0]), min_score=-3.0, candidate_k=10)
    chunks = [
        _scored("a#0", 0.03, {"dense": 0.85}),
        _scored("b#0", 0.02, {"dense": 0.84}),
    ]
    assert gate.would_refuse("q", chunks)


def test_rerank_gate_accepts_and_preserves_hybrid_order():
    gate = RerankGate(_FakeReranker([0.1, 0.9]), min_score=-3.0, candidate_k=10)
    chunks = [
        _scored("a#0", 0.03, {"dense": 0.85}),
        _scored("b#0", 0.02, {"dense": 0.84}),
    ]
    filtered = gate.filter("q", chunks)
    assert [sc.chunk.id for sc in filtered] == ["a#0", "b#0"]
    assert filtered[1].components["rerank"] == pytest.approx(0.9)


def test_rerank_gate_uses_best_score_not_top1():
    gate = RerankGate(_FakeReranker([-4.0, -2.0]), min_score=-3.0, candidate_k=10)
    chunks = [
        _scored("a#0", 0.03, {"dense": 0.85}),
        _scored("b#0", 0.02, {"dense": 0.84}),
    ]
    assert not gate.would_refuse("q", chunks)


def test_rerank_gate_annotation_exposes_decision_inputs():
    gate = RerankGate(_FakeReranker([0.1, 0.9]), min_score=-3.0, candidate_k=10)
    chunks = [
        _scored("a#0", 0.03, {"dense": 0.85}),
        _scored("b#0", 0.02, {"dense": 0.84}),
    ]
    annotated = gate.annotate("q", chunks)
    accepted, best = gate.decide(annotated)

    assert accepted is True
    assert best == pytest.approx(0.9)
    assert [sc.chunk.id for sc in annotated] == ["a#0", "b#0"]
    assert all("rerank" in sc.components for sc in annotated)


def test_pipeline_trace_records_gate_decision_on_refusal():
    gate = RerankGate(_FakeReranker([-4.0, -5.0]), min_score=-3.0, candidate_k=10)
    chunks = [
        _scored("a#0", 0.03, {"dense": 0.85, "sparse": 2.0}),
        _scored("b#0", 0.02, {"dense": 0.84}),
    ]
    pipeline = RAGPipeline(
        _FakeRetriever({"q": chunks}),
        ExtractiveGenerator(),
        top_k=2,
        candidate_k=2,
        gate=gate,
    )
    answer, trace = pipeline.answer_with_trace("q")

    assert answer.refused
    assert trace.gate_accepted is False
    assert trace.gate_threshold == -3.0
    assert trace.gate_best == pytest.approx(-4.0)
    assert trace.contexts == []
    assert [sc.components["rerank"] for sc in trace.candidates] == [-4.0, -5.0]


# -- chunking -----------------------------------------------------------


def test_chunk_document_keeps_section_context_and_metadata():
    doc = Document(
        id="student-visa__money",
        text="a b c d e f",
        title="Student visa",
        section_id="student-visa__money",
        section_path=("Student visa", "Money you need"),
        source="https://www.gov.uk/student-visa/money",
        updated_at="2026-01-01",
    )
    chunks = chunk_document(doc, chunk_size=4, overlap=0)
    assert chunks[0].text.startswith("[Document: Student visa | Section: Money you need]")
    assert chunks[0].section_id == "student-visa__money"
    assert chunks[0].updated_at == "2026-01-01"


# -- reranking ----------------------------------------------------------


def test_reranked_retriever_respects_k_and_candidate_pool():
    base = _FakeRetriever({"q": [_scored("a#0", 1.0, {}), _scored("b#0", 1.0, {})]})
    wrapped = RerankedRetriever(base, NoOpReranker(), candidate_k=10)
    assert len(wrapped.retrieve("q", k=1)) == 1


# -- eval runner --------------------------------------------------------


def test_evaluate_retriever_reports_metrics_and_error_types():
    chunks = {
        "q1": [_scored("a#0", 1.0, {"dense": 0.9}), _scored("b#0", 0.5, {"dense": 0.9})],
        "q2": [_scored("d#0", 1.0, {"dense": 0.9}), _scored("c#0", 0.5, {"dense": 0.9})],
    }
    queries = [
        EvalQuery(
            id="q1",
            question="q1",
            category="exact",
            language="en",
            answerable=True,
            reference_answer="a",
            split="test",
            relevant_chunk_ids=("a#0",),
            relevant_section_ids=("a",),
        ),
        EvalQuery(
            id="q2",
            question="q2",
            category="paraphrase",
            language="en",
            answerable=True,
            reference_answer="c",
            split="test",
            relevant_chunk_ids=("c#0",),
            relevant_section_ids=("c",),
        ),
        EvalQuery(
            id="q3",
            question="q3",
            category="unanswerable",
            language="en",
            answerable=False,
            reference_answer="refuse",
            split="test",
        ),
    ]
    report = evaluate_retriever("fake", _FakeRetriever(chunks), queries, k=2)
    assert report.metrics["recall"] == pytest.approx(1.0)
    assert report.metrics["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert report.metrics["ndcg"] == pytest.approx(
        (1.0 + ndcg_at_k(["d#0", "c#0"], {"c#0"}, 2)) / 2
    )
    assert report.error_counts["hit"] == 1
    assert report.error_counts["ranking_error"] == 1
    assert report.error_counts["unanswerable_skipped"] == 1


def test_real_eval_set_is_valid_and_versioned():
    queries, digest = load_eval_set(ROOT / "evidence" / "eval")
    assert len(queries) == 50
    assert sum(1 for q in queries if q.answerable) == 42
    assert sum(1 for q in queries if q.split == "dev") == 25
    assert digest


# -- end-to-end gate regression ----------------------------------------


def test_pipeline_refuses_when_gate_rejects_retrieval_miss():
    chat = _RecordingChat()
    retriever = _FakeRetriever(
        {"out of domain": [_scored("a#0", 0.03, {"dense": 0.70, "rrf": 0.03})]}
    )
    pipeline = RAGPipeline(
        retriever,
        LLMGenerator(chat),
        top_k=5,
        gate=RelevanceGate(GateConfig(min_dense_score=0.80)),
    )
    answer = pipeline.answer("out of domain")
    assert answer.refused is True
    assert answer.refused_reason == "no_relevant_context"
    assert chat.called is False


def test_pipeline_answers_when_gate_accepts_context():
    chat = _RecordingChat("Answer [1].")
    retriever = _FakeRetriever(
        {"in domain": [_scored("a#0", 0.03, {"dense": 0.86, "rrf": 0.03})]}
    )
    pipeline = RAGPipeline(
        retriever,
        LLMGenerator(chat),
        top_k=5,
        gate=RelevanceGate(GateConfig(min_dense_score=0.80)),
    )
    answer = pipeline.answer("in domain")
    assert answer.refused is False
    assert chat.called is True
    assert answer.citations[0].chunk_id == "a#0"


def test_citations_carry_section_path_and_snippet():
    chat = _RecordingChat("Answer [1].")
    chunk = Chunk(
        id="a#0",
        doc_id="a",
        ordinal=0,
        text="The answer is 20 hours.",
        title="Student visa",
        source="https://www.gov.uk/student-visa",
        section_id="a",
        section_ids=("a",),
        section_path=("Student visa", "Work"),
        updated_at="2026-01-01",
    )
    gen = LLMGenerator(chat)
    answer = gen.generate("q", [ScoredChunk(chunk=chunk, score=1.0, components={"dense": 0.9})])
    citation = answer.citations[0]
    assert citation.section_path == ("Student visa", "Work")
    assert citation.updated_at == "2026-01-01"
    assert "20 hours" in citation.snippet

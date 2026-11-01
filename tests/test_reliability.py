"""Failure-injection and data-integrity tests.

These cover the reliability claims in ``evidence/reliability/``: retries are
bounded, non-transient errors are not retried, model outages are not disguised
as refusals, bad corpus data fails loudly, and the bundled corpus matches its
manifest hashes.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from ragkit.config import Settings
from ragkit.embeddings import HashingEmbedder
from ragkit.errors import (
    ConfigurationError,
    DataError,
    EmbeddingError,
    GenerationError,
    TransientModelError,
    is_transient,
)
from ragkit.generator import (
    ExtractiveGenerator,
    FallbackGenerator,
    LLMGenerator,
    build_context,
)
from ragkit.ingest import build_corpus, load_documents_jsonl
from ragkit.pipeline import RAGPipeline
from ragkit.retry import retry_call
from ragkit.types import Chunk, Document, ScoredChunk

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


def test_retry_succeeds_after_transient_failures() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientModelError("temporary")
        return "ok"

    result = retry_call(
        flaky,
        attempts=3,
        base_delay=1.0,
        sleep=sleeps.append,
        rng=random.Random(0),
        retry_if=lambda exc: isinstance(exc, TransientModelError),
    )

    assert result == "ok"
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert 0.9 <= sleeps[0] <= 1.1
    assert 1.8 <= sleeps[1] <= 2.2


def test_retry_does_not_retry_non_transient_error() -> None:
    calls: list[int] = []

    def reject() -> str:
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        retry_call(
            reject,
            attempts=3,
            sleep=lambda _s: None,
            retry_if=lambda exc: isinstance(exc, TransientModelError),
        )
    assert len(calls) == 1


def test_retry_reraises_last_error_after_budget() -> None:
    calls: list[int] = []

    def always() -> str:
        calls.append(1)
        raise TransientModelError(f"failure {len(calls)}")

    with pytest.raises(TransientModelError, match="failure 3"):
        retry_call(
            always,
            attempts=3,
            base_delay=0.0,
            sleep=lambda _s: None,
            retry_if=lambda exc: isinstance(exc, TransientModelError),
        )
    assert len(calls) == 3


def test_transient_classification() -> None:
    class RateLimited(Exception):
        status_code = 429

    class InternalServerError(Exception):
        pass

    class BadRequest(Exception):
        status_code = 400

    assert is_transient(RateLimited())
    assert is_transient(InternalServerError())
    assert is_transient(TimeoutError())
    assert not is_transient(BadRequest())
    assert not is_transient(ValueError())


# --------------------------------------------------------------------------
# Generation failures and fallback
# --------------------------------------------------------------------------


def _scored() -> ScoredChunk:
    chunk = Chunk(
        id="working#0",
        doc_id="working",
        ordinal=0,
        text=(
            "Students on a full-time degree course may work up to 20 hours per week "
            "during term time. Term-time hours are a strict weekly limit."
        ),
        title="Working during your studies",
        source="https://example.test/working",
        section_id="working",
        section_path=("Working during your studies",),
        updated_at="2026-01-01",
    )
    return ScoredChunk(chunk=chunk, score=1.0, components={"dense": 0.9})


class FailingChat:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or GenerationError("backend down")

    def complete(self, system: str, user: str) -> str:
        raise self.error


class EmptyChat:
    def complete(self, system: str, user: str) -> str:
        return "   "


def test_generation_error_is_not_a_refusal() -> None:
    generator = LLMGenerator(FailingChat(), Settings())
    with pytest.raises(GenerationError):
        generator.generate("How many hours can I work?", [_scored()])


def test_empty_model_response_is_an_explicit_refusal() -> None:
    answer = LLMGenerator(EmptyChat(), Settings()).generate("hours?", [_scored()])
    assert answer.refused
    assert answer.refused_reason == "empty_model_response"


def test_fallback_degrades_to_grounded_extractive_answer() -> None:
    generator = FallbackGenerator(LLMGenerator(FailingChat(), Settings()))
    answer = generator.generate("How many hours can I work?", [_scored()])

    assert not answer.refused
    assert answer.degraded
    assert answer.degraded_reason.startswith("GenerationError")
    assert answer.mode == "extractive"
    assert "20 hours per week" in answer.text
    assert answer.citations and answer.citations[0].marker == 1


def test_fallback_keeps_primary_answer_when_it_succeeds() -> None:
    class WorkingChat:
        def complete(self, system: str, user: str) -> str:
            return "You may work 20 hours per week [1]."

    generator = FallbackGenerator(LLMGenerator(WorkingChat(), Settings()))
    answer = generator.generate("hours?", [_scored()])
    assert not answer.degraded
    assert answer.mode == "llm"


def test_extractive_answer_is_always_cited() -> None:
    answer = ExtractiveGenerator().generate("20 hours", [_scored()])
    assert not answer.refused
    assert "[1]" in answer.text
    assert answer.citations[0].chunk_id == "working#0"


def test_extractive_trims_a_long_table_shaped_sentence() -> None:
    long = (
        "[Document: Rules | Section: Work]\n"
        "Student following a full-time course of degree level or above study: "
        + "sponsored by a provider with a track record of compliance; " * 20
        + "20 hours per week during term-time. "
        + "Other conditions apply for placements and sabbatical officers."
    )
    chunk = Chunk(id="rules#0", doc_id="rules", ordinal=0, text=long, title="Rules")
    answer = ExtractiveGenerator().generate(
        "How many hours can I work per week during term time?",
        [ScoredChunk(chunk=chunk, score=1.0, components={"dense": 0.9})],
    )

    assert not answer.refused
    assert "20 hours per week" in answer.text
    assert len(answer.text.split()) <= 80
    assert answer.citations[0].marker == 1


def test_extractive_prefers_the_rule_over_a_nearby_numeric_sentence() -> None:
    rule = Chunk(
        id="rules#0",
        doc_id="rules",
        ordinal=0,
        text=(
            "[Document: Rules | Section: Work]\n"
            "The applicant may work 20 hours per week during term-time. "
            "Other employment is restricted."
        ),
    )
    distractor = Chunk(
        id="course#0",
        doc_id="course",
        ordinal=0,
        text="You can take a full-time course with at least 15 hours a week.",
    )
    answer = ExtractiveGenerator().generate(
        "How many hours per week during term time?",
        [
            ScoredChunk(chunk=rule, score=1.0, components={"dense": 0.9}),
            ScoredChunk(chunk=distractor, score=0.9, components={"dense": 0.8}),
        ],
    )
    assert "20 hours per week" in answer.text
    assert "15 hours" not in answer.text
    assert answer.citations[0].chunk_id == "rules#0"


def test_extractive_falls_back_to_a_concrete_sentence_for_chinese() -> None:
    chunk = Chunk(
        id="rules#0",
        doc_id="rules",
        ordinal=0,
        text=(
            "[Document: Rules | Section: Work]\n"
            "The applicant may work 20 hours per week during term-time. "
            "Other employment is restricted."
        ),
    )
    answer = ExtractiveGenerator().generate(
        "学期期间每周最多能工作多少小时？",
        [ScoredChunk(chunk=chunk, score=1.0, components={"dense": 0.9})],
    )
    assert not answer.refused
    assert "20 hours per week" in answer.text
    assert answer.citations[0].chunk_id == "rules#0"


def test_extractive_centres_a_long_no_overlap_quote_on_a_unit_value() -> None:
    long = (
        "[Document: Rules | Section: Work]\n"
        "Student following a full-time course of degree level or above study "
        + "sponsored by a provider with a track record of compliance; " * 12
        + "20 hours per week during term-time. Other conditions apply."
    )
    chunk = Chunk(id="rules#0", doc_id="rules", ordinal=0, text=long, title="Rules")
    answer = ExtractiveGenerator().generate(
        "学期期间每周最多能工作多少小时？",
        [ScoredChunk(chunk=chunk, score=1.0, components={"dense": 0.9})],
    )
    assert not answer.refused
    assert "20 hours per week" in answer.text
    assert len(answer.text.split()) <= 80


def test_build_context_does_not_repeat_the_retrieval_header() -> None:
    chunk = Chunk(
        id="a#0",
        doc_id="a",
        ordinal=0,
        text="[Document: Guide | Section: Money]\nThe fee is £558.",
        title="Guide",
        section_path=("Money",),
    )
    context = build_context([ScoredChunk(chunk=chunk, score=1.0)])
    assert "[Document:" not in context
    assert "The fee is £558." in context
    assert "Guide — Money" in context


def test_empty_question_is_rejected_before_retrieval(corpus) -> None:
    pipeline = RAGPipeline(
        corpus.hybrid,
        LLMGenerator(FailingChat(), Settings()),
        top_k=3,
        candidate_k=5,
    )
    with pytest.raises(ValueError, match="question must not be empty"):
        pipeline.answer_with_trace("   ")


# --------------------------------------------------------------------------
# Corpus validation
# --------------------------------------------------------------------------


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "documents.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_corpus_file_fails_with_rebuild_hint(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="corpus not found"):
        load_documents_jsonl(tmp_path / "missing.jsonl")


def test_empty_corpus_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="corpus is empty"):
        load_documents_jsonl(_write(tmp_path, "\n\n"))


def test_malformed_json_reports_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "a", "text": "ok"}\n{not json}\n')
    with pytest.raises(DataError, match=":2: invalid JSON"):
        load_documents_jsonl(path)


def test_document_without_text_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "a"}\n')
    with pytest.raises(DataError, match="missing required field"):
        load_documents_jsonl(path)


def test_duplicate_document_ids_are_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '{"id": "a", "text": "one"}\n{"id": "a", "text": "two"}\n',
    )
    with pytest.raises(DataError, match="duplicate document id"):
        load_documents_jsonl(path)


def test_zero_documents_cannot_build_a_corpus() -> None:
    with pytest.raises(DataError, match="zero documents"):
        build_corpus([], HashingEmbedder(dim=8), Settings())


def test_invalid_chunk_settings_are_configuration_errors() -> None:
    docs = [Document(id="a", text="hello world")]
    with pytest.raises(ConfigurationError, match="chunk_size"):
        build_corpus(docs, HashingEmbedder(dim=8), Settings(chunk_size=0))
    with pytest.raises(ConfigurationError, match="chunk_overlap"):
        build_corpus(
            docs,
            HashingEmbedder(dim=8),
            Settings(chunk_size=10, chunk_overlap=10),
        )


class EmptyEmbedder:
    dim = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def test_embedder_output_count_mismatch_is_an_embedding_error() -> None:
    docs = [Document(id="a", text="one two three four five six seven")]
    with pytest.raises(EmbeddingError, match="vectors"):
        build_corpus(
            docs,
            EmptyEmbedder(),
            Settings(chunk_size=10, chunk_overlap=2),
        )


# --------------------------------------------------------------------------
# Bundled artifact integrity
# --------------------------------------------------------------------------


def test_bundled_corpus_matches_manifest_hashes() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))

    documents_bytes = (CORPUS / "documents.jsonl").read_bytes()
    chunks_bytes = (CORPUS / "chunks.jsonl").read_bytes()

    assert hashlib.sha256(documents_bytes).hexdigest() == manifest["corpus_hash"]
    assert hashlib.sha256(chunks_bytes).hexdigest() == manifest["chunk_hash"]

    assert manifest["total_documents"] == len(
        load_documents_jsonl(CORPUS / "documents.jsonl")
    )
    assert manifest["total_chunks"] == chunks_bytes.decode("utf-8").count("\n")

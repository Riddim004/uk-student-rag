"""Tests for generation (citations, refusal) and the end-to-end pipeline."""

from __future__ import annotations

from conftest import FakeChatModel
from ragkit.config import Settings
from ragkit.generator import REFUSAL_TEXT, LLMGenerator, build_context
from ragkit.pipeline import RAGPipeline
from ragkit.types import Chunk, ScoredChunk


def _scored(cid: str, text: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(id=cid, doc_id=cid.split("#")[0], ordinal=0, text=text, title=cid)
    return ScoredChunk(chunk=chunk, score=score)


# -- context rendering --------------------------------------------------

def test_build_context_numbers_passages():
    ctx = build_context([_scored("a#0", "alpha"), _scored("b#0", "beta")])
    assert "[1] a#0\nalpha" in ctx
    assert "[2] b#0\nbeta" in ctx


# -- refusal ------------------------------------------------------------

def test_generator_refuses_without_context_and_skips_llm():
    chat = FakeChatModel(reply="should never be used")
    gen = LLMGenerator(chat, Settings())
    answer = gen.generate("anything", [])
    assert answer.refused is True
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []
    assert chat.last_user is None  # LLM was not called


def test_generator_marks_refusal_when_model_refuses():
    chat = FakeChatModel(reply=REFUSAL_TEXT)
    gen = LLMGenerator(chat, Settings())
    answer = gen.generate("q", [_scored("a#0", "irrelevant text")])
    assert answer.refused is True
    assert answer.citations == []


# -- grounded answers + citations ---------------------------------------

def test_generator_extracts_citations_in_range():
    chat = FakeChatModel(reply="You can work 20 hours [1]. See also [2].")
    gen = LLMGenerator(chat, Settings())
    answer = gen.generate("q", [_scored("working#0", "..."), _scored("finance#0", "...")])
    assert answer.refused is False
    assert [c.marker for c in answer.citations] == [1, 2]
    assert answer.citations[0].doc_id == "working"


def test_generator_drops_out_of_range_citations():
    chat = FakeChatModel(reply="Fabricated reference [5].")
    gen = LLMGenerator(chat, Settings())
    answer = gen.generate("q", [_scored("a#0", "only one chunk")])
    assert answer.refused is False
    assert answer.citations == []  # [5] has no matching chunk


def test_generator_passes_context_to_model():
    chat = FakeChatModel(reply="ok [1]")
    gen = LLMGenerator(chat, Settings())
    gen.generate("my question", [_scored("a#0", "SECRET_CONTEXT_TOKEN")])
    assert "SECRET_CONTEXT_TOKEN" in chat.last_user
    assert "my question" in chat.last_user


# -- end-to-end pipeline ------------------------------------------------

def test_pipeline_answers_from_corpus(corpus):
    chat = FakeChatModel(reply="You may work up to 20 hours per week [1].")
    gen = LLMGenerator(chat, Settings())
    pipeline = RAGPipeline(corpus.bm25, gen, top_k=5)

    answer = pipeline.answer("how many hours can I work per week")
    assert answer.refused is False
    assert answer.contexts, "expected retrieved contexts"
    assert answer.citations and answer.citations[0].marker == 1


def test_pipeline_refuses_when_retrieval_is_empty(corpus):
    chat = FakeChatModel(reply="should not be called")
    gen = LLMGenerator(chat, Settings())
    pipeline = RAGPipeline(corpus.bm25, gen, top_k=5)

    answer = pipeline.answer("xylophone quantum banana")  # nothing matches
    assert answer.refused is True
    assert chat.last_user is None

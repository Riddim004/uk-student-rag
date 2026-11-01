"""API tests: serialization, refusal payloads, SSE stages, and static mount."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import DOCS
from ragkit.api import create_app
from ragkit.config import Settings
from ragkit.embeddings import HashingEmbedder
from ragkit.generator import ExtractiveGenerator
from ragkit.ingest import build_corpus
from ragkit.pipeline import RAGPipeline
from ragkit.relevance import GateConfig, RelevanceGate
from ragkit.runtime import Runtime


def _runtime(gate: RelevanceGate) -> Runtime:
    settings = Settings(
        chunk_size=30,
        chunk_overlap=5,
        top_k=3,
        candidate_k=6,
        min_dense_score=0.0,
        min_sparse_score=0.0,
    )
    corpus = build_corpus(DOCS, HashingEmbedder(dim=128), settings)
    pipeline = RAGPipeline(
        corpus.hybrid,
        ExtractiveGenerator(),
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        gate=gate,
    )
    return Runtime(
        settings=settings,
        corpus=corpus,
        pipeline=pipeline,
        corpus_manifest={
            "corpus_hash": "a" * 64,
            "chunk_hash": "b" * 64,
            "total_chunks": len(corpus.chunks),
            "total_documents": len(corpus.documents),
            "sources": [],
            "generated_at": "2026-11-01T00:00:00+00:00",
        },
        eval_summary={
            "dataset": {"n_queries": 50},
            "hybrid": {"recall": 0.972, "ndcg": 0.82},
            "gate": {"gate_accuracy": 0.92},
            "metadata": {},
        },
        examples=[
            {
                "id": "q01",
                "question": "How many hours can I work?",
                "category": "exact",
                "language": "en",
            }
        ],
        mode="extractive",
        gate_mode="scalar",
        startup_ms=12.3,
    )


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(_runtime(RelevanceGate(GateConfig())), web_dir=tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_corpus_and_mode(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["product_mode"] == "extractive"
    assert body["corpus"]["total_chunks"] == 8
    assert body["eval"]["hybrid"]["recall"] == 0.972


def test_ask_returns_cited_extractive_answer(client: TestClient) -> None:
    response = client.post(
        "/api/ask",
        json={"question": "How many hours can I work during term time?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert not body["refused"]
    assert body["answer_mode"] == "extractive"
    assert body["citations"]
    assert body["citations"][0]["marker"] == 1
    assert body["contexts"]
    assert body["timings_ms"]["total"] >= 0
    assert body["gate"]["accepted"] is True


def test_refusal_keeps_the_closest_passages(tmp_path: Path) -> None:
    gate = RelevanceGate(GateConfig(min_dense_score=2.0, min_sparse_score=10**9))
    app = create_app(_runtime(gate), web_dir=tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/ask", json={"question": "Can I bring my dog?"})
    body = response.json()
    assert response.status_code == 200
    assert body["refused"] is True
    assert body["refused_reason"] == "no_relevant_context"
    assert body["contexts"] == []
    assert body["trail"]
    assert body["gate"]["accepted"] is False


def test_sse_stream_emits_real_stages_then_answer(client: TestClient) -> None:
    response = client.get(
        "/api/ask/stream",
        params={"q": "How many hours can I work during term time?"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    for stage in ("retrieve", "gate", "generate"):
        assert f'data: {{"stage": "{stage}"}}' in text
    assert "event: answer" in text
    assert "event: error" not in text


def test_empty_question_is_rejected(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": " "})
    assert response.status_code == 422


def test_examples_endpoint(client: TestClient) -> None:
    response = client.get("/api/examples")
    assert response.status_code == 200
    assert response.json()["examples"][0]["id"] == "q01"

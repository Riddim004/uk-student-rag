"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """All tunable knobs, with reproducible defaults.

    Read from environment variables (see ``.env.example``) but also
    constructible directly, which keeps tests hermetic.
    """

    # LLM / embeddings
    api_base: str | None = None
    embedding_backend: str = "local"  # local | openai | hashing
    local_embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_output_tokens: int = 400
    embedding_batch_size: int = 32
    device: str = "cpu"

    # Network reliability: bounded timeouts + explicit retries.
    request_timeout_s: float = 30.0
    max_retries: int = 2
    retry_base_delay_s: float = 0.5

    # Product resilience: if the LLM is unavailable, serve a grounded
    # extractive answer and mark it as degraded instead of failing closed.
    fallback_to_extractive: bool = True

    # Chunking (word-based, section-aware)
    chunk_size: int = 260
    chunk_overlap: int = 40

    # Retrieval
    top_k: int = 5
    candidate_k: int = 50
    rerank_candidate_k: int = 15
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # Relevance gate. The scalar dense/sparse thresholds are kept as a
    # documented negative result; the shipped gate uses the reranker score.
    # See evidence/eval/gate_calibration.md.
    min_dense_score: float = 0.30
    min_sparse_score: float = 0.0
    min_rerank_score: float = -3.0
    refuse_when_empty: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_base=os.getenv("OPENAI_API_BASE") or None,
            embedding_backend=os.getenv("EMBEDDING_BACKEND", cls.embedding_backend),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", cls.local_embedding_model
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chat_model=os.getenv("CHAT_MODEL", cls.chat_model),
            temperature=float(os.getenv("LLM_TEMPERATURE", cls.temperature)),
            max_output_tokens=int(
                os.getenv("MAX_OUTPUT_TOKENS", cls.max_output_tokens)
            ),
            embedding_batch_size=int(
                os.getenv("EMBEDDING_BATCH_SIZE", cls.embedding_batch_size)
            ),
            device=os.getenv("EMBEDDING_DEVICE", cls.device),
            request_timeout_s=float(
                os.getenv("REQUEST_TIMEOUT_S", cls.request_timeout_s)
            ),
            max_retries=int(os.getenv("MAX_RETRIES", cls.max_retries)),
            retry_base_delay_s=float(
                os.getenv("RETRY_BASE_DELAY_S", cls.retry_base_delay_s)
            ),
            fallback_to_extractive=_as_bool(
                os.getenv("FALLBACK_TO_EXTRACTIVE"), cls.fallback_to_extractive
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(os.getenv("TOP_K", cls.top_k)),
            candidate_k=int(os.getenv("CANDIDATE_K", cls.candidate_k)),
            rerank_candidate_k=int(
                os.getenv("RERANK_CANDIDATE_K", cls.rerank_candidate_k)
            ),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            rrf_k=int(os.getenv("RRF_K", cls.rrf_k)),
            dense_weight=float(os.getenv("DENSE_WEIGHT", cls.dense_weight)),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", cls.sparse_weight)),
            min_dense_score=float(
                os.getenv("MIN_DENSE_SCORE", cls.min_dense_score)
            ),
            min_sparse_score=float(
                os.getenv("MIN_SPARSE_SCORE", cls.min_sparse_score)
            ),
            min_rerank_score=float(
                os.getenv("MIN_RERANK_SCORE", cls.min_rerank_score)
            ),
            refuse_when_empty=_as_bool(
                os.getenv("REFUSE_WHEN_EMPTY"), cls.refuse_when_empty
            ),
        )

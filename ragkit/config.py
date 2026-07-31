"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """All tunable knobs, with sensible defaults.

    Read from environment variables (see ``.env.example``) but also
    constructible directly, which keeps tests hermetic.
    """

    # LLM / embeddings
    api_base: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.1

    # Chunking (word-based)
    chunk_size: int = 180
    chunk_overlap: int = 40

    # Retrieval
    top_k: int = 5
    candidate_k: int = 20  # per-retriever pool before fusion
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # Generation
    min_context_score: float = 0.0  # below this, treat context as empty
    refuse_when_empty: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_base=os.getenv("OPENAI_API_BASE") or None,
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chat_model=os.getenv("CHAT_MODEL", cls.chat_model),
            temperature=float(os.getenv("LLM_TEMPERATURE", cls.temperature)),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(os.getenv("TOP_K", cls.top_k)),
            candidate_k=int(os.getenv("CANDIDATE_K", cls.candidate_k)),
            rrf_k=int(os.getenv("RRF_K", cls.rrf_k)),
        )

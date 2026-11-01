"""Embedding backends.

The pipeline only needs two operations: embed documents for indexing and embed a
query for search. Keeping them separate matters for E5-style models, which
require different prefixes; it also makes the retrieval interface explicit.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from .config import Settings
from .errors import (
    ConfigurationError,
    EmbeddingError,
    TransientModelError,
    is_transient,
)
from .retry import retry_call
from .tokenization import tokenize


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalised vector per document text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return an L2-normalised query vector."""
        ...


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class HashingEmbedder:
    """Hashing-trick bag-of-words embedder — deterministic and offline.

    It is intentionally *not* semantic. It exists for hermetic unit tests and
    for CI paths where downloading a model is undesirable, not for quality
    claims.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalise(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAIEmbedder:
    """Real embeddings via an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_base: str | None = None,
        batch_size: int = 100,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_base_delay_s: float = 0.5,
    ) -> None:
        from openai import OpenAI  # imported lazily so tests need no dependency

        kwargs = {"timeout": timeout_s, "max_retries": 0}
        if api_base:
            kwargs["base_url"] = api_base
        self._client = OpenAI(**kwargs)
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_base_delay_s = retry_base_delay_s
        self.dim = 1536

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(model=self._model, input=batch)
        except Exception as exc:
            if is_transient(exc):
                raise TransientModelError(
                    f"embedding request failed: {type(exc).__name__}"
                ) from exc
            raise EmbeddingError(
                f"embedding request rejected: {type(exc).__name__}: {exc}"
            ) from exc
        if len(resp.data) != len(batch):
            raise EmbeddingError(
                f"embedding response has {len(resp.data)} vectors for {len(batch)} inputs"
            )
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [_l2_normalise(list(d.embedding)) for d in ordered]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                batch_vectors = retry_call(
                    lambda: self._embed_batch(batch),
                    attempts=self._max_retries + 1,
                    base_delay=self._retry_base_delay_s,
                    retry_if=lambda exc: isinstance(exc, TransientModelError),
                )
            except TransientModelError as exc:
                raise EmbeddingError(
                    f"embedding backend unavailable after "
                    f"{self._max_retries + 1} attempts: {exc}"
                ) from exc
            vectors.extend(batch_vectors)
        if vectors:
            dim = len(vectors[0])
            if any(len(vector) != dim for vector in vectors):
                raise EmbeddingError("embedding backend returned inconsistent dimensions")
            self.dim = dim
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    """Local multilingual embeddings via sentence-transformers.

    Defaults to ``intfloat/multilingual-e5-small``: 384 dimensions, CPU-friendly,
    and strong enough for English + Chinese queries. E5 models expect
    ``query:`` / ``passage:`` prefixes; they are applied automatically.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = query_prefix if query_prefix is not None else ("query: " if is_e5 else "")
        self._passage_prefix = (
            passage_prefix if passage_prefix is not None else ("passage: " if is_e5 else "")
        )
        self._model = None
        self._dim = 0

    @property
    def dim(self) -> int:
        self._ensure_model()
        return self._dim

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise EmbeddingError(
                "sentence-transformers is not installed. Run "
                "`pip install -r requirements-local.txt` or set EMBEDDING_BACKEND=openai."
            ) from exc
        try:
            model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:  # pragma: no cover - model download/runtime
            raise EmbeddingError(
                f"could not load local embedding model {self._model_name!r}: {exc}"
            ) from exc
        dim_getter = getattr(model, "get_embedding_dimension", None)
        self._dim = int(
            dim_getter() if dim_getter else model.get_sentence_embedding_dimension()
        )
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        assert self._model is not None
        prefixed = [self._passage_prefix + text for text in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"local embedder returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        assert self._model is not None
        vectors = self._model.encode(
            [self._query_prefix + text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if len(vectors) != 1:
            raise EmbeddingError("local embedder returned no query vector")
        return list(map(float, vectors[0]))


def make_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or Settings.from_env()
    backend = settings.embedding_backend.lower()
    if backend == "local":
        return SentenceTransformerEmbedder(
            model_name=settings.local_embedding_model,
            device=settings.device,
            batch_size=settings.embedding_batch_size,
        )
    if backend == "openai":
        return OpenAIEmbedder(
            model=settings.embedding_model,
            api_base=settings.api_base,
            batch_size=settings.embedding_batch_size,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.max_retries,
            retry_base_delay_s=settings.retry_base_delay_s,
        )
    if backend == "hashing":
        return HashingEmbedder()
    raise ConfigurationError(f"unknown embedding backend: {settings.embedding_backend}")

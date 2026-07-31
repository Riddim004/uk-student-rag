"""Embedding backends.

``Embedder`` is a tiny protocol. Two implementations ship:

* ``OpenAIEmbedder`` — real embeddings via an OpenAI-compatible API.
* ``HashingEmbedder`` — a deterministic, dependency-free, offline embedder.
  It is not semantically strong, but it is *stable and free*, which is exactly
  what the test suite and CI need: dense retrieval and the full pipeline can run
  with no API key and no network.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalised vector per input text."""
        ...


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class HashingEmbedder:
    """Hashing-trick bag-of-words embedder — deterministic and offline.

    Each token is hashed into a fixed-width vector (with a sign bit to reduce
    collisions cancelling out), then the vector is L2-normalised so cosine
    similarity reduces to a dot product.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalise(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Real embeddings via an OpenAI-compatible endpoint."""

    def __init__(self, model: str = "text-embedding-3-small", api_base: str | None = None) -> None:
        from openai import OpenAI  # imported lazily so tests need no dependency

        self._client = OpenAI(base_url=api_base) if api_base else OpenAI()
        self._model = model
        # text-embedding-3-small is 1536-dim; kept as an attribute for callers.
        self.dim = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [_l2_normalise(d.embedding) for d in resp.data]

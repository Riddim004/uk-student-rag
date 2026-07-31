"""Retrieval strategies: sparse (BM25), dense, and hybrid (RRF fusion)."""

from .base import Retriever
from .bm25 import BM25Retriever, tokenize
from .dense import DenseRetriever
from .hybrid import HybridRetriever

__all__ = [
    "Retriever",
    "BM25Retriever",
    "tokenize",
    "DenseRetriever",
    "HybridRetriever",
]

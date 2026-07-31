"""Retrieval-quality evaluation.

Most RAG demos only eyeball the final answer. This package evaluates the
*retriever* itself with standard IR metrics against a labelled query set, which
is what lets you say "hybrid beats dense by X on recall@5" with evidence.

Deterministic and offline: metrics are pure functions, and the runner takes any
retriever, so with the ``HashingEmbedder`` the whole thing runs in CI.
"""

from .dataset import DEFAULT_EVAL_SET, EvalQuery
from .metrics import average_precision, hit_rate, mrr, precision_at_k, recall_at_k
from .runner import RetrievalReport, evaluate_retriever

__all__ = [
    "DEFAULT_EVAL_SET",
    "EvalQuery",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "hit_rate",
    "average_precision",
    "RetrievalReport",
    "evaluate_retriever",
]

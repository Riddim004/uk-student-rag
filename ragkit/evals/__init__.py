"""Retrieval-quality evaluation.

Deterministic and offline by default: metrics are pure functions, and the
runner takes any retriever, so the dataset, metrics and failure taxonomy can be
re-run from a pinned corpus snapshot.
"""

from .dataset import EvalQuery, load_eval_set
from .metrics import (
    average_precision,
    bootstrap_mean_ci,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .runner import QueryResult, RetrievalReport, compare_to_markdown, evaluate_retriever

__all__ = [
    "EvalQuery",
    "load_eval_set",
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "mrr",
    "hit_rate",
    "average_precision",
    "bootstrap_mean_ci",
    "RetrievalReport",
    "QueryResult",
    "evaluate_retriever",
    "compare_to_markdown",
]

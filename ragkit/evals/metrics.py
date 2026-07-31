"""Standard information-retrieval metrics.

All functions take a ranked list of retrieved ids (best first) and the set of
relevant ids, and return a score in ``[0, 1]``. Pure and dependency-free so
they are trivial to unit-test with hand-checked expected values.

Relevance here is graded at the *document* level: a retrieved chunk id like
``visa#3`` counts as hitting document ``visa``. This matches how the eval set is
labelled (by document), and is handled by the runner before calling these.
"""

from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for item in topk if item in relevant)
    return hits / len(topk)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant item (0 if none retrieved)."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant item is in the top-k, else 0.0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average precision — rewards ranking relevant items higher."""
    if not relevant:
        return 0.0
    hits = 0
    summed = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            summed += hits / rank
    return summed / len(relevant)

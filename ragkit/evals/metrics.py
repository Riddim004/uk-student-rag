"""Standard information-retrieval metrics + bootstrap confidence intervals.

All functions take a ranked list of retrieved ids (best first) and the set of
relevant ids. ``precision_at_k`` divides by ``k`` (not by the number of items
actually returned), which is the standard definition and avoids inflating a
retriever that returns very few results.
"""

from __future__ import annotations

import math
import random
from statistics import mean


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    summed = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            summed += hits / rank
    return summed / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for the mean of ``values``.

    Deterministic for a fixed seed; resamples queries, not documents.
    """
    if not values:
        return 0.0, 0.0, 0.0
    point = mean(values)
    if len(values) == 1:
        return point, point, point
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    low = samples[max(0, int((alpha / 2) * n_boot) - 1)]
    high = samples[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, low, high

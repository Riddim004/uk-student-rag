"""Evaluate retrievers on the labelled query set and render evidence reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean

from ..relevance import Gate
from ..retrievers.base import Retriever
from ..types import ScoredChunk
from .dataset import EvalQuery
from .metrics import (
    average_precision,
    bootstrap_mean_ci,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

METRIC_NAMES = (
    "recall",
    "precision",
    "ndcg",
    "mrr",
    "hit",
    "map",
    "section_recall",
)

ERROR_TYPES = (
    "hit",
    "partial_hit",
    "ranking_error",
    "retrieval_miss",
    "over_refusal",
    "correct_refusal",
    "unanswerable_answered",
    "unanswerable_skipped",
)


@dataclass
class QueryResult:
    query_id: str
    question: str
    category: str
    language: str
    answerable: bool
    retrieved_chunk_ids: list[str]
    retrieved_section_ids: list[str]
    metrics: dict[str, float]
    error_type: str
    gate_refused: bool | None = None


@dataclass
class RetrievalReport:
    name: str
    k: int
    n_queries: int
    n_answerable: int
    n_unanswerable: int
    metrics: dict[str, float]
    ci: dict[str, tuple[float, float]]
    per_category: dict[str, dict[str, float]]
    error_counts: dict[str, int]
    per_query: list[QueryResult]
    refusal: dict[str, float] | None = None
    extra: dict[str, float] = field(default_factory=dict)


def _section_ids(sc: ScoredChunk) -> list[str]:
    ids = sc.chunk.section_ids or (sc.chunk.section_id,)
    return [item for item in ids if item]


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _retrieval_metrics(
    retrieved_chunks: list[str],
    relevant_chunks: set[str],
    retrieved_sections: list[str],
    relevant_sections: set[str],
    k: int,
) -> dict[str, float]:
    return {
        "recall": recall_at_k(retrieved_chunks, relevant_chunks, k),
        "precision": precision_at_k(retrieved_chunks, relevant_chunks, k),
        "ndcg": ndcg_at_k(retrieved_chunks, relevant_chunks, k),
        "mrr": mrr(retrieved_chunks, relevant_chunks),
        "hit": hit_rate(retrieved_chunks, relevant_chunks, k),
        "map": average_precision(retrieved_chunks, relevant_chunks),
        "section_recall": recall_at_k(retrieved_sections, relevant_sections, k),
    }


def evaluate_retriever(
    name: str,
    retriever: Retriever,
    queries: list[EvalQuery],
    k: int = 10,
    gate: Gate | None = None,
) -> RetrievalReport:
    """Retrieve top-k for every query and aggregate IR metrics."""
    results: list[QueryResult] = []
    for query in queries:
        hits = retriever.retrieve(query.question, k)
        retrieved_chunks = [sc.chunk.id for sc in hits]
        retrieved_sections = _unique([sid for sc in hits for sid in _section_ids(sc)])
        gate_refused = (
            gate.would_refuse(query.question, hits) if gate is not None else None
        )

        if query.answerable:
            metrics = _retrieval_metrics(
                retrieved_chunks,
                set(query.relevant_chunk_ids),
                retrieved_sections,
                set(query.relevant_section_ids),
                k,
            )
            hit_ids = set(retrieved_chunks) & set(query.relevant_chunk_ids)
            if gate_refused:
                error_type = "over_refusal" if hit_ids else "retrieval_miss"
            elif not hit_ids:
                error_type = "retrieval_miss"
            elif retrieved_chunks[0] not in query.relevant_chunk_ids:
                error_type = "ranking_error"
            elif len(hit_ids) < len(query.relevant_chunk_ids):
                error_type = "partial_hit"
            else:
                error_type = "hit"
        else:
            metrics = {}
            if gate is None:
                error_type = "unanswerable_skipped"
            else:
                error_type = "correct_refusal" if gate_refused else "unanswerable_answered"

        results.append(
            QueryResult(
                query_id=query.id,
                question=query.question,
                category=query.category,
                language=query.language,
                answerable=query.answerable,
                retrieved_chunk_ids=retrieved_chunks,
                retrieved_section_ids=retrieved_sections,
                metrics=metrics,
                error_type=error_type,
                gate_refused=gate_refused,
            )
        )

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    metrics: dict[str, float] = {}
    ci: dict[str, tuple[float, float]] = {}
    for metric in METRIC_NAMES:
        values = [r.metrics[metric] for r in answerable]
        point, low, high = bootstrap_mean_ci(values)
        metrics[metric] = point
        ci[metric] = (low, high)

    per_category: dict[str, dict[str, float]] = {}
    for category in sorted({r.category for r in answerable}):
        rows = [r for r in answerable if r.category == category]
        per_category[category] = {
            "n": float(len(rows)),
            **{
                metric: mean([r.metrics[metric] for r in rows])
                for metric in METRIC_NAMES
            },
        }

    error_counts = Counter(r.error_type for r in results)
    refusal = None
    if gate is not None:
        correct_refusals = sum(1 for r in unanswerable if r.gate_refused)
        over_refusals = sum(1 for r in answerable if r.gate_refused)
        refusal = {
            "unanswerable_refusal_rate": (
                correct_refusals / len(unanswerable) if unanswerable else 0.0
            ),
            "answerable_over_refusal_rate": (
                over_refusals / len(answerable) if answerable else 0.0
            ),
            "gate_accuracy": (
                (correct_refusals + len(answerable) - over_refusals) / len(results)
                if results
                else 0.0
            ),
            "n_correct_refusals": float(correct_refusals),
            "n_over_refusals": float(over_refusals),
        }

    return RetrievalReport(
        name=name,
        k=k,
        n_queries=len(queries),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        metrics=metrics,
        ci=ci,
        per_category=per_category,
        error_counts={name: error_counts.get(name, 0) for name in ERROR_TYPES},
        per_query=results,
        refusal=refusal,
    )


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _fmt_ci(value: float, interval: tuple[float, float], digits: int = 3) -> str:
    low, high = interval
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _metric_label(metric: str, k: int) -> str:
    if metric in {"mrr", "map"}:
        return metric.upper() if metric == "mrr" else "MAP"
    return f"{metric.replace('_', ' ')}@{k}"


def compare_to_markdown(reports: list[RetrievalReport]) -> str:
    """Render the aggregate, per-category, error-taxonomy and failure sections."""
    if not reports:
        return "(no reports)"
    k = reports[0].k
    lines: list[str] = []

    lines.append(f"## Aggregate retrieval metrics (k={k}, 95% bootstrap CI)")
    lines.append("")
    header = ["System"] + [_metric_label(m, k) for m in METRIC_NAMES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for report in reports:
        cells = [report.name]
        for metric in METRIC_NAMES:
            cells.append(_fmt_ci(report.metrics[metric], report.ci[metric]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"Answerable queries: {reports[0].n_answerable}; "
        f"unanswerable queries: {reports[0].n_unanswerable}."
    )
    lines.append("")

    if any(report.refusal is not None for report in reports):
        lines.append("## Refusal gate")
        lines.append("")
        lines.append(
            "| System | unanswerable refused | answerable over-refused | gate accuracy |"
        )
        lines.append("|---|---|---|---|")
        for report in reports:
            if report.refusal is None:
                continue
            refusal = report.refusal
            lines.append(
                "| "
                + " | ".join(
                    [
                        report.name,
                        _fmt(refusal["unanswerable_refusal_rate"]),
                        _fmt(refusal["answerable_over_refusal_rate"]),
                        _fmt(refusal["gate_accuracy"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Per-category recall@k")
    lines.append("")
    categories = sorted(reports[0].per_category)
    header = ["System"] + categories
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for report in reports:
        cells = [report.name]
        for category in categories:
            cells.append(_fmt(report.per_category[category]["recall"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Error taxonomy")
    lines.append("")
    header = ["System"] + list(ERROR_TYPES)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for report in reports:
        cells = [report.name] + [str(report.error_counts[err]) for err in ERROR_TYPES]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Failure examples")
    lines.append("")
    for report in reports:
        failures = [
            r
            for r in report.per_query
            if r.error_type in {"retrieval_miss", "ranking_error", "over_refusal", "unanswerable_answered"}
        ]
        lines.append(f"### {report.name} ({len(failures)} failures)")
        lines.append("")
        if not failures:
            lines.append("No failures in this run.")
            lines.append("")
            continue
        for row in failures[:12]:
            lines.append(
                f"- `{row.query_id}` [{row.category}] {row.question}  \n"
                f"  error=`{row.error_type}`  \n"
                f"  retrieved={row.retrieved_chunk_ids[:3]}"
            )
        lines.append("")

    return "\n".join(lines)

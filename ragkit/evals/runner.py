"""Runner: evaluate a retriever over the labelled set and aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from ..retrievers.base import Retriever
from .dataset import DEFAULT_EVAL_SET, EvalQuery
from .metrics import average_precision, hit_rate, mrr, precision_at_k, recall_at_k


@dataclass
class RetrievalReport:
    """Aggregated retrieval metrics at a fixed cut-off ``k``."""

    k: int
    recall: float
    precision: float
    mrr: float
    hit_rate: float
    map: float
    n_queries: int

    def as_dict(self) -> dict[str, float]:
        return {
            f"recall@{self.k}": self.recall,
            f"precision@{self.k}": self.precision,
            "mrr": self.mrr,
            f"hit_rate@{self.k}": self.hit_rate,
            "map": self.map,
        }


def _retrieved_doc_ids(retriever: Retriever, question: str, k: int) -> list[str]:
    """Retrieve k chunks and collapse to document ids, preserving rank order.

    A document is credited at the rank of its *first* retrieved chunk, so a doc
    with many chunks isn't double-counted.
    """
    hits = retriever.retrieve(question, k)
    seen: list[str] = []
    for sc in hits:
        if sc.chunk.doc_id not in seen:
            seen.append(sc.chunk.doc_id)
    return seen


def evaluate_retriever(
    retriever: Retriever,
    queries: tuple[EvalQuery, ...] = DEFAULT_EVAL_SET,
    k: int = 5,
) -> RetrievalReport:
    """Compute mean retrieval metrics for ``retriever`` over ``queries``."""
    recalls, precisions, mrrs, hits, maps = [], [], [], [], []
    for q in queries:
        retrieved = _retrieved_doc_ids(retriever, q.question, k)
        relevant = set(q.relevant_doc_ids)
        recalls.append(recall_at_k(retrieved, relevant, k))
        precisions.append(precision_at_k(retrieved, relevant, k))
        mrrs.append(mrr(retrieved, relevant))
        hits.append(hit_rate(retrieved, relevant, k))
        maps.append(average_precision(retrieved, relevant))

    return RetrievalReport(
        k=k,
        recall=mean(recalls) if recalls else 0.0,
        precision=mean(precisions) if precisions else 0.0,
        mrr=mean(mrrs) if mrrs else 0.0,
        hit_rate=mean(hits) if hits else 0.0,
        map=mean(maps) if maps else 0.0,
        n_queries=len(queries),
    )


def compare_to_markdown(reports: dict[str, RetrievalReport]) -> str:
    """Render a comparison table of several named retrievers."""
    if not reports:
        return "(no reports)"
    any_report = next(iter(reports.values()))
    k = any_report.k
    header = ["Retriever", f"recall@{k}", f"precision@{k}", "MRR", f"hit@{k}", "MAP"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, r in reports.items():
        lines.append(
            f"| {name} | {r.recall:.3f} | {r.precision:.3f} | {r.mrr:.3f} | "
            f"{r.hit_rate:.3f} | {r.map:.3f} |"
        )
    return "\n".join(lines)

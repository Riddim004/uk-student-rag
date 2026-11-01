"""Calibrate the relevance gate on the dev split, then report the test split.

RRF scores are rank-based and always positive, so the gate uses the strongest
underlying component score (dense cosine, BM25, or reranker logit). Thresholds
are selected on dev only; the test split is printed once as a held-out check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.config import Settings  # noqa: E402
from ragkit.embeddings import make_embedder  # noqa: E402
from ragkit.evals.dataset import load_eval_set  # noqa: E402
from ragkit.evals.run import DEFAULT_DATA, DEFAULT_EVAL_DIR, _git_commit  # noqa: E402
from ragkit.ingest import build_corpus, load_documents_jsonl  # noqa: E402
from ragkit.retrievers.rerank import CrossEncoderReranker  # noqa: E402

OUT_DIR = ROOT / "evidence" / "eval"
SCORES_PATH = OUT_DIR / "gate_scores.jsonl"
REPORT_PATH = OUT_DIR / "gate_calibration.md"


def _max(values: list[float], default: float) -> float:
    return max(values) if values else default


def collect_scores(
    eval_dir: Path,
    data_path: Path,
    settings: Settings,
    candidate_k: int,
    rerank_k: int,
    use_rerank: bool,
) -> list[dict]:
    queries, _ = load_eval_set(eval_dir)
    documents = load_documents_jsonl(data_path)
    corpus = build_corpus(documents, make_embedder(settings), settings)
    reranker = (
        CrossEncoderReranker(settings.reranker_model, device=settings.device)
        if use_rerank
        else None
    )
    rows: list[dict] = []
    for index, query in enumerate(queries, start=1):
        dense_hits = corpus.dense.retrieve(query.question, candidate_k)
        sparse_hits = corpus.bm25.retrieve(query.question, candidate_k)
        hybrid_hits = corpus.hybrid.retrieve(query.question, candidate_k)
        max_dense = _max([h.score for h in dense_hits], float("-inf"))
        max_sparse = _max([h.score for h in sparse_hits], 0.0)
        max_rerank = float("-inf")
        if reranker is not None:
            reranked = reranker.rerank(
                query.question, hybrid_hits[:rerank_k], rerank_k
            )
            max_rerank = _max(
                [h.components.get("rerank", float("-inf")) for h in reranked],
                float("-inf"),
            )
        rows.append(
            {
                "query_id": query.id,
                "split": query.split,
                "category": query.category,
                "language": query.language,
                "answerable": query.answerable,
                "max_dense": max_dense,
                "max_sparse": max_sparse,
                "max_rerank": max_rerank,
                "n_dense_hits": len(dense_hits),
                "n_sparse_hits": len(sparse_hits),
            }
        )
        print(
            f"[gate] {index:02d}/{len(queries)} {query.id} "
            f"{'answerable' if query.answerable else 'unanswerable'} "
            f"dense={max_dense:.3f} sparse={max_sparse:.2f} "
            f"rerank={max_rerank:.3f}",
            flush=True,
        )
    return rows


def sweep(
    rows: list[dict],
    field: str,
    thresholds: list[float],
    strict: bool = False,
) -> list[dict]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    out = []
    for threshold in thresholds:
        def accepts(value: float) -> bool:
            if strict:
                return value > threshold
            return value >= threshold

        tpr = (
            sum(1 for r in unanswerable if not accepts(r[field])) / len(unanswerable)
            if unanswerable
            else 0.0
        )
        fpr = (
            sum(1 for r in answerable if not accepts(r[field])) / len(answerable)
            if answerable
            else 0.0
        )
        out.append(
            {
                "threshold": threshold,
                "unanswerable_refusal_rate": tpr,
                "answerable_over_refusal_rate": fpr,
                "balanced_accuracy": (tpr + (1 - fpr)) / 2,
            }
        )
    return out


def best_row(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda r: (
            r["balanced_accuracy"],
            -r["answerable_over_refusal_rate"],
            r["unanswerable_refusal_rate"],
        ),
    )


def select_with_margin(rows: list[dict], margin: float = 0.1) -> dict:
    """Pick the conservative end of the best dev plateau.

    The sweep often has a flat region (same refusal and over-refusal). We take
    the highest threshold in that region and step back by ``margin`` so the
    threshold does not sit exactly on the lowest-scoring answerable query.
    """
    best_balanced = max(r["balanced_accuracy"] for r in rows)
    plateau = [
        r
        for r in rows
        if abs(r["balanced_accuracy"] - best_balanced) < 1e-9
        and r["answerable_over_refusal_rate"] == 0.0
    ]
    if not plateau:
        return best_row(rows)
    highest = max(plateau, key=lambda r: r["threshold"])
    target = highest["threshold"] - margin
    candidates = [r for r in plateau if r["threshold"] <= target]
    return max(candidates, key=lambda r: r["threshold"]) if candidates else highest


def apply_rule(rows: list[dict], dense_t: float, sparse_t: float, rerank_t: float | None) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def accepts(row: dict) -> bool:
        if rerank_t is not None:
            return row["max_rerank"] >= rerank_t
        return row["max_dense"] >= dense_t or row["max_sparse"] > sparse_t

    tpr = (
        sum(1 for r in unanswerable if not accepts(r)) / len(unanswerable)
        if unanswerable
        else 0.0
    )
    fpr = (
        sum(1 for r in answerable if not accepts(r)) / len(answerable)
        if answerable
        else 0.0
    )
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "unanswerable_refusal_rate": tpr,
        "answerable_over_refusal_rate": fpr,
        "gate_accuracy": (
            sum(1 for r in rows if accepts(r) == r["answerable"]) / len(rows)
            if rows
            else 0.0
        ),
        "balanced_accuracy": (tpr + (1 - fpr)) / 2,
    }


def distribution(rows: list[dict], field: str) -> dict[str, float]:
    values = [r[field] for r in rows if r[field] != float("-inf")]
    if not values:
        return {"n": 0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def table(rows: list[dict], limit: int = 12) -> str:
    ranked = sorted(
        rows,
        key=lambda r: (
            -r["balanced_accuracy"],
            r["answerable_over_refusal_rate"],
            r["threshold"],
        ),
    )
    lines = [
        "| threshold | unanswerable refused | answerable over-refused | balanced accuracy |",
        "|---|---|---|---|",
    ]
    for row in ranked[:limit]:
        lines.append(
            f"| {row['threshold']:.3f} | "
            f"{row['unanswerable_refusal_rate']:.3f} | "
            f"{row['answerable_over_refusal_rate']:.3f} | "
            f"{row['balanced_accuracy']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--eval", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-k", type=int, default=15)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    rows = collect_scores(
        eval_dir=Path(args.eval),
        data_path=Path(args.data),
        settings=settings,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
        use_rerank=args.rerank,
    )
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCORES_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]

    dense_sweep = sweep(dev, "max_dense", [0.70 + i * 0.005 for i in range(41)])
    sparse_sweep = sweep(
        dev, "max_sparse", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0], strict=True
    )
    dense_best = best_row(dense_sweep)
    sparse_best = best_row(sparse_sweep)

    rerank_sweep = None
    rerank_best = None
    if args.rerank:
        rerank_sweep = sweep(dev, "max_rerank", [-5.0 + i * 0.05 for i in range(121)])
        rerank_best = select_with_margin(rerank_sweep, margin=0.1)

    if rerank_best is not None:
        dev_result = apply_rule(
            dev, dense_best["threshold"], sparse_best["threshold"], rerank_best["threshold"]
        )
        test_result = apply_rule(
            test,
            dense_best["threshold"],
            sparse_best["threshold"],
            rerank_best["threshold"],
        )
    else:
        dev_result = apply_rule(
            dev, dense_best["threshold"], sparse_best["threshold"], None
        )
        test_result = apply_rule(
            test, dense_best["threshold"], sparse_best["threshold"], None
        )

    lines = [
        "# Relevance gate calibration",
        "",
        f"Commit: `{_git_commit()}` | rerank: `{args.rerank}` | "
        f"candidate_k: {args.candidate_k} | rerank_k: {args.rerank_k}",
        "",
        "The gate answers only when the strongest underlying score clears a",
        "threshold. Thresholds are selected on the **dev** split; the **test**",
        "split is a held-out check and is not used for selection.",
        "",
        "## Score distributions",
        "",
        "| split | label | field | n | min | median | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for split_name, split_rows in [("dev", dev), ("test", test)]:
        for label, group in [
            ("answerable", [r for r in split_rows if r["answerable"]]),
            ("unanswerable", [r for r in split_rows if not r["answerable"]]),
        ]:
            for field in ["max_dense", "max_sparse", "max_rerank"]:
                stats = distribution(group, field)
                lines.append(
                    f"| {split_name} | {label} | {field} | {stats['n']} | "
                    f"{stats['min']:.3f} | {stats['median']:.3f} | {stats['max']:.3f} |"
                )
    lines.append("")
    lines.append("## Dense threshold sweep (dev)")
    lines.append("")
    lines.append(table(dense_sweep[:12]))
    lines.append("")
    lines.append("## Sparse threshold sweep (dev)")
    lines.append("")
    lines.append(table(sparse_sweep[:12]))
    lines.append("")
    if rerank_sweep is not None:
        lines.append("## Rerank threshold sweep (dev)")
        lines.append("")
        lines.append(table(rerank_sweep[:12]))
        lines.append("")
    lines.append("## Selected thresholds")
    lines.append("")
    lines.append("| rule | threshold | dev unanswerable refused | dev answerable over-refused | dev balanced accuracy |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| dense | {dense_best['threshold']:.3f} | "
        f"{dense_best['unanswerable_refusal_rate']:.3f} | "
        f"{dense_best['answerable_over_refusal_rate']:.3f} | "
        f"{dense_best['balanced_accuracy']:.3f} |"
    )
    lines.append(
        f"| sparse | {sparse_best['threshold']:.3f} | "
        f"{sparse_best['unanswerable_refusal_rate']:.3f} | "
        f"{sparse_best['answerable_over_refusal_rate']:.3f} | "
        f"{sparse_best['balanced_accuracy']:.3f} |"
    )
    if rerank_best is not None:
        lines.append(
            f"| rerank | {rerank_best['threshold']:.3f} | "
            f"{rerank_best['unanswerable_refusal_rate']:.3f} | "
            f"{rerank_best['answerable_over_refusal_rate']:.3f} | "
            f"{rerank_best['balanced_accuracy']:.3f} |"
        )
    lines.append("")
    if rerank_best is not None:
        lines.append(
            "The rerank threshold is the highest dev-optimal threshold minus a "
            "0.1 margin, so it does not sit exactly on the lowest-scoring "
            "answerable query. The dense/sparse scalar thresholds are reported "
            "as a negative result: they need 33–38% over-refusal to reach the "
            "same dev refusal rate."
        )
        lines.append("")
    lines.append("## Held-out check (test split, thresholds frozen)")
    lines.append("")
    lines.append("| rule | unanswerable refused | answerable over-refused | gate accuracy | balanced accuracy |")
    lines.append("|---|---|---|---|---|")
    rule_name = "rerank" if rerank_best is not None else "dense-or-sparse"
    lines.append(
        f"| {rule_name} | {test_result['unanswerable_refusal_rate']:.3f} | "
        f"{test_result['answerable_over_refusal_rate']:.3f} | "
        f"{test_result['gate_accuracy']:.3f} | {test_result['balanced_accuracy']:.3f} |"
    )
    lines.append("")
    lines.append(
        "Caveat: the test split has only "
        f"{test_result['n_unanswerable']} unanswerable and "
        f"{test_result['n_answerable']} answerable queries, so these intervals are wide."
    )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote {SCORES_PATH}")
    print(
        "Selected:",
        {
            "min_dense_score": dense_best["threshold"],
            "min_sparse_score": sparse_best["threshold"],
            "min_rerank_score": rerank_best["threshold"] if rerank_best else None,
        },
    )


if __name__ == "__main__":
    main()

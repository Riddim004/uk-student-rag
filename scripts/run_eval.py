"""Run the retrieval eval and write reproducible evidence artifacts.

Outputs:
    evidence/eval/report.md
    evidence/eval/results.json
    evidence/eval/errors.md

By default, runs both the dev and test splits. Gate thresholds are tuned on dev;
the test split is the final held-out report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.config import Settings  # noqa: E402
from ragkit.evals.run import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_EVAL_DIR,
    metadata,
    render_report,
    results_payload,
    run_evaluations,
)
from ragkit.evals.runner import ERROR_TYPES  # noqa: E402


def render_errors(run, include_rerank: bool) -> str:
    meta = metadata(run, include_rerank)
    by_id = {query.id: query for query in run.queries}
    lines = [
        f"# Error analysis — {meta['split']} split",
        "",
        f"Commit: `{meta['git_commit']}` | corpus: `{meta['corpus_hash'][:16]}` "
        f"| eval: `{meta['eval_hash'][:16]}`",
        "",
        "Each row is a query whose retrieval or gate outcome was not a clean hit.",
        "`relevant` is the labelled set; `retrieved` is the top-3 ranked chunk ids.",
        "",
    ]
    for report in run.reports:
        lines.append(f"## {report.name}")
        lines.append("")
        lines.append("| Error type | Count |")
        lines.append("|---|---|")
        for error_type in ERROR_TYPES:
            lines.append(f"| {error_type} | {report.error_counts[error_type]} |")
        lines.append("")
        for error_type in ERROR_TYPES:
            if error_type in {"hit", "correct_refusal"}:
                continue
            rows = [r for r in report.per_query if r.error_type == error_type]
            if not rows:
                continue
            lines.append(f"### {error_type} ({len(rows)})")
            lines.append("")
            for row in rows:
                query = by_id[row.query_id]
                lines.append(f"- `{row.query_id}` [{row.category}] {query.question}")
                lines.append(f"  - retrieved: `{row.retrieved_chunk_ids[:3]}`")
                lines.append(f"  - relevant: `{list(query.relevant_chunk_ids)}`")
                lines.append(f"  - reference: {query.reference_answer}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--eval", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--embedding-backend", choices=["local", "openai", "hashing"]
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--rerank-k",
        type=int,
        help="candidate pool size for reranking (default: settings.rerank_candidate_k)",
    )
    parser.add_argument(
        "--split",
        choices=["all", "dev", "test", "both"],
        default="both",
        help="which split(s) to evaluate (default: both)",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.embedding_backend:
        settings = replace(settings, embedding_backend=args.embedding_backend)
    if args.rerank_k:
        settings = replace(settings, rerank_candidate_k=args.rerank_k)

    split_values = (
        ["dev", "test"] if args.split == "both" else [None if args.split == "all" else args.split]
    )
    runs = run_evaluations(
        data_path=args.data,
        eval_dir=args.eval,
        settings=settings,
        k=args.k,
        include_rerank=args.rerank,
        splits=tuple(split_values),
        verbose=True,
    )

    if len(runs) == 1:
        report = render_report(runs[0], include_rerank=args.rerank)
        payload = results_payload(runs[0], include_rerank=args.rerank)
    else:
        report = "# Retrieval evaluation\n\n" + "\n\n---\n\n".join(
            render_report(run, include_rerank=args.rerank) for run in runs
        )
        payload = {
            "generated_at": metadata(runs[0], args.rerank)["generated_at"],
            "splits": {
                run.split or "all": results_payload(run, include_rerank=args.rerank)
                for run in runs
            },
        }

    eval_dir = Path(args.eval)
    errors = render_errors(runs[-1], include_rerank=args.rerank)
    (eval_dir / "report.md").write_text(report, encoding="utf-8")
    (eval_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "errors.md").write_text(errors, encoding="utf-8")
    print(report)
    print(f"\nWrote {eval_dir / 'report.md'}")
    print(f"Wrote {eval_dir / 'results.json'}")
    print(f"Wrote {eval_dir / 'errors.md'}")


if __name__ == "__main__":
    main()

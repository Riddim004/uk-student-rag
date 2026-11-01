"""CLI: ``python -m ragkit.evals``.

Runs the versioned eval set against bm25 / dense / hybrid (and optionally
hybrid+rerank), then prints a Markdown report with confidence intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..config import Settings
from .run import DEFAULT_DATA, DEFAULT_EVAL_DIR, render_report, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ragkit.evals",
        description="Evaluate and compare retrievers on the labelled query set.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="documents.jsonl")
    parser.add_argument("--eval", default=str(DEFAULT_EVAL_DIR), help="eval directory")
    parser.add_argument("-k", type=int, default=10, help="cut-off k (default: 10)")
    parser.add_argument(
        "--embedding-backend",
        choices=["local", "openai", "hashing"],
        help="override EMBEDDING_BACKEND",
    )
    parser.add_argument("--rerank", action="store_true", help="add hybrid+rerank")
    parser.add_argument("-o", "--output", help="write the report to this file")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.embedding_backend:
        settings = replace(settings, embedding_backend=args.embedding_backend)

    run = run_evaluation(
        data_path=args.data,
        eval_dir=args.eval,
        settings=settings,
        k=args.k,
        include_rerank=args.rerank,
    )
    report = render_report(run, include_rerank=args.rerank)
    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

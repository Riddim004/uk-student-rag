"""Compare retrievers on the labelled set: ``python -m ragkit.evals``.

Runs **offline by default** using the deterministic ``HashingEmbedder``, so it
needs no API key — you get a bm25 / dense / hybrid comparison table out of the
box. Pass ``--openai`` to use real embeddings instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Settings
from ..embeddings import HashingEmbedder, OpenAIEmbedder
from ..ingest import build_corpus, load_documents
from .runner import compare_to_markdown, evaluate_retriever

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "uk_student_visa"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ragkit.evals",
        description="Evaluate and compare retrievers on the labelled query set.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="documents directory")
    parser.add_argument("-k", type=int, default=5, help="cut-off k (default: 5)")
    parser.add_argument(
        "--openai", action="store_true", help="use real OpenAI embeddings (needs API key)"
    )
    parser.add_argument("-o", "--output", help="write the comparison table to this file")
    args = parser.parse_args()

    settings = Settings.from_env()
    embedder = (
        OpenAIEmbedder(settings.embedding_model, settings.api_base)
        if args.openai
        else HashingEmbedder()
    )

    corpus = build_corpus(load_documents(args.data), embedder, settings)
    reports = {
        "bm25": evaluate_retriever(corpus.bm25, k=args.k),
        "dense": evaluate_retriever(corpus.dense, k=args.k),
        "hybrid": evaluate_retriever(corpus.hybrid, k=args.k),
    }

    table = compare_to_markdown(reports)
    print(table)
    if args.output:
        Path(args.output).write_text(table, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

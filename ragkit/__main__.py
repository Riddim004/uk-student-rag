"""CLI: ``python -m ragkit "your question"``.

Builds the corpus from the bundled data directory, then answers with the chosen
retriever. Requires an API key (real embeddings + chat). Use ``--retriever`` to
compare bm25 / dense / hybrid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .embeddings import OpenAIEmbedder
from .generator import LLMGenerator
from .ingest import build_corpus, load_documents
from .llm import OpenAIChatModel
from .pipeline import RAGPipeline

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uk_student_visa"


def _pick_retriever(corpus, name: str):
    return {"bm25": corpus.bm25, "dense": corpus.dense, "hybrid": corpus.hybrid}[name]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ragkit", description="Ask a question against the document corpus."
    )
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="documents directory")
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    embedder = OpenAIEmbedder(settings.embedding_model, settings.api_base)
    corpus = build_corpus(load_documents(args.data), embedder, settings)

    generator = LLMGenerator(OpenAIChatModel(settings), settings)
    pipeline = RAGPipeline(_pick_retriever(corpus, args.retriever), generator, settings.top_k)

    answer = pipeline.answer(args.question)
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            print(f"  [{c.marker}] {c.title or c.doc_id} — {c.source}")


if __name__ == "__main__":
    main()

"""CLI: ask a question against the bundled gov.uk snapshot.

Examples:
    python -m ragkit "How many hours can I work during term time?"
    python -m ragkit "毕业后可以留多久？" --no-llm
    python -m ragkit "..." --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .config import Settings
from .embeddings import make_embedder
from .errors import RagkitError
from .generator import ExtractiveGenerator, FallbackGenerator, LLMGenerator
from .ingest import build_corpus, load_documents_jsonl
from .llm import OpenAIChatModel
from .pipeline import RAGPipeline
from .relevance import RelevanceGate, RerankGate
from .retrievers import CrossEncoderReranker
from .types import Answer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "corpus" / "documents.jsonl"


def _pick_retriever(corpus, name: str):
    return {"bm25": corpus.bm25, "dense": corpus.dense, "hybrid": corpus.hybrid}[name]


def _build_generator(settings: Settings, no_llm: bool):
    if no_llm or not os.getenv("OPENAI_API_KEY"):
        return ExtractiveGenerator(), "extractive"
    primary = LLMGenerator(OpenAIChatModel(settings), settings)
    if settings.fallback_to_extractive:
        return FallbackGenerator(primary), "llm"
    return primary, "llm"


def _print_human(answer: Answer, timings: dict, gate_line: str) -> None:
    print(answer.text)
    if answer.citations:
        print("\nSources")
        for citation in answer.citations:
            section = citation.section_path[-1] if citation.section_path else ""
            updated = f" · updated {citation.updated_at[:10]}" if citation.updated_at else ""
            print(f"  [{citation.marker}] {citation.title} — {section}{updated}")
            print(f"      {citation.source}")
    flags = [f"mode={answer.mode or 'refusal'}"]
    if answer.degraded:
        flags.append("degraded=extractive")
    print(f"\n{' | '.join(flags)} | {gate_line} | {timings['total']} ms")
    if answer.degraded:
        print(f"degraded_reason: {answer.degraded_reason}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragkit", description=__doc__)
    parser.add_argument("question", help="question to answer")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument(
        "--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid"
    )
    parser.add_argument("--embedding-backend", choices=["local", "openai", "hashing"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true", help="use the scalar gate")
    parser.add_argument("--no-gate", action="store_true", help="disable gating")
    parser.add_argument("--no-llm", action="store_true", help="extractive answer only")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.embedding_backend:
        settings = replace(settings, embedding_backend=args.embedding_backend)
    if args.top_k:
        settings = replace(settings, top_k=args.top_k)
    if args.candidate_k:
        settings = replace(settings, candidate_k=args.candidate_k)

    try:
        documents = load_documents_jsonl(args.data)
        corpus = build_corpus(documents, make_embedder(settings), settings)

        gate = None
        gate_mode = "none"
        if not args.no_gate:
            if args.no_rerank:
                gate = RelevanceGate.from_settings(settings)
                gate_mode = "scalar"
            else:
                reranker = CrossEncoderReranker(
                    model_name=settings.reranker_model,
                    device=settings.device,
                )
                gate = RerankGate(
                    reranker,
                    min_score=settings.min_rerank_score,
                    candidate_k=settings.rerank_candidate_k,
                )
                gate_mode = "rerank"

        generator, product_mode = _build_generator(settings, args.no_llm)
        pipeline = RAGPipeline(
            _pick_retriever(corpus, args.retriever),
            generator,
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
            gate=gate,
        )
        answer, trace = pipeline.answer_with_trace(args.question)
    except RagkitError as exc:
        print(f"error ({type(exc).__name__}): {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if trace.gate_threshold is not None:
        best = trace.gate_best if trace.gate_best is not None else float("-inf")
        gate_line = (
            f"gate={gate_mode} best={best:.2f} "
            f"threshold={trace.gate_threshold:.2f}"
        )
    else:
        gate_line = f"gate={gate_mode}"

    if args.json:
        payload = {
            "question": args.question,
            "answer": asdict(answer),
            "timings_ms": trace.timings_ms,
            "gate": {
                "mode": gate_mode,
                "accepted": trace.gate_accepted,
                "best": trace.gate_best,
                "threshold": trace.gate_threshold,
            },
            "product_mode": product_mode,
            "corpus": {
                "documents": len(corpus.documents),
                "chunks": len(corpus.chunks),
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(answer, trace.timings_ms, gate_line)


if __name__ == "__main__":
    main()

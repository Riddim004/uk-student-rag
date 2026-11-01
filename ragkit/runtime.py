"""Build the runtime once and serve it from the CLI/API.

The product path is: hybrid retrieval → cross-encoder gate → LLM with a cited
extractive fallback. If no API key is present, or the chat client cannot be
constructed, the runtime stays fully functional in ``extractive`` demo mode.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Settings
from .embeddings import make_embedder
from .generator import ExtractiveGenerator, FallbackGenerator, Generator, LLMGenerator
from .ingest import Corpus, build_corpus, load_documents_jsonl
from .llm import OpenAIChatModel
from .pipeline import PipelineTrace, RAGPipeline
from .relevance import RelevanceGate, RerankGate
from .retrievers import CrossEncoderReranker
from .types import Answer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "corpus" / "documents.jsonl"
DEFAULT_EVAL = ROOT / "evidence" / "eval"

EXAMPLE_IDS = ("q01", "q11", "q21", "q31", "q43", "q39")


@dataclass
class Runtime:
    settings: Settings
    corpus: Corpus
    pipeline: RAGPipeline
    corpus_manifest: dict
    eval_summary: dict
    examples: list[dict]
    mode: str
    gate_mode: str
    warnings: list[str] = field(default_factory=list)
    started_at: str = ""
    startup_ms: float = 0.0

    def ask(
        self,
        question: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[Answer, PipelineTrace]:
        return self.pipeline.answer_with_trace(question, on_stage=on_stage)


def _read_json(path: Path, default: dict | None = None) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default or {}


def _eval_summary(eval_dir: Path) -> dict:
    results = _read_json(eval_dir / "results.json")
    stats = _read_json(eval_dir / "dataset_stats.json")
    test = results.get("splits", {}).get("test", {})
    hybrid = test.get("systems", {}).get("hybrid", {})
    gate = test.get("systems", {}).get("hybrid+rerank-gate", {})
    return {
        "generated_at": results.get("generated_at"),
        "dataset": stats,
        "metadata": test.get("metadata", {}),
        "hybrid": {
            "recall": hybrid.get("metrics", {}).get("recall"),
            "ndcg": hybrid.get("metrics", {}).get("ndcg"),
            "mrr": hybrid.get("metrics", {}).get("mrr"),
            "hit": hybrid.get("metrics", {}).get("hit"),
        },
        "gate": gate.get("refusal", {}),
    }


def _examples(eval_dir: Path) -> list[dict]:
    path = eval_dir / "queries.jsonl"
    if not path.exists():
        return []
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return [rows[qid] for qid in EXAMPLE_IDS if qid in rows]


def build_runtime(
    settings: Settings | None = None,
    data_path: str | Path = DEFAULT_DATA,
    eval_dir: str | Path = DEFAULT_EVAL,
    use_llm: bool | None = None,
    use_rerank: bool = True,
) -> Runtime:
    started = time.perf_counter()
    settings = settings or Settings.from_env()
    corpus_manifest = _read_json(Path(data_path).parent / "manifest.json")
    warnings: list[str] = []

    documents = load_documents_jsonl(data_path)
    corpus = build_corpus(documents, make_embedder(settings), settings)

    gate = None
    gate_mode = "none"
    if use_rerank:
        try:
            reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.device,
            )
            reranker.rerank(
                "warmup",
                corpus.hybrid.retrieve("warmup", 2),
                2,
            )
            gate = RerankGate(
                reranker,
                min_score=settings.min_rerank_score,
                candidate_k=settings.rerank_candidate_k,
            )
            gate_mode = "rerank"
        except Exception as exc:  # model load/inference is environment-dependent
            warnings.append(f"reranker unavailable, using scalar gate: {exc}")
            gate = RelevanceGate.from_settings(settings)
            gate_mode = "scalar"
    else:
        gate = RelevanceGate.from_settings(settings)
        gate_mode = "scalar"

    llm_available = bool(os.getenv("OPENAI_API_KEY")) if use_llm is None else use_llm
    generator: Generator
    mode = "extractive"
    if llm_available:
        try:
            primary = LLMGenerator(OpenAIChatModel(settings), settings)
            generator = (
                FallbackGenerator(primary)
                if settings.fallback_to_extractive
                else primary
            )
            mode = "llm"
        except Exception as exc:  # bad base URL, missing key, SDK error
            warnings.append(f"chat model unavailable, using extractive mode: {exc}")
            generator = ExtractiveGenerator()
    else:
        generator = ExtractiveGenerator()

    pipeline = RAGPipeline(
        corpus.hybrid,
        generator,
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        gate=gate,
    )
    return Runtime(
        settings=settings,
        corpus=corpus,
        pipeline=pipeline,
        corpus_manifest=corpus_manifest,
        eval_summary=_eval_summary(Path(eval_dir)),
        examples=_examples(Path(eval_dir)),
        mode=mode,
        gate_mode=gate_mode,
        warnings=warnings,
        started_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        startup_ms=round((time.perf_counter() - started) * 1000, 1),
    )

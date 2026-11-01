"""Measure latency, throughput, memory, and per-query cost.

Run: python scripts/measure.py
Writes evidence/performance/{report.md,results.json} and
evidence/cost/{report.md,results.json}. Every number is tied to the corpus,
eval set, git state, and machine it ran on.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.config import Settings  # noqa: E402
from ragkit.embeddings import make_embedder  # noqa: E402
from ragkit.evals.dataset import load_eval_set  # noqa: E402
from ragkit.generator import SYSTEM_PROMPT, ExtractiveGenerator, build_context  # noqa: E402
from ragkit.ingest import build_corpus, load_documents_jsonl  # noqa: E402
from ragkit.pipeline import RAGPipeline  # noqa: E402
from ragkit.relevance import RerankGate  # noqa: E402
from ragkit.retrievers import (  # noqa: E402
    BM25Retriever,
    CrossEncoderReranker,
    RerankedRetriever,
)

CORPUS = ROOT / "data" / "corpus"
EVAL = ROOT / "evidence" / "eval"
PERF = ROOT / "evidence" / "performance"
COST = ROOT / "evidence" / "cost"

# Configurable snapshot. Verify against the provider's pricing page before
# quoting these externally; the benchmark reads these constants, not the web.
PRICE_SNAPSHOT = {
    "as_of": "2026-11-01",
    "source": "https://openai.com/api/pricing/",
    "gpt-4o-mini": {
        "input_per_1m_usd": float(os.getenv("PRICE_GPT4O_MINI_INPUT", "0.15")),
        "output_per_1m_usd": float(os.getenv("PRICE_GPT4O_MINI_OUTPUT", "0.60")),
    },
    "text-embedding-3-small": {
        "per_1m_usd": float(os.getenv("PRICE_EMBEDDING_3_SMALL", "0.02")),
    },
}


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[index]


def _latency_summary(samples_ms: list[float]) -> dict:
    return {
        "n": len(samples_ms),
        "p50_ms": round(_percentile(samples_ms, 50), 2),
        "p90_ms": round(_percentile(samples_ms, 90), 2),
        "p95_ms": round(_percentile(samples_ms, 95), 2),
        "p99_ms": round(_percentile(samples_ms, 99), 2),
        "mean_ms": round(statistics.fmean(samples_ms), 2),
        "max_ms": round(max(samples_ms), 2),
        "throughput_qps": round(1000.0 / statistics.fmean(samples_ms), 1),
    }


def _environment() -> dict:
    import torch

    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": _git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "torch": _version("torch"),
        "torch_threads": torch.get_num_threads(),
        "sentence_transformers": _version("sentence-transformers"),
        "psutil": _version("psutil"),
        "tiktoken": _version("tiktoken"),
        "process_rss_baseline_mb": _rss_mb(),
    }


def _measure(fn, questions: list[str], passes: int) -> tuple[list[float], float]:
    for question in questions[:3]:
        fn(question)
    samples: list[float] = []
    started = time.perf_counter()
    for _ in range(passes):
        for question in questions:
            tick = time.perf_counter_ns()
            fn(question)
            samples.append((time.perf_counter_ns() - tick) / 1e6)
    wall = time.perf_counter() - started
    return samples, wall


def _count_tokens(encoder, text: str) -> int:
    return len(encoder.encode(text))


def _cost_report(
    queries,
    answers: list[dict],
    chunks: list,
    settings: Settings,
) -> dict:
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    answered = [row for row in answers if not row["refused"]]
    input_tokens = [
        _count_tokens(
            encoder,
            SYSTEM_PROMPT
            + "\n\nContext:\n"
            + build_context(row["contexts"])
            + f"\n\nQuestion: {row['question']}",
        )
        for row in answered
    ]
    output_tokens = [
        _count_tokens(encoder, row["reference_answer"]) for row in answered
    ]
    query_tokens = [_count_tokens(encoder, q.question) for q in queries]
    index_tokens = sum(_count_tokens(encoder, chunk.text) for chunk in chunks)

    avg_input = statistics.fmean(input_tokens) if input_tokens else 0.0
    avg_output = statistics.fmean(output_tokens) if output_tokens else 0.0
    avg_query_embedding = statistics.fmean(query_tokens) if query_tokens else 0.0

    prices = PRICE_SNAPSHOT
    llm_per_query = (
        avg_input * prices["gpt-4o-mini"]["input_per_1m_usd"]
        + avg_output * prices["gpt-4o-mini"]["output_per_1m_usd"]
    ) / 1_000_000
    query_embedding_per_query = (
        avg_query_embedding * prices["text-embedding-3-small"]["per_1m_usd"]
    ) / 1_000_000
    index_once = index_tokens * prices["text-embedding-3-small"]["per_1m_usd"] / 1_000_000

    return {
        "price_snapshot": prices,
        "token_counts": {
            "index_tokens": index_tokens,
            "avg_query_tokens": round(avg_query_embedding, 1),
            "avg_prompt_input_tokens": round(avg_input, 1),
            "avg_reference_output_tokens": round(avg_output, 1),
            "answered_queries": len(answered),
            "total_queries": len(queries),
        },
        "per_query_usd": {
            "local_extractive": 0.0,
            "local_retrieval_plus_gpt4o_mini": round(llm_per_query, 8),
            "openai_embeddings_plus_gpt4o_mini": round(
                llm_per_query + query_embedding_per_query, 8
            ),
        },
        "per_1k_queries_usd": {
            "local_extractive": 0.0,
            "local_retrieval_plus_gpt4o_mini": round(llm_per_query * 1000, 4),
            "openai_embeddings_plus_gpt4o_mini": round(
                (llm_per_query + query_embedding_per_query) * 1000, 4
            ),
        },
        "one_time_index_usd_openai_embeddings": round(index_once, 8),
        "assumptions": [
            "Output tokens are estimated from the curated reference answer, a lower bound for a conversational model.",
            "Refused queries make no chat call; their LLM cost is zero.",
            "Local embedding and extraction run on the measured CPU; their marginal API cost is zero.",
            "Prices are a configurable snapshot, not a live quote; verify before external use.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=5, help="timed passes over the query set")
    parser.add_argument("--split", default=None, help="dev | test | all (default: all)")
    args = parser.parse_args()

    PERF.mkdir(parents=True, exist_ok=True)
    COST.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    queries, eval_hash = load_eval_set(EVAL)
    if args.split and args.split != "all":
        queries = [q for q in queries if q.split == args.split]
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    environment = _environment()

    print("[measure] loading local embedder ...", flush=True)
    tick = time.perf_counter()
    embedder = make_embedder(settings)
    _ = embedder.dim
    embedding_model_load_ms = round((time.perf_counter() - tick) * 1000, 1)
    rss_after_embedder = _rss_mb()

    print("[measure] building corpus ...", flush=True)
    tick = time.perf_counter()
    corpus = build_corpus(load_documents_jsonl(CORPUS / "documents.jsonl"), embedder, settings)
    corpus_build_ms = round((time.perf_counter() - tick) * 1000, 1)

    tick = time.perf_counter()
    BM25Retriever(corpus.chunks)
    bm25_build_ms = round((time.perf_counter() - tick) * 1000, 1)
    rss_after_corpus = _rss_mb()

    print("[measure] loading reranker ...", flush=True)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model, device=settings.device)
    tick = time.perf_counter()
    reranker.rerank(queries[0].question, corpus.hybrid.retrieve(queries[0].question, 2), 2)
    reranker_load_ms = round((time.perf_counter() - tick) * 1000, 1)
    rss_after_reranker = _rss_mb()

    gate = RerankGate(
        reranker,
        min_score=settings.min_rerank_score,
        candidate_k=settings.rerank_candidate_k,
    )
    pipeline = RAGPipeline(
        corpus.hybrid,
        ExtractiveGenerator(),
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        gate=gate,
    )
    reorder = RerankedRetriever(
        base=corpus.hybrid,
        reranker=reranker,
        candidate_k=settings.rerank_candidate_k,
    )

    questions = [q.question for q in queries]
    systems = {
        "bm25": lambda q: corpus.bm25.retrieve(q, 10),
        "dense": lambda q: corpus.dense.retrieve(q, 10),
        "hybrid": lambda q: corpus.hybrid.retrieve(q, 10),
        "hybrid+rerank(reorder)": lambda q: reorder.retrieve(q, 10),
        "hybrid+rerank-gate": lambda q: gate.filter(
            q, corpus.hybrid.retrieve(q, settings.candidate_k)
        ),
        "pipeline(extractive)": lambda q: pipeline.answer_with_trace(q),
    }

    latency: dict[str, dict] = {}
    for name, fn in systems.items():
        print(f"[measure] {name} ({args.passes} passes) ...", flush=True)
        samples, wall = _measure(fn, questions, args.passes)
        summary = _latency_summary(samples)
        summary["wall_s"] = round(wall, 2)
        latency[name] = summary
        print(
            f"  p50={summary['p50_ms']}ms p95={summary['p95_ms']}ms "
            f"p99={summary['p99_ms']}ms qps={summary['throughput_qps']}",
            flush=True,
        )

    print("[measure] collecting pipeline outputs for cost ...", flush=True)
    rows = []
    peak_rss = _rss_mb()
    for query in queries:
        answer, trace = pipeline.answer_with_trace(query.question)
        peak_rss = max(peak_rss, _rss_mb())
        rows.append(
            {
                "query_id": query.id,
                "question": query.question,
                "answerable": query.answerable,
                "refused": answer.refused,
                "mode": answer.mode,
                "degraded": answer.degraded,
                "contexts": trace.contexts,
                "reference_answer": query.reference_answer,
            }
        )

    resource = {
        "embedding_model_load_ms": embedding_model_load_ms,
        "corpus_build_ms": corpus_build_ms,
        "bm25_build_ms": bm25_build_ms,
        "estimated_embedding_ms_per_chunk": round(
            max(0.0, corpus_build_ms - bm25_build_ms) / max(1, len(corpus.chunks)), 2
        ),
        "reranker_load_ms": reranker_load_ms,
        "rss_baseline_mb": environment["process_rss_baseline_mb"],
        "rss_after_embedder_mb": rss_after_embedder,
        "rss_after_corpus_mb": rss_after_corpus,
        "rss_after_reranker_mb": rss_after_reranker,
        "rss_peak_mb": peak_rss,
        "corpus_chunks": len(corpus.chunks),
        "corpus_documents": len(corpus.documents),
    }

    cost = _cost_report(queries, rows, corpus.chunks, settings)

    common = {
        "environment": environment,
        "dataset": {
            "queries": len(queries),
            "split": args.split or "all",
            "eval_hash": eval_hash,
        },
        "corpus": {
            "corpus_hash": manifest["corpus_hash"],
            "chunk_hash": manifest["chunk_hash"],
            "documents": manifest["total_documents"],
            "chunks": manifest["total_chunks"],
            "words": manifest["total_words"],
        },
        "settings": {
            "embedding_model": settings.local_embedding_model,
            "reranker_model": settings.reranker_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "candidate_k": settings.candidate_k,
            "rerank_candidate_k": settings.rerank_candidate_k,
            "min_rerank_score": settings.min_rerank_score,
        },
        "method": {
            "passes": args.passes,
            "samples_per_system": len(questions) * args.passes,
            "warmup_queries": min(3, len(questions)),
            "timing": "perf_counter_ns around one call, single process, single client",
            "percentile": "nearest-rank",
        },
    }

    perf_payload = {**common, "cold_start": resource, "latency": latency,
                    "resource": {"rss_peak_mb": peak_rss}}
    cost_payload = {**common, "cost": cost}

    (PERF / "results.json").write_text(
        json.dumps(perf_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (COST / "results.json").write_text(
        json.dumps(cost_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PERF / "report.md").write_text(_render_perf(perf_payload), encoding="utf-8")
    (COST / "report.md").write_text(_render_cost(cost_payload), encoding="utf-8")
    print("\nwrote evidence/performance/ and evidence/cost/", flush=True)


def _render_perf(payload: dict) -> str:
    env = payload["environment"]
    cold = payload["cold_start"]
    lines = [
        "# Performance evidence",
        "",
        "> Generated by `python scripts/measure.py`. Single process, CPU only,",
        "> single client. Latency is one call measured with `perf_counter_ns`;",
        "> percentiles use the nearest-rank method.",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| generated_at | `{env['generated_at']}` |",
        f"| git_commit | `{env['git_commit']}` (dirty={env['git_dirty']}) |",
        f"| python | `{env['python']}` |",
        f"| platform | `{env['platform']}` |",
        f"| processor | `{env['processor']}` |",
        f"| CPU cores | {env['cpu_count_physical']} physical / {env['cpu_count_logical']} logical |",
        f"| RAM | {env['ram_total_gb']} GB |",
        f"| torch / threads | `{env['torch']}` / {env['torch_threads']} |",
        f"| sentence-transformers | `{env['sentence_transformers']}` |",
        f"| corpus | {payload['corpus']['documents']} sections / {payload['corpus']['chunks']} chunks / {payload['corpus']['words']} words |",
        f"| corpus_hash | `{payload['corpus']['corpus_hash']}` |",
        f"| eval | {payload['dataset']['queries']} queries, split=`{payload['dataset']['split']}` |",
        f"| samples per system | {payload['method']['samples_per_system']} |",
        "",
        "## Cold start",
        "",
        "| Stage | Time | RSS after |",
        "|---|---:|---:|",
        f"| embedding model load | {cold['embedding_model_load_ms'] / 1000:.2f} s | {cold['rss_after_embedder_mb']} MB |",
        f"| corpus build (embed {cold['corpus_chunks']} chunks + index) | {cold['corpus_build_ms'] / 1000:.2f} s | {cold['rss_after_corpus_mb']} MB |",
        f"| BM25 index only | {cold['bm25_build_ms'] / 1000:.2f} s | — |",
        f"| estimated embedding cost | {cold['estimated_embedding_ms_per_chunk']} ms/chunk | — |",
        f"| reranker load | {cold['reranker_load_ms'] / 1000:.2f} s | {cold['rss_after_reranker_mb']} MB |",
        "",
        "Cold start is dominated by loading model weights. The HTTP service pays",
        "this once at startup, not per request.",
        "",
        "## Latency distribution",
        "",
        "| System | n | p50 | p90 | p95 | p99 | mean | max | sequential QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["latency"].items():
        lines.append(
            f"| {name} | {row['n']} | {row['p50_ms']} ms | {row['p90_ms']} ms | "
            f"{row['p95_ms']} ms | {row['p99_ms']} ms | {row['mean_ms']} ms | "
            f"{row['max_ms']} ms | {row['throughput_qps']} |"
        )
    lines += [
        "",
        "`pipeline(extractive)` is the complete no-API-key product path: hybrid",
        "retrieval over 50 candidates, reranker gate over 15, then a cited extractive",
        "answer. `hybrid+rerank-gate` isolates the retrieval + gate cost without",
        "generation. `hybrid+rerank(reorder)` is the rejected reordering variant.",
        "",
        "## Memory",
        "",
        "| Stage | RSS |",
        "|---|---:|",
        f"| Python imports only | {cold['rss_baseline_mb']} MB |",
        f"| after embedding model | {cold['rss_after_embedder_mb']} MB |",
        f"| after corpus + BM25 index | {cold['rss_after_corpus_mb']} MB |",
        f"| after reranker | {cold['rss_after_reranker_mb']} MB |",
        f"| peak during measurement | {cold['rss_peak_mb']} MB |",
        "",
        "RSS is process-level and includes the Python runtime, PyTorch, and model",
        "weights. It is not a measure of incremental corpus size.",
        "",
        "## What this does not measure",
        "",
        "- Concurrency or queueing: the numbers are single-client sequential.",
        "- GPU latency: the shipped path targets CPU.",
        "- Network/LLM time: no API key is used in this benchmark.",
        "- Cold HTTP request overhead: measured at the pipeline boundary, not over a socket.",
        "",
    ]
    return "\n".join(lines)


def _render_cost(payload: dict) -> str:
    cost = payload["cost"]
    tokens = cost["token_counts"]
    prices = cost["price_snapshot"]
    lines = [
        "# Cost evidence",
        "",
        "> Generated by `python scripts/measure.py` from the same run as the",
        "> performance report. Prices are configurable constants, not a live quote.",
        "",
        "## Token accounting",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| corpus chunks | {payload['corpus']['chunks']} |",
        f"| index tokens (cl100k_base) | {tokens['index_tokens']:,} |",
        f"| average query tokens | {tokens['avg_query_tokens']} |",
        f"| average prompt input tokens | {tokens['avg_prompt_input_tokens']} |",
        f"| average reference output tokens | {tokens['avg_reference_output_tokens']} |",
        f"| answered queries in cost sample | {tokens['answered_queries']} / {tokens['total_queries']} |",
        "",
        "Prompt tokens are counted from the exact system prompt, numbered context",
        "block, and question. Output tokens use the curated reference answer as a",
        "conservative proxy; a chat model may be more verbose.",
        "",
        "## Price snapshot",
        "",
        f"Snapshot `{prices['as_of']}` from <{prices['source']}>. Verify before",
        "quoting externally; override with the `PRICE_*` environment variables.",
        "",
        "| Item | USD per 1M tokens |",
        "|---|---:|",
        f"| gpt-4o-mini input | {prices['gpt-4o-mini']['input_per_1m_usd']:.2f} |",
        f"| gpt-4o-mini output | {prices['gpt-4o-mini']['output_per_1m_usd']:.2f} |",
        f"| text-embedding-3-small | {prices['text-embedding-3-small']['per_1m_usd']:.2f} |",
        "",
        "## Cost by configuration",
        "",
        "| Configuration | Per query | Per 1,000 | API marginal cost |",
        "|---|---:|---:|---|",
        f"| local retrieval + extractive | $0 | $0 | none (CPU time only) |",
        f"| local retrieval + gpt-4o-mini | ${cost['per_query_usd']['local_retrieval_plus_gpt4o_mini']:.6f} | ${cost['per_1k_queries_usd']['local_retrieval_plus_gpt4o_mini']:.2f} | chat only |",
        f"| OpenAI embeddings + gpt-4o-mini | ${cost['per_query_usd']['openai_embeddings_plus_gpt4o_mini']:.6f} | ${cost['per_1k_queries_usd']['openai_embeddings_plus_gpt4o_mini']:.2f} | query embedding + chat |",
        "",
        f"One-time indexing with text-embedding-3-small: **${cost['one_time_index_usd_openai_embeddings']:.6f}** for {tokens['index_tokens']:,} tokens.",
        "",
        "The local configuration is the shipped demo: E5-small and the cross-encoder",
        "run on CPU, so there is no per-query API bill. Its real cost is latency and",
        "host memory, reported in the performance evidence.",
        "",
        "## Assumptions and scope",
        "",
    ]
    lines += [f"- {item}" for item in cost["assumptions"]]
    lines += [
        "- No retries are billed in these figures; a retry adds one input request.",
        "- Token counts exclude provider-side prompt caching and tool overhead.",
        "- Self-hosted electricity and hardware amortisation are not included.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

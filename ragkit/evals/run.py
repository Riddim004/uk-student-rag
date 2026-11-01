"""Shared evaluation runner used by the CLI and the evidence scripts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..embeddings import make_embedder
from ..ingest import build_corpus, load_documents_jsonl
from ..relevance import Gate, GateConfig, RelevanceGate, RerankGate
from ..retrievers import CrossEncoderReranker, RerankedRetriever
from ..retrievers.base import Retriever
from ..retrievers.rerank import Reranker
from .dataset import EvalQuery, load_eval_set
from .runner import RetrievalReport, compare_to_markdown, evaluate_retriever

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "corpus" / "documents.jsonl"
DEFAULT_EVAL_DIR = ROOT / "evidence" / "eval"


@dataclass
class EvalRun:
    reports: list[RetrievalReport]
    queries: list[EvalQuery]
    eval_hash: str
    corpus_manifest: dict
    settings: Settings
    k: int
    split: str | None = None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return True


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _load_manifest(path: Path) -> dict:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_systems(
    corpus,
    settings: Settings,
    include_rerank: bool = False,
    reranker: Reranker | None = None,
) -> dict[str, Retriever]:
    systems: dict[str, Retriever] = {
        "bm25": corpus.bm25,
        "dense": corpus.dense,
        "hybrid": corpus.hybrid,
    }
    if include_rerank:
        if reranker is None:
            reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.device,
            )
        systems["hybrid+rerank"] = RerankedRetriever(
            base=corpus.hybrid,
            reranker=reranker,
            candidate_k=settings.rerank_candidate_k,
        )
        # Same hybrid ranking, but the cross-encoder is used only to decide
        # whether to answer. This preserves MRR while adding refusal.
        systems["hybrid+rerank-gate"] = corpus.hybrid
    return systems


def run_evaluations(
    data_path: str | Path = DEFAULT_DATA,
    eval_dir: str | Path = DEFAULT_EVAL_DIR,
    settings: Settings | None = None,
    k: int = 10,
    include_rerank: bool = False,
    splits: tuple[str | None, ...] = (None,),
    verbose: bool = False,
) -> list[EvalRun]:
    """Build the corpus and models once, then evaluate each split."""
    import time

    settings = settings or Settings()
    data_path = Path(data_path)
    eval_dir = Path(eval_dir)
    queries, eval_hash = load_eval_set(eval_dir)
    documents = load_documents_jsonl(data_path)

    if verbose:
        print(f"[eval] building corpus from {len(documents)} sections ...", flush=True)
    started = time.perf_counter()
    embedder = make_embedder(settings)
    corpus = build_corpus(documents, embedder, settings)
    reranker = (
        CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=settings.device,
        )
        if include_rerank
        else None
    )
    systems = build_systems(
        corpus,
        settings,
        include_rerank=include_rerank,
        reranker=reranker,
    )
    gates: dict[str, Gate | None] = {
        name: RelevanceGate.from_settings(settings) for name in systems
    }
    if include_rerank and reranker is not None:
        gates["hybrid+rerank-gate"] = RerankGate(
            reranker,
            min_score=settings.min_rerank_score,
            candidate_k=settings.rerank_candidate_k,
        )
    if verbose:
        print(
            f"[eval] corpus ready in {time.perf_counter() - started:.1f}s "
            f"({len(corpus.chunks)} chunks), systems={list(systems)}",
            flush=True,
        )

    runs: list[EvalRun] = []
    for split in splits:
        selected = queries
        if split is not None:
            selected = [query for query in queries if query.split == split]
            if not selected:
                raise ValueError(f"no queries for split={split!r}")
        if verbose:
            print(f"[eval] split={split or 'all'} ({len(selected)} queries)", flush=True)
        reports = []
        for name, retriever in systems.items():
            system_started = time.perf_counter()
            if verbose:
                print(f"[eval]   {name} ...", flush=True)
            reports.append(
                evaluate_retriever(
                    name, retriever, selected, k=k, gate=gates.get(name)
                )
            )
            if verbose:
                print(
                    f"[eval]   {name} done in "
                    f"{time.perf_counter() - system_started:.1f}s",
                    flush=True,
                )
        runs.append(
            EvalRun(
                reports=reports,
                queries=selected,
                eval_hash=eval_hash,
                corpus_manifest=_load_manifest(data_path),
                settings=settings,
                k=k,
                split=split,
            )
        )
    return runs


def run_evaluation(
    data_path: str | Path = DEFAULT_DATA,
    eval_dir: str | Path = DEFAULT_EVAL_DIR,
    settings: Settings | None = None,
    k: int = 10,
    include_rerank: bool = False,
    split: str | None = None,
    verbose: bool = False,
) -> EvalRun:
    return run_evaluations(
        data_path=data_path,
        eval_dir=eval_dir,
        settings=settings,
        k=k,
        include_rerank=include_rerank,
        splits=(split,),
        verbose=verbose,
    )[0]


def _dataset_summary(queries: list[EvalQuery]) -> dict:
    labels = [
        len(q.relevant_chunk_ids) for q in queries if q.answerable
    ]
    return {
        "n_queries": len(queries),
        "n_answerable": sum(1 for q in queries if q.answerable),
        "n_unanswerable": sum(1 for q in queries if not q.answerable),
        "categories": dict(sorted(Counter(q.category for q in queries).items())),
        "languages": dict(sorted(Counter(q.language for q in queries).items())),
        "splits": dict(sorted(Counter(q.split for q in queries).items())),
        "labels": {
            "min_chunks_per_answerable_query": min(labels) if labels else 0,
            "max_chunks_per_answerable_query": max(labels) if labels else 0,
            "mean_chunks_per_answerable_query": (
                round(sum(labels) / len(labels), 2) if labels else 0.0
            ),
            "queries_with_multiple_chunks": sum(1 for n in labels if n > 1),
        },
    }


def metadata(run: EvalRun, include_rerank: bool) -> dict:
    settings = run.settings
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "split": run.split or "all",
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "corpus_hash": run.corpus_manifest.get("corpus_hash", "unknown"),
        "chunk_hash": run.corpus_manifest.get("chunk_hash", "unknown"),
        "eval_hash": run.eval_hash,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "embedding_backend": settings.embedding_backend,
        "embedding_model": (
            settings.local_embedding_model
            if settings.embedding_backend == "local"
            else settings.embedding_model
        ),
        "sentence_transformers": _version("sentence-transformers"),
        "torch": _version("torch"),
        "k": run.k,
        "candidate_k": settings.candidate_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "rrf_k": settings.rrf_k,
        "sparse_weight": settings.sparse_weight,
        "dense_weight": settings.dense_weight,
        "gate": asdict(
            GateConfig(
                min_dense_score=settings.min_dense_score,
                min_sparse_score=settings.min_sparse_score,
                min_rerank_score=settings.min_rerank_score,
            )
        ),
        "rerank": {
            "enabled": include_rerank,
            "model": settings.reranker_model if include_rerank else None,
            "candidate_k": settings.rerank_candidate_k if include_rerank else None,
            "min_rerank_score": settings.min_rerank_score if include_rerank else None,
        },
        "corpus": {
            "documents": run.corpus_manifest.get("total_documents"),
            "words": run.corpus_manifest.get("total_words"),
            "chunks": run.corpus_manifest.get("total_chunks"),
            "sources": len(run.corpus_manifest.get("sources", [])),
        },
    }


def render_report(run: EvalRun, include_rerank: bool) -> str:
    meta = metadata(run, include_rerank)
    summary = _dataset_summary(run.queries)
    lines = [
        f"# Retrieval evaluation report — {meta['split']} split",
        "",
        "> Generated by `python scripts/run_eval.py`. Every number is tied to a",
        "> corpus hash, eval-set hash and commit; re-run the command to reproduce it.",
        "",
        "## Experiment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in [
        "generated_at",
        "split",
        "git_commit",
        "git_dirty",
        "corpus_hash",
        "chunk_hash",
        "eval_hash",
        "python",
        "platform",
        "embedding_backend",
        "embedding_model",
        "k",
        "candidate_k",
        "chunk_size",
        "chunk_overlap",
        "rrf_k",
        "gate",
        "rerank",
    ]:
        lines.append(f"| {key} | `{meta[key]}` |")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| queries | {summary['n_queries']} |")
    lines.append(f"| answerable | {summary['n_answerable']} |")
    lines.append(f"| unanswerable | {summary['n_unanswerable']} |")
    lines.append(f"| categories | `{summary['categories']}` |")
    lines.append(f"| languages | `{summary['languages']}` |")
    lines.append(f"| splits | `{summary['splits']}` |")
    lines.append(f"| chunk labels | `{summary['labels']}` |")
    lines.append("")
    lines.append(compare_to_markdown(run.reports))
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "Retrieval metrics are computed over answerable queries only. "
        "Unanswerable queries are used for the refusal gate, not for recall. "
        "Confidence intervals are 95% bootstrap intervals over queries "
        "(2,000 resamples, seed 42)."
    )
    lines.append("")
    return "\n".join(lines)


def results_payload(run: EvalRun, include_rerank: bool) -> dict:
    return {
        "metadata": metadata(run, include_rerank),
        "dataset": _dataset_summary(run.queries),
        "split": run.split or "all",
        "systems": {
            report.name: {
                "metrics": report.metrics,
                "ci": {name: list(ci) for name, ci in report.ci.items()},
                "per_category": report.per_category,
                "error_counts": report.error_counts,
                "refusal": report.refusal,
                "per_query": [asdict(row) for row in report.per_query],
            }
            for report in run.reports
        },
    }

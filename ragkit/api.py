"""FastAPI surface for the evidence-board UI.

The API is intentionally thin: it serializes the pipeline trace so every answer
can be inspected, and streams real stage events (retrieve → gate → generate)
instead of faking a typing animation.
"""

from __future__ import annotations

import json
import math
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .errors import ModelError
from .generator import body_text, clean_quote
from .pipeline import PipelineTrace
from .runtime import ROOT, Runtime, build_runtime
from .types import Answer, ScoredChunk


def _finite(value: float) -> float | None:
    return round(float(value), 4) if math.isfinite(float(value)) else None


def _scored_payload(sc: ScoredChunk, include_text: bool = False) -> dict:
    chunk = sc.chunk
    payload = {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "title": chunk.title or chunk.doc_id,
        "source": chunk.source,
        "section": chunk.section_path[-1] if chunk.section_path else chunk.section_id,
        "section_path": list(chunk.section_path),
        "updated_at": chunk.updated_at,
        "score": _finite(sc.score),
        "components": {key: _finite(value) for key, value in sc.components.items()},
        "snippet": " ".join(clean_quote(body_text(chunk.text)).split())[:280],
    }
    if include_text:
        payload["text"] = clean_quote(body_text(chunk.text))
    return payload


def _answer_payload(
    question: str,
    answer: Answer,
    trace: PipelineTrace,
    runtime: Runtime,
) -> dict:
    trail = trace.candidates[:8]
    return {
        "question": question,
        "answer": answer.text,
        "refused": answer.refused,
        "refused_reason": answer.refused_reason,
        "answer_mode": answer.mode,
        "degraded": answer.degraded,
        "degraded_reason": answer.degraded_reason,
        "citations": [asdict(citation) for citation in answer.citations],
        "contexts": [_scored_payload(sc, include_text=True) for sc in trace.contexts],
        "trail": [_scored_payload(sc) for sc in trail],
        "counts": {
            "candidates": len(trace.candidates),
            "contexts": len(trace.contexts),
        },
        "gate": {
            "mode": runtime.gate_mode,
            "accepted": trace.gate_accepted,
            "best_rerank": _finite(trace.gate_best) if trace.gate_best is not None else None,
            "threshold": trace.gate_threshold,
            "candidate_k": trace.gate_candidate_k,
        },
        "timings_ms": trace.timings_ms,
        "product_mode": runtime.mode,
        "corpus": {
            "corpus_hash": runtime.corpus_manifest.get("corpus_hash"),
            "chunk_hash": runtime.corpus_manifest.get("chunk_hash"),
            "chunks": runtime.corpus_manifest.get("total_chunks"),
        },
    }


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)


def _runtime_payload(runtime: Runtime) -> dict:
    settings = runtime.settings
    return {
        "status": "ok",
        "started_at": runtime.started_at,
        "startup_ms": runtime.startup_ms,
        "product_mode": runtime.mode,
        "gate_mode": runtime.gate_mode,
        "warnings": runtime.warnings,
        "corpus": runtime.corpus_manifest,
        "eval": runtime.eval_summary,
        "settings": {
            "embedding_model": settings.local_embedding_model
            if settings.embedding_backend == "local"
            else settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "top_k": settings.top_k,
            "candidate_k": settings.candidate_k,
            "rerank_candidate_k": settings.rerank_candidate_k,
            "min_rerank_score": settings.min_rerank_score,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "concurrency": "serialized by an in-process lock",
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    runtime: Runtime | None = None,
    web_dir: str | Path | None = None,
) -> FastAPI:
    web_dir = Path(web_dir) if web_dir is not None else ROOT / "web"
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or build_runtime()
        yield

    app = FastAPI(
        title="UK Student Visa Evidence Desk",
        description="Hybrid retrieval, reranker gate, cited answers, and the evidence behind them.",
        version="2.0.0",
        lifespan=lifespan,
    )

    def get_runtime() -> Runtime:
        return app.state.runtime

    @app.get("/api/health")
    def health() -> dict:
        return _runtime_payload(get_runtime())

    @app.get("/api/examples")
    def examples() -> dict:
        return {"examples": get_runtime().examples}

    @app.get("/api/eval")
    def evaluation() -> dict:
        runtime_ = get_runtime()
        return {"eval": runtime_.eval_summary, "corpus": runtime_.corpus_manifest}

    def _ask(question: str) -> dict:
        runtime_ = get_runtime()
        with lock:
            try:
                answer, trace = runtime_.ask(question)
            except ModelError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _answer_payload(question, answer, trace, runtime_)

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        return _ask(request.question.strip())

    @app.get("/api/ask/stream")
    def ask_stream(
        q: str = Query(min_length=2, max_length=500),
    ) -> StreamingResponse:
        runtime_ = get_runtime()
        question = q.strip()
        events: queue.Queue = queue.Queue()

        def on_stage(stage: str) -> None:
            events.put(("stage", {"stage": stage}))

        def work() -> None:
            try:
                with lock:
                    answer, trace = runtime_.ask(question, on_stage=on_stage)
                events.put(("answer", _answer_payload(question, answer, trace, runtime_)))
            except Exception as exc:  # surface the failure, never fake an answer
                events.put(
                    (
                        "error",
                        {
                            "code": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                )
            finally:
                events.put(None)

        def stream():
            yield _sse("open", {"question": question})
            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            while True:
                try:
                    item = events.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                event, data = item
                yield _sse(event, data)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app

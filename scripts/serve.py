"""Serve the evidence desk: python scripts/serve.py [--demo] [--port 8000]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from ragkit.api import create_app  # noqa: E402
from ragkit.runtime import build_runtime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--demo", action="store_true", help="force extractive mode")
    parser.add_argument("--no-rerank", action="store_true", help="use the scalar gate")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.reload:
        uvicorn.run(
            "ragkit.api:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    runtime = build_runtime(
        use_llm=False if args.demo else None,
        use_rerank=not args.no_rerank,
    )
    print(
        f"mode={runtime.mode} gate={runtime.gate_mode} "
        f"chunks={len(runtime.corpus.chunks)} startup={runtime.startup_ms / 1000:.1f}s",
        flush=True,
    )
    for warning in runtime.warnings:
        print(f"warning: {warning}", flush=True)
    print(f"http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

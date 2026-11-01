# ADR-007 — FastAPI + a no-build evidence board, with SSE stage events

**Status:** accepted · 2026-11-01

## Context

The retrieval work only becomes a product if a reader can check a citation
without reading Python. The UI must run on a CPU box with no Node toolchain,
work offline, and expose the pipeline's real decisions instead of a generic chat
bubble.

## Decision

- FastAPI serves `/api/health`, `/api/examples`, `/api/ask`,
  `/api/ask/stream`, `/api/eval`, and the static UI.
- The UI is handwritten HTML/CSS/JS (no build step, no external fonts/assets).
- The answer stream emits the real stages (`retrieve`, `gate`, `generate`) from
  a worker thread via SSE; it does not fake a typing animation.
- The answer and the evidence share one screen: numbered citations, source
  cards, dense/sparse/rerank meters, gate score vs threshold, stage timings.
- Bilingual UI labels; the answer language follows the question in LLM mode.
- Demo mode is the same pipeline with the extractive generator; no key required.

## Alternatives

| Alternative | Benefit | Cost |
|---|---|---|
| Streamlit/Gradio | fastest to prototype | hard to show a trace and custom evidence layout |
| React + Vite | component ecosystem | Node build, lockfile churn, overkill for one screen |
| Next.js / hosted SaaS | easy deploy | keys and data leave the machine; more moving parts |
| Non-streaming JSON | simplest | hides where the 1 s goes; worse perceived latency |

## Consequences

- One process: `python scripts/serve.py --demo`; startup ~19–23 s cold (models),
  then ~0.9–1.1 s per query on this CPU.
- The API serializes the gate decision and retrieval trail, so the UI can explain
  refusals from the closest passages even when contexts are empty.
- A process-wide lock serializes generation; concurrency is not claimed.
  Benchmark numbers are single-client and labelled as such.
- Static files are served from `web/`; screenshots live in `evidence/product/`.

## Would change my mind

If multiple concurrent users matter, add a proper worker queue and per-request
model session; if the UI needs collaborative annotation, move to a component
framework. Neither is justified for an evidence-first portfolio product.

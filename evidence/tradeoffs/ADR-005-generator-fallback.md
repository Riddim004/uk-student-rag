# ADR-005 — Cited LLM answers with a visible extractive fallback

**Status:** accepted · 2026-11-01

## Context

The desk must be demonstrable without an API key, and it must not turn a backend
outage into a fake "I don't know". A refusal is a product state; an outage is an
error state. Conflating them was Incident 001/006.

## Decision

Two generators share the same interface:

1. `LLMGenerator` — grounded prompt, inline `[n]` citations, exact refusal text,
   answer in the question's language.
2. `ExtractiveGenerator` — copies the best-matching sentences from the accepted
   passages and attaches the same citation markers. Deterministic, no key.

`FallbackGenerator` tries the LLM; on `GenerationError` it returns the
extractive answer marked `degraded=True` with the failure reason. If no key is
configured, the service starts in extractive mode and says so.

## Alternatives

| Alternative | Benefit | Cost |
|---|---|---|
| LLM only | best prose | no offline demo; outage becomes 500 |
| Template answers | safe | cannot handle paraphrases/multi-hop |
| Silent fallback | seamless | hides degradation; users cannot tell quote from composition |
| Fail closed on outage | simple | unhelpful when retrieval is perfectly good |

## Consequences

- The demo is honest: the UI/CLI labels `extractive` and `degraded`.
- Extractive answers are verbatim English source sentences; Chinese UI labels the
  source language. LLM mode answers in the question language.
- Prompt input averages ~1,806 tokens for 5 passages; output is bounded by
  `MAX_OUTPUT_TOKENS=400`. Cost is in `evidence/cost/report.md`.
- Retries: 3 attempts, exponential backoff, transient-only (timeouts, 429, 5xx).
  Auth/validation failures fail fast.
- Tests: generation error ≠ refusal; empty completion is retried then errors;
  fallback is cited; out-of-range markers are dropped.

## Would change my mind

If an LLM-free deployment is required, make extractive the primary mode and
invest in sentence compression. If answer faithfulness becomes measurable, add a
citation-support check as a second gate on generated claims.

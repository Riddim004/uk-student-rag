# ADR-004 — Use the cross-encoder as a gate, not a re-ranker

**Status:** accepted · 2026-11-01

## Context

Hybrid retrieval always returns 50 chunks, even for "Can I bring my dog?".
Something must decide whether to answer. Scalar dense/BM25 thresholds were
calibrated first and failed: on dev they either refused nothing or over-refused
a third of answerable queries.

A cross-encoder can read each (query, passage) pair and produce a calibrated-ish
relevance logit. Running it over all 50 candidates and reordering is the obvious
use, but it is not free (~0.9 s per query on this CPU).

## Decision

Run the cross-encoder over the top 15 hybrid candidates. Use only the **best
score** to make an answer/refuse decision at `-3.0`; keep the hybrid order for
the passages sent to the generator.

## Alternatives

| Alternative | Dev result | Test result | Cost |
|---|---|---|---|
| Dense threshold 0.875 | 100% unanswerable refused, 38% answerable refused | — | cheap, unusable |
| BM25 threshold 12.0 | 100% / 33% | — | cheap, unusable |
| Rerank + reorder | recall 0.948, MRR 0.909, nDCG 0.867 | recall 0.988, MRR 0.771, nDCG 0.808 | ~0.9 s/query |
| **Rerank gate, hybrid order** | recall 0.976, MRR 0.944, gate acc 0.960 | recall 0.972, MRR 0.845, gate acc 0.920 | ~0.9 s/query |

Reordering raised test recall but *lowered* MRR and nDCG: the labelled first hit
was pushed down by near-duplicate or unlabelled-but-relevant passages. The gate
keeps the retrieval metrics identical to hybrid while adding refusal.

## Consequences

- End-to-end gate cost ~0.90 s p50; the full extractive pipeline is ~1.0 s p50. This is the main
  latency item and is reported, not hidden.
- Two unanswerable test questions still pass (`q40`, `q42`): they are semantically
  close to the corpus, and one scalar cannot separate them. An answerability
  verifier is the next experiment.
- The gate must be calibrated on dev; the test split is only a check. Threshold
  `-3.0` is the conservative end of the dev plateau (margin 0.1).
- Evidence: `evidence/eval/gate_calibration.md`, Incident 003, `evidence/eval/report.md`.

## Would change my mind

If answer-level evaluation shows the reranked order produces better answers
despite worse MRR, revisit the order. If a small NLI verifier catches q40/q42
without over-refusing, use it *in addition to* the score gate.

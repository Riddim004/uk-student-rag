# ADR-003 — Hybrid retrieval with reciprocal rank fusion

**Status:** accepted · 2026-11-01

## Context

Lexical and dense retrieval fail in different ways. BM25 wins on exact fees,
dates, and quoted policy language; dense wins on paraphrases and cross-lingual
questions. The product needs one ranked list and an explainable score.

## Decision

Run BM25 and dense retrieval independently, fuse with Reciprocal Rank Fusion
(`k=60`, equal channel weights), and keep 50 candidates before gating. Store
`sparse`, `dense`, `sparse_rank`, `dense_rank`, and `rrf` on every candidate.

RRF is used for *ranking only*. It is never interpreted as a relevance
probability; a positive RRF score does not mean the corpus can answer.

## Alternatives

| Alternative | Benefit | Cost |
|---|---|---|
| Dense only | simplest | loses exact-number wins; lower recall on fee/date queries |
| BM25 only | fast, no model | Chinese recall 0.25; poor paraphrase |
| Weighted score sum | uses magnitudes | BM25 and cosine are not calibrated; one dominates |
| Learned fusion / LTR | best possible ranking | labels are too few (50 queries); overfits |

## Consequences

Test split (k=10): BM25 recall 0.770, dense 0.901, hybrid 0.972; hit@10 1.000.
Hybrid adds ~1 ms over dense because BM25 is sub-millisecond.
The component scores make the gate and the UI possible without re-running
retrieval.

## Would change my mind

With hundreds of labelled queries and per-query features, a small learning-to-rank
model could beat RRF. With 50 queries it would mostly memorise the dev split.

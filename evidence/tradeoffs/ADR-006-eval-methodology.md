# ADR-006 — Evaluation: 50 chunk-labelled queries, dev/test, explicit refusals

**Status:** accepted · 2026-11-01

## Context

The first eval was structurally unable to fail: 6 documents, 7 chunks, one
relevant document per query, `precision@k` divided by returned items, no split
and no unanswerable set. It reported recall@5 = 1.000. A metric can be correct
and still be meaningless.

## Decision

- 74 sections / 167 chunks from a frozen gov.uk snapshot.
- 50 queries: 10 exact, 10 paraphrase, 8 numeric, 6 multi-hop, 8 Chinese,
  8 unanswerable. 42 answerable, 8 unanswerable.
- **Chunk-level** labels: 1–4 relevant chunks per answerable query (mean 1.74);
  multiple chunks for 23 queries.
- Stratified 25/25 dev/test split, fixed before calibration.
- Gate thresholds selected on dev only; test is a held-out check.
- Metrics: recall@k, precision@k (denominator = k), nDCG@k, MRR, hit@k, MAP,
  section recall, bootstrap 95% CIs, per-category recall.
- Error taxonomy: hit, partial_hit, ranking_error, retrieval_miss, over_refusal,
  correct_refusal, unanswerable_answered, unanswerable_skipped.
- Every report is tied to corpus hash, chunk hash, eval hash, commit, dirty flag,
  Python/platform, models and settings.

## Alternatives

| Alternative | Benefit | Cost |
|---|---|---|
| Document-level labels | cheap | cannot tell precise citations from lucky documents |
| LLM-as-judge labels | scale | expensive, noisy, needs its own validation |
| Random split | simple | leaks near-duplicate categories across dev/test |
| Tune on test | looks better | dishonest; no held-out evidence |
| Accuracy only | simple | hides refusal vs retrieval trade-off |

## Consequences

- Test has only 4 unanswerable queries, so gate CIs are wide; treat gate accuracy
  as indicative and report intervals.
- q25/q49 reveal near-duplicate overlap chunks; labels choose the best chunk but
  ranking errors remain.
- The test split was not used to choose `-3.0`; changing that invalidates the
  held-out claim.
- Evidence: `evidence/eval/dataset_card.md`, `report.md`, `gate_calibration.md`,
  `errors.md`, `tests/test_artifacts.py`.

## Would change my mind

Add ~30 human-rated answer-level questions and an NLI/LLM faithfulness check.
Keep this retrieval set as the regression floor.

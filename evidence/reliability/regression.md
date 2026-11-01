# Regression strategy

The project has two kinds of checks:

1. **Deterministic unit/integration tests** — run in CI on every push, no
   network, no model downloads, ~0.2 s.
2. **Model-dependent evaluation** — run manually (or on a GPU runner) because it
   downloads E5 + a cross-encoder and takes several minutes on CPU. The resulting
   artifacts are committed and checked for internal consistency in CI.

## Deterministic suite

```bash
python -m pytest -q
# 78 passed
```

| File | Tests | Protects |
|---|---:|---|
| `tests/test_reliability.py` | 24 | retry budget/backoff, transient classification, generation failure vs refusal, degraded fallback, corpus validation, embedder output validation, manifest hashes |
| `tests/test_eval_and_gate.py` | 20 | CJK tokenization, precision/nDCG formulas, gate cannot be fooled by RRF, reranker gate/annotation, chunk headers, eval dataset schema, pipeline refusal/acceptance |
| `tests/test_api.py` | 6 | health payload, cited answer JSON, refusal trail, SSE stage events, validation, examples |
| `tests/test_chunking_and_metrics.py` | 9 | section-aware chunking, overlap, recall/precision/MRR/MAP/nDCG |
| `tests/test_pipeline.py` | 8 | prompt construction, citation parsing, refusal text, empty context |
| `tests/test_artifacts.py` | 6 | report hashes vs corpus/eval files, dataset stats, label → chunk referential integrity, quality floors, threshold/report agreement, failure coverage |
| `tests/test_retrievers.py` | 5 | BM25 behaviour, RRF fusion and top-k |

High-value regressions:

- `test_gate_cannot_be_fooled_by_positive_rrf_scores` — the original refusal bug.
- `test_pipeline_refuses_when_gate_rejects_retrieval_miss` — end-to-end refusal,
  not just a gate unit test.
- `test_retry_does_not_retry_non_transient_error` — auth failures must not burn
  the retry budget.
- `test_generation_error_is_not_a_refusal` — an outage must not masquerade as
  "no information".
- `test_fallback_degrades_to_grounded_extractive_answer` — every degraded
  answer is still cited.
- `test_bundled_corpus_matches_manifest_hashes` — provenance includes Windows.
- `test_all_labelled_chunks_exist_in_the_corpus` — eval labels cannot drift from
  the snapshot.
- `test_reported_quality_is_above_the_shipped_regression_floor` — the committed
  evidence cannot silently fall below recall 0.90, hit 0.95, gate accuracy 0.90,
  or over-refusal 0.05.

The last test is deliberately a **floor, not a target**. It catches a report
that was regenerated after a quality collapse; it does not claim the numbers are
good.

## Model-dependent evaluation

```bash
# Full dev+test report (5 systems, CIs, error taxonomy)
python scripts/run_eval.py --rerank

# Gate calibration (dev selects, test is frozen)
python scripts/calibrate_gate.py --rerank
```

Both commands write into `evidence/eval/`. The report records corpus hash,
chunk hash, eval hash, git commit, dirty-tree flag, Python/platform, model
names, and all retrieval settings. A result without those fields is not
accepted.

### Frozen floors

| Metric | Dev | Test | Floor |
|---|---:|---:|---:|
| hybrid recall@10 | 0.976 | 0.972 | 0.90 |
| hybrid hit@10 | 1.000 | 1.000 | 0.95 |
| hybrid nDCG@10 | 0.897 | 0.820 | — |
| reranker gate accuracy | 0.960 | 0.920 | 0.90 |
| answerable over-refusal | 0.000 | 0.000 | ≤ 0.05 |
| unanswerable refused (dev) | 0.750 | — | ≥ 0.70 |

Thresholds were selected on dev only. The test column is a held-out check and
must not be used to re-tune `min_rerank_score`.

## CI gates

`.github/workflows/ci.yml` runs on Python 3.12:

1. `python -m compileall -q ragkit scripts`
2. `python -m pytest -q`
3. A data/eval artifact check (covered by `tests/test_artifacts.py`)
4. A no-network smoke run of the CLI in deterministic mode:
   `python -m ragkit "How many hours can I work?" --retriever hybrid --no-llm`

CI intentionally does **not** download sentence-transformers or torch. The
local-model path is exercised by the manual eval command above. This is a
deliberate trade-off: a 2 GB dependency on every push buys little beyond what
the deterministic tests cover, while the manual eval pins the real numbers.

## Updating evidence

When code or data changes:

1. Rebuild data: `python scripts/build_data.py --offline` (or without
   `--offline` to refresh gov.uk).
2. Re-run eval and calibration with `--rerank`.
3. Re-run `python -m pytest -q`; artifact tests will fail if the report and the
   files disagree.
4. Commit the regenerated `data/` and `evidence/` together. Never commit one
   without the other.

## Known gaps

- No concurrency/load regression; performance evidence is single-client.
- No answer-level faithfulness judge.
- Flaky-network behaviour is tested with fakes, not against a live API.
- The 50-query eval set is small; confidence intervals are reported, but a
  regression smaller than ~0.05 recall should be treated as noise.

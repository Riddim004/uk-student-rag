# Failure modes

What can go wrong, how it is detected, what the system does, and what the user
sees. Figures are from the test split of the 50-query eval set
(`evidence/eval/report.md`, commit `7b88dec`, corpus
`e12e0d7011f1d5c1`, reranker gate at `-3.0`).

## Failure taxonomy

### Retrieval and gating

| Failure | Definition | Detection | Current count (hybrid+rerank-gate, test) | User-visible handling |
|---|---|---|---:|---|
| `hit` | At least one labelled chunk at rank 1 | labelled chunk = retrieved #1 | 14 / 21 answerable | answer with citations |
| `partial_hit` | Labelled chunk present, but not first | labels recovered from `results.json` | 2 | answer with citations; ranking diagnostic in evidence panel |
| `ranking_error` | Labelled chunk in top-k but outside the position needed for a clean hit | per-query metrics | 5 | answer; the "why" panel exposes dense/sparse/rerank ranks |
| `retrieval_miss` | No labelled chunk in top-k | recall = 0 | 0 | gate refuses, or answers from an unlabelled near-duplicate if the gate passes |
| `over_refusal` | Answerable query refused | gate decision + labels | 0 | refusal screen; counted as a product defect |
| `correct_refusal` | Unanswerable query refused | gate decision + labels | 2 / 4 | explicit refusal with a link to official guidance |
| `unanswerable_answered` | Unanswerable query answered | gate decision + labels | 2 / 4 | grounded answer to a question the corpus cannot answer; residual risk |

The two remaining gate misses are `q40` ("British citizenship after a Student
visa") and `q42` ("private health insurance in addition to the IHS"). Both are
semantically adjacent to the corpus, so the cross-encoder scores them highly.
A single scalar threshold cannot separate them; this is a known limitation, not
a threshold that was tuned away.

### Generation

| Failure | Detection | Handling |
|---|---|---|
| Chat model timeout / 429 / 5xx | `is_transient` on the exception | up to 3 attempts with exponential backoff + jitter |
| Auth or malformed request | non-transient status | fail fast with `GenerationError`; no retry |
| Empty completion | `text.strip() == ""` | retried as transient; then a `GenerationError` |
| Model refuses despite relevant context | exact refusal string | `Answer(refused=True, refused_reason="model_refusal")` |
| Model cites a passage that does not exist | marker parsing | out-of-range markers are dropped; only real chunks become citations |
| Model unavailable for the whole retry budget | `GenerationError` | product path falls back to a cited extractive answer, flagged `degraded`; CLI/API expose the error |
| Model makes an unsupported claim | not automatically detected | **known gap**: there is no answer-level faithfulness metric yet; retrieval evidence is shown so a reader can check |

### Data and configuration

| Failure | Exception | Handling |
|---|---|---|
| Corpus file missing | `DataError` | message includes the rebuild command |
| Empty corpus | `DataError` | fails at load, not as a fake refusal |
| Malformed JSONL | `DataError` | includes file and line number |
| Document without text / duplicate id | `DataError` | includes the offending id or line |
| Embedder returns wrong number of vectors or inconsistent dimensions | `EmbeddingError` | refuses to build a corrupt index |
| Invalid chunk size/overlap or unknown backend | `ConfigurationError` | fails at startup |
| Manifest hash mismatch | test failure | `test_bundled_corpus_matches_manifest_hashes` |

## Invariants

1. **No silent data loss.** A missing or corrupt corpus raises `DataError`; it is
   never rendered as "I don't know".
2. **No fake refusal.** Backend failures raise `GenerationError` or produce a
   visibly `degraded` extractive answer. They are not converted into the
   relevance refusal text.
3. **No unsourced citation.** Citations are built by parsing the markers in the
   answer and mapping them back to the exact chunks that were sent; markers
   outside that range are discarded.
4. **Bounded work.** Chat/embedding calls have a 30 s timeout and at most 3
   attempts. Retrieval itself is in-memory and deterministic.
5. **Hashes identify stored bytes.** Build artifacts are UTF-8 + LF on every OS
   (see Incident 005), and eval artifacts are checked against them in CI.

## Residual risks

| Risk | Why it remains | Mitigation / next step |
|---|---|---|
| Two unanswerable questions still answered | embedding similarity does not equal answerability | add an NLI/LLM answerability verifier over the top passages; measure it on the same 8 queries |
| No answer faithfulness metric | requires an LLM judge or human annotation | add a small human-rated answer set (20–30 questions) and a citation-support check |
| Test split is small (4 unanswerable) | 50-query budget | report bootstrap CIs; treat gate accuracy as indicative, not precise |
| Live OpenAI path not exercised in CI | no key / cost | inject fake SDK errors in tests; document a manual smoke command |
| Sequential benchmark only | no concurrent load test | M3 reports throughput for 1 client; a concurrency test is listed as future work |
| Chinese answers in degraded mode are English quotes | extractive fallback copies source sentences | UI labels the mode; an LLM answer path is required for Chinese prose |

## How the failure paths are exercised

```bash
python -m pytest -q tests/test_reliability.py tests/test_artifacts.py
```

`tests/test_reliability.py` injects backend failures, corrupt files, duplicate
ids, bad chunk settings, and a wrong-count embedder. `tests/test_artifacts.py`
checks that the reports still describe the data that is actually shipped.

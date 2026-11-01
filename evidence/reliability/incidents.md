# Incidents and design findings

These are real issues found while rebuilding the project. Each entry records the
symptom, evidence, root cause, fix and regression test. The point is not that
the first version was broken; it is that the second version can prove what was
broken and that it stays fixed.

## Incident 001 — the system would not refuse

**Symptom.** The default hybrid pipeline answered `xylophone quantum banana`
instead of refusing. The LLM was called with five irrelevant contexts.

**Evidence (before the fix).**

```text
[hybrid] nonsense query -> refused = False | LLM called = True
top contexts: [('applying', 0.01639), ('working', 0.01613), ...]
[dense ] nonsense query -> refused = False | LLM called = True
dense scores: [0.075955, 0.059949, 0.0, 0.0, 0.0]
```

**Root cause.** Two independent mistakes compounded:

1. The generator filtered contexts with `score >= min_context_score` and
   `min_context_score` defaulted to `0.0`. Dense retrieval returns top-k even
   when every cosine is zero, so zero-score chunks passed.
2. The default retriever was hybrid, whose RRF score is always positive
   (`1 / (60 + rank)`). RRF is a *rank fusion* score, not a probability or a
   similarity. It can never answer "is this query in the corpus?".

The only refusal test used BM25, which returns an empty list for unknown terms,
so the test passed while the default path was broken.

**Fix.**

- The gate now reads the underlying component scores (`dense`, `sparse`,
  `rerank`) instead of the fused score.
- A query-level gate uses the strongest evidence, not any weak per-chunk hit.
- `RerankGate` uses a cross-encoder only for the answer/refuse decision.
- Regression tests:
  - `test_gate_cannot_be_fooled_by_positive_rrf_scores`
  - `test_pipeline_refuses_when_gate_rejects_retrieval_miss`
  - `test_rerank_gate_refuses_when_best_score_is_too_low`

**After the fix (test split, 4 unanswerable / 21 answerable).**

| Gate | Unanswerable refused | Answerable over-refused |
|---|---|---|
| dense-only scalar | 0.000 | 0.000 |
| BM25 scalar | 0.000 | 0.143 |
| hybrid scalar | 0.000 | 0.000 |
| reranker gate `-3.0` | 0.500 | 0.000 |

The reranker gate is the only one that refuses anything without over-refusing.
It still misses two out-of-domain questions (`q40` British citizenship, `q42`
private health insurance) because they are semantically close to the corpus.
A scalar threshold cannot fix that; the next step is an answerability verifier
(LLM self-check or NLI) over the top passages.

## Incident 002 — Chinese queries silently returned nothing for BM25

**Symptom.** README examples in Chinese produced empty BM25 results, and the
old `HashingEmbedder` produced an all-zero vector for
`毕业后能在英国停留多久？`.

**Root cause.** Both tokenizers used `[a-z0-9]+`, which drops every CJK
character.

**Fix.** `ragkit/tokenization.py` now normalises with NFKC and emits Latin words
plus CJK unigrams and bigrams. The dense backend is a multilingual E5 model.

**After the fix (test split, recall@10).**

| System | Chinese | English exact | English paraphrase |
|---|---:|---:|---:|
| BM25 | 0.250 | 1.000 | 0.900 |
| dense (multilingual E5) | 1.000 | 1.000 | 0.800 |
| hybrid | 1.000 | 1.000 | 1.000 |

The hybrid's Chinese recall is 1.000. The BM25-only row is a useful negative
result: lexical retrieval alone cannot serve the bilingual product.

## Incident 003 — reranking improved recall but hurt early rank

**Symptom.** Adding the fast multilingual cross-encoder to the end of hybrid
raised recall@10 on test from 0.972 to 0.988, but MRR fell from 0.845 to 0.771
and nDCG@10 fell from 0.820 to 0.808.

**Why this is not automatically a bug.** Recall@10 counts whether any relevant
chunk is somewhere in the top 10. MRR and nDCG care where it is. This corpus has
1–4 labelled chunks per query and many near-duplicate "money"/"visa fee"
sections; the cross-encoder promotes a different relevant chunk, or an
unlabelled near-duplicate, above the labelled first hit.

**Decision.** Do not ship reranking as the default ranking. Use the
cross-encoder only as a **gate**: keep the hybrid order for the answer, but use
the best rerank score to decide whether to answer.

**Evidence.** `hybrid+rerank-gate` has exactly the hybrid retrieval numbers
(recall 0.972, MRR 0.845, nDCG 0.820) and the reranker gate's refusal numbers
(50% unanswerable refused, 0% answerable over-refused on test). The reordering
variant has higher recall but worse MRR/nDCG. See
[`evidence/eval/report.md`](../eval/report.md).

**Trade-off.** This choice preserves measured ranking quality but may feed the
generator a lower-ranked relevant passage when the best reranked passage is not
first. The next step is an answer-level eval to decide whether the gate-only
order or the reranked order produces better answers.

## Incident 004 — the first eval was structurally unable to fail

**Symptom.** The first version's scorecard showed `recall@5 = 1.000`,
`MRR = 1.000`, `MAP = 1.000` for BM25 and near-perfect scores for every
retriever.

**Root cause.** The corpus had 6 documents and 7 chunks; `k=5` returned 5 of 7
chunks, so recall@5 was near 1.0 by construction. Each query had exactly one
relevant *document*, so precision@5 was capped at 0.2, and the implementation
divided by the number of returned items instead of `k`. There was no dev/test
split and no unanswerable set.

**Fix.** 74 sections / 167 chunks; 50 chunk-level labelled queries; 25/25
stratified dev/test split; 8 unanswerable queries; proper `precision@k`
denominator; nDCG; bootstrap confidence intervals; per-category results and an
error taxonomy.

**Lesson.** A metric can be perfectly implemented and still be meaningless. The
dataset card and the dev/test split are part of the model, not paperwork.

## Incident 005 — manifest hashes did not match the files on disk

**Symptom.** A new integrity test compared the SHA-256 of the stored corpus with
`data/corpus/manifest.json` and failed on Windows:

```text
documents.jsonl  sha256 8fa1452b2fdf...  != manifest e12e0d7011f1...
chunks.jsonl     sha256 2ba0c6d4f256...  != manifest 40c5b768d5e6...
```

The hashes were the ones printed during the original build, so the data itself
was fine; the bytes on disk were not the bytes that had been hashed.

**Root cause.** The build hashed the in-memory string (LF newlines), then wrote
it with `Path.write_text`, which translates `\n` to `\r\n` on Windows. Anyone
re-verifying the artifact on Windows got a different hash; the provenance claim
was only true on Linux.

**Fix.** All build artifacts are written as UTF-8 bytes with explicit LF
(`write_utf8` in `ragkit/govuk.py`). The corpus was rebuilt and both hashes now
match the published values exactly. `.gitattributes` also forces LF on checkout,
so cloning on Windows cannot reintroduce the mismatch. Regression test:
`test_bundled_corpus_matches_manifest_hashes`.

**Lesson.** A provenance hash must be computed over the bytes that are actually
stored, not over an intermediate string. This is also why the eval report pins
both `corpus_hash` and `chunk_hash`.

## Incident 006 — model outages had no timeout, retry, or failure state

**Symptom.** Failure-path audit: `OpenAIChatModel.complete` called the SDK once,
with no explicit timeout and no retry. A timeout, 429, or 5xx escaped as a raw
SDK exception; an empty completion was converted into a normal refusal.

**Why it matters.** Two bad outcomes: a transient outage becomes a 500 with no
recovery, or a backend problem is reported to the user as "I don't have enough
information", which is factually wrong and untraceable.

**Fix.**

- Chat and embedding clients set an explicit timeout and disable the SDK's
  hidden retry policy.
- A bounded retry loop (3 attempts, exponential backoff, jitter) retries only
  timeouts, connection errors, 429, and 5xx. Auth/validation errors fail fast.
- Exhausted retries raise `GenerationError` / `EmbeddingError`; they are never
  turned into a relevance refusal.
- For the product surface, `FallbackGenerator` degrades to a cited extractive
  answer and marks it `degraded`, so an outage is visible but the user still
  gets grounded text.
- Failure-injection tests cover retry success, retry exhaustion, non-transient
  fast failure, empty completions, and the degraded path.

**Evidence.** `tests/test_reliability.py` (23 tests, no network).


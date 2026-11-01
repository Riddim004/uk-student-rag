# Design notes

A one-page map of the code and the invariants it protects. The measured
evidence and the rejected alternatives live in [`evidence/`](../evidence);
this file is about how the pieces fit together.

## Problem

A student asks a procedural question — fees, hours, dependants, deadlines — and
needs an answer they can verify. A generic chatbot is the wrong shape: it hides
its sources, cannot say "the file does not cover this", and gives no way to tell
whether a change helped.

The system therefore has four jobs:

1. Retrieve the right **section** from official guidance, in English or Chinese.
2. Decide whether the corpus can answer at all.
3. Produce a cited answer, from an LLM or from verbatim source sentences.
4. Expose enough state that a reader — and the evaluation — can audit 1–3.

## Data flow

```text
gov.uk Content API
  → data/raw/govuk/{doc}.json          frozen JSON + fetched_at + content hash
  → sections                          one guide part or H2 section
  → data/corpus/documents.jsonl       74 sections
  → chunks                            260 words, 40 overlap, section header
  → data/corpus/chunks.jsonl          167 chunks (+ hash in manifest)
```

`ragkit/govuk.py` owns acquisition and serialization. `ragkit/chunking.py` owns
the section/window logic. `ragkit/ingest.py` loads JSONL, validates it, embeds
chunks, and wires the retrievers. `scripts/build_data.py` orchestrates and writes
all hashes.

The data build is deterministic offline: the same raw snapshot produces the same
`documents.jsonl`, `chunks.jsonl`, corpus hash, and chunk hash. Artifacts are
written as UTF-8 + LF on every OS (Incident 005).

## Retrieval

- `ragkit/tokenization.py`: NFKC, Latin words, CJK unigrams + bigrams.
- `BM25Retriever`: pure Python, returns `components={"sparse": score}`.
- `DenseRetriever` + `InMemoryVectorStore`: E5-small, normalized cosine.
- `HybridRetriever`: RRF with `k=60`, tracks sparse/dense scores and ranks.

RRF is a rank-fusion score, not a probability. Nothing downstream interprets it
as "the corpus can answer". This distinction caused Incident 001.

## Gate

`ragkit/relevance.py` defines two gates:

- `RelevanceGate`: scalar thresholds on the strongest dense/BM25 component.
  Kept as a documented negative result and for cheap/offline paths.
- `RerankGate`: runs the cross-encoder over the top 15 hybrid candidates and
  compares the **best** rerank score to `min_rerank_score = -3.0`. It annotates
  candidates but keeps the hybrid order; `decide()` returns `(accepted, best)`.

The pipeline calls `annotate()` once, records `gate_accepted`, `gate_best`,
`gate_threshold`, and `gate_candidate_k` in `PipelineTrace`, and refuses before
generation if the gate rejects.

## Generation

`ragkit/generator.py` has three implementations of the same `Generator`
protocol:

- `LLMGenerator`: grounded prompt, citation parsing, exact refusal handling.
- `ExtractiveGenerator`: rare-term-weighted sentence selection, long-sentence
  trimming, verbatim quotes with markers; deterministic and key-free.
- `FallbackGenerator`: on `GenerationError`, degrade to extractive and mark the
  answer.

`ragkit/llm.py` owns timeouts/retries and raises `GenerationError` after the
budget; it never turns an outage into a refusal. `ragkit/errors.py` is the
taxonomy used by the API and tests.

## API and product

`ragkit/runtime.py` builds the corpus, warms the reranker, selects LLM vs
extractive mode, and holds the eval summary/examples. `ragkit/api.py` serializes
the answer, gate decision, contexts, and retrieval trail. SSE emits the real
stages from a worker thread; the process-wide lock makes the concurrency claim
explicit (serialized).

The UI in `web/` is deliberately build-free. It renders citations, exhibits,
score meters, the gate meter, and timings from the same payload used by
`/api/ask`.

## Evaluation

`ragkit/evals/` separates:

- `dataset.py`: joins `queries.jsonl` + `labels.jsonl`, validates, hashes.
- `metrics.py`: recall, precision (denominator k), nDCG, MRR, MAP, bootstrap CI.
- `runner.py`: per-query results and the error taxonomy.
- `run.py`: builds models once, evaluates systems/splits, renders metadata.

`scripts/run_eval.py` writes the report; `scripts/calibrate_gate.py` selects
thresholds on dev and checks them on test; `scripts/measure.py` writes the
performance and cost evidence. `tests/test_artifacts.py` fails if the committed
reports and the data hashes disagree.

## Invariants

1. Data problems raise `DataError`; they never look like a refusal.
2. Model problems raise `ModelError`/`GenerationError` or produce a visibly
   degraded answer; they never look like a refusal.
3. A citation maps to a chunk that was actually sent to the generator.
4. Gate thresholds are selected on dev; test is a held-out check.
5. Every reported number is tied to corpus, chunk, eval, and code identity.
6. The deterministic test suite runs without a network or a model download.

## Where to extend

- New source: add a `SourceSpec` in `ragkit/govuk.py`, rebuild, relabel.
- New retriever: implement `Retriever.retrieve` and add it to `build_systems`.
- New gate: implement `filter`/`would_refuse` (and optionally `annotate`).
- New generator: implement `generate`/`refuse`; wrap with `FallbackGenerator`.
- Answer-level eval: add a human-rated set and a citation-support check; keep the
  retrieval floors as the regression base.

Design trade-offs and their costs: [ADR index](../evidence/tradeoffs/).
Failure history: [incidents](../evidence/reliability/incidents.md).

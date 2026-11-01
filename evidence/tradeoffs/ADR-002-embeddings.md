# ADR-002 — Local multilingual E5-small as the dense backend

**Status:** accepted · 2026-11-01

## Context

The product is bilingual (English questions, Chinese questions over English
gov.uk text). The original tokenizer dropped every CJK character, so BM25
returned nothing for Chinese. Dense retrieval had to work offline, on CPU, with
no per-query bill.

## Decision

Use `intfloat/multilingual-e5-small` (384 dimensions) with E5 `query:` /
`passage:` prefixes and L2 normalisation. Keep BM25 as the lexical channel.
Allow `EMBEDDING_BACKEND=openai` for hosted embeddings and `hashing` only for
hermetic tests.

## Alternatives

| Alternative | Benefit | Cost |
|---|---|---|
| OpenAI `text-embedding-3-small` | strong, no local model | network, key, per-query cost, data leaves the machine |
| English-only MiniLM | smaller/faster | Chinese queries fail; the original bug returns |
| BM25 + character n-grams only | zero model | Chinese recall 0.25 in the eval; no semantic paraphrases |
| `multilingual-e5-base` | likely better quality | ~3× parameters and latency for a CPU demo |

## Consequences

- Chinese recall@10: BM25 0.250 → hybrid 1.000 (test split).
- Model load ~8.0 s, embedding 167 chunks ~9.2 s (55 ms/chunk), RSS +~970 MB.
- Dense query latency ~16 ms; hybrid ~16 ms. The reranker, not the embedder,
  dominates end-to-end latency.
- The local path is free at the margin; the hosted path is priced in
  `evidence/cost/report.md`.
- Evidence: Incident 002, `evidence/eval/report.md`, `evidence/performance/report.md`.

## Would change my mind

If the deployment target has no 2 GB RAM headroom, use hosted embeddings; if
Chinese answer quality becomes the priority, evaluate a larger multilingual
model and measure the latency/cost delta rather than assuming it is better.

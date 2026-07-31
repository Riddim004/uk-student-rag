# ragkit — Design Notes

> Why this RAG engine is built the way it is. The code shows *what*; this file
> is the *why*, plus the honest limitations and an interview-style Q&A. It
> doubles as my own study material — if I can't answer a question here in my own
> words, I don't understand my own system yet.

## The problem I set out to solve

Most RAG demos are "embed everything, cosine-search, stuff into a prompt". They
look fine on the query you tried and fall over on the next one, and — crucially —
there's no way to know whether a change helped. I wanted a RAG system where the
**retriever is measured**, the **answers are traceable**, and the **whole thing
is testable without spending money or hitting the network**.

## Design goals

1. **Offline-testable.** The retriever, generator, pipeline and evaluation must
   run with no API key and no network, so they live in CI. This drove two key
   choices: a pure-Python BM25 and a deterministic `HashingEmbedder`.
2. **Measure retrieval, not just vibes.** Separate retrieval-quality evaluation
   (IR metrics on labelled queries) from answer quality.
3. **Trustworthy answers.** Grounded, cited, and willing to refuse.
4. **Swappable everything.** Embedder, chat model and retriever are protocols so
   real backends drop in without touching the pipeline.

## Retrieval: why hybrid, and why RRF

BM25 and dense retrieval fail differently:

- **BM25** (lexical) is unbeatable when the query shares exact terms with the
  text — names, numbers, jargon — but it's blind to paraphrase ("work hours" vs
  "employment limit").
- **Dense** (semantic) catches paraphrase but drifts on rare tokens and exact
  figures, and depends entirely on embedding quality.

**Reciprocal Rank Fusion** combines them by rank, not score:

```
RRF(chunk) = Σ_retrievers  weight / (rrf_k + rank_in_that_retriever)
```

I fuse *ranks* on purpose: BM25 scores and cosine similarities live on totally
different scales, so averaging the raw scores is meaningless. Ranks are
comparable. `rrf_k` (default 60, from the original RRF paper) damps the
influence of the very top ranks so one retriever can't completely dominate.

### BM25 from scratch

I implemented Okapi BM25 by hand rather than importing `rank_bm25`, because being
able to explain it is the point:

- **`k1`** controls term-frequency saturation — the 10th occurrence of a word
  adds far less than the 2nd.
- **`b`** controls length normalisation — without it, long documents win just by
  containing more words.

Explaining those two knobs is a common interview probe, and now the code is mine
to explain.

## Generation: grounding, citations, refusal

The generator enforces two guarantees:

- **Citations map to real chunks.** The prompt requires inline `[n]` markers; I
  then parse those markers and keep only the ones in range, building `Citation`
  objects that point at actual chunks. An out-of-range `[5]` (a hallucinated
  reference) is dropped, not surfaced.
- **Refusal is programmatic first.** If retrieval returns nothing usable, I
  return the refusal answer **before** calling the model — no tokens spent, no
  chance to hallucinate. The model is also instructed to refuse with a fixed
  string if the context doesn't cover the question, which I detect to set
  `refused=True`.

## Testability: dependency injection

Every external dependency is a small protocol:

- `Embedder` — `HashingEmbedder` (offline) or `OpenAIEmbedder`.
- `ChatModel` — `FakeChatModel` (tests) or `OpenAIChatModel`.
- `Retriever` — BM25 / dense / hybrid, or a fake returning canned rankings.

That's why the RRF fusion order is tested with *fake* retrievers returning known
rankings (exact, deterministic), and the pipeline is tested with a fake chat
model that records its prompt. **The evaluation runs on the offline embedder, so
CI gets a real scorecard for free.**

## Reading the offline scorecard honestly

On the bundled set, BM25 currently *beats* dense and hybrid. I could have hidden
that; instead it's in the README, because it's the most instructive result here:

- The eval questions overlap lexically with the docs → BM25's best case.
- The offline `HashingEmbedder` has **no semantics** (it's a hashing bag of
  words), so dense retrieval is weak, and RRF inherits that weakness.
- Therefore **hybrid only pays off with a real embedder and paraphrased
  queries**, which is exactly when lexical overlap drops. Run
  `python -m ragkit.evals --openai` to see that regime.

The honest takeaway: hybrid isn't magic; it's a hedge. Whether it wins is an
empirical question — which is why the eval harness exists.

## Limitations & what I'd do next

- **Weak offline embedder.** `HashingEmbedder` exists for testability, not
  quality. Real deployments use `OpenAIEmbedder` (or a local sentence encoder).
- **Word-window chunking** can split a sentence mid-thought. Paragraph- or
  heading-aware chunking is the next improvement.
- **Small eval set, doc-level labels.** Six queries prove the harness; a real
  benchmark needs dozens, with chunk-level relevance and paraphrased queries.
- **No reranker.** A cross-encoder reranking stage after fusion usually helps
  precision and slots in behind the same `Retriever` interface.
- **Retrieval-only evals.** I measure retrieval; answer-level faithfulness /
  citation-accuracy evals are the natural next layer.

## Interview prep — questions I should be able to answer

- **Why fuse by rank (RRF) instead of averaging scores?** Because BM25 scores and
  cosine similarities are on different, uncalibrated scales; ranks are directly
  comparable, so fusion isn't dominated by whichever scorer has larger numbers.
- **What do BM25's `k1` and `b` do?** `k1` = term-frequency saturation; `b` =
  document-length normalisation.
- **Your hybrid loses to BM25 in the scorecard — is that a bug?** No. The offline
  embedder has no semantics and the queries are keyword-heavy, so BM25 should
  win. Hybrid's benefit appears with real embeddings and paraphrased queries.
- **How do you catch a hallucinated citation?** Inline `[n]` markers are parsed
  and validated against the retrieved chunk count; out-of-range markers are
  dropped and never become citations.
- **How can you test a RAG system without an API?** Inject a deterministic
  embedder and a fake chat model. BM25 is pure Python, so retrieval, generation
  logic and the metrics are all deterministic and offline.
- **Why evaluate retrieval separately from the answer?** Because most answer
  failures are retrieval failures. If the right chunk never surfaces, no prompt
  will save the answer — so I measure recall@k / MRR first.

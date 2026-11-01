# ADR-001 — Section-aware word chunks (260 words, 40 overlap)

**Status:** accepted · 2026-11-01

## Context

The corpus is 74 gov.uk sections (one guide part or one H2 section). Some
sections are a few hundred words; the Immigration Rules section is ~9,000. A
retrieval unit must be small enough to cite precisely, but large enough to carry
the rule and its exceptions.

## Decision

Chunk each section into overlapping 260-word windows with 40-word overlap, and
prefix every chunk with `[Document: … | Section: …]`. Never cross a section
boundary. The header is stripped before generation/display, but stays in the
indexed text and in BM25 terms.

## Alternatives

| Alternative | Benefit | Cost |
|---|---|---|
| One chunk per section | perfect section recall, simple citations | 9k-word chunks dilute dense embeddings and blow the prompt budget |
| Fixed 500-character chunks | predictable token size | cuts sentences/tables mid-rule; poor citations |
| Semantic/embedding-boundary chunking | coherent passages | another model, unstable boundaries, harder to reproduce, no gain proven on 167 chunks |
| Sentence-window retrieval | precise | many more chunks and a parent-document store |

## Consequences

- 167 chunks from 74 sections; mean 1.74 labelled chunks per answerable query.
- The header makes chunks self-describing but duplicates metadata; the generator
  removes it with `body_text` so prompts and quotes stay clean.
- Overlap preserves rules that straddle a boundary; it also creates near
  duplicates (visible in the q25/q49 ranking errors).
- Evidence: `evidence/eval/dataset_card.md`, Incident 005 on stored-byte hashes.

## Would change my mind

If answer-level evaluation shows that missed rules are caused by window cuts,
move to sentence-window + parent expansion. The current retrieval eval alone
cannot distinguish that from ranking noise.

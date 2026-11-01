# Dataset card — UK Student visa retrieval eval

## What this is

A small, versioned retrieval evaluation set over **real gov.uk guidance** for
international students. It is designed to answer one question honestly: does
the retriever put the answer-bearing passage in the top-k, and can the system
tell when the corpus does not contain the answer?

This is **not immigration advice**. The corpus is a dated snapshot; always check
the live page on gov.uk.

## Corpus snapshot

- Snapshot date: **2026-11-01**
- Sources: **9** gov.uk pages / guidance documents
- Sections: **74**
- Words: **30,035**
- Chunks: **167**
- Chunking: section-aware word windows, **260 words**, **40-word overlap**
- Corpus hash: `e12e0d7011f1d5c1d1df0ea3d975281729478dba4cc2619bc340ce98ecd41083`
- Chunk hash: `40c5b768d5e6d2c49a15d6ef904f62aa4ff240705d82483911cc7d3c4700cb20`

Sources were fetched through the [gov.uk Content API](https://www.gov.uk/api/content)
and saved as raw JSON, extracted Markdown and section-level JSONL. Every source
records its URL, publication update date, fetch time and content hash.

| doc_id | title | updated | sections | words |
|---|---|---|---:|---:|
| `student-visa` | Student visa | 2024-10-31 | 9 | 5,753 |
| `graduate-visa` | Graduate visa | 2024-10-31 | 6 | 3,139 |
| `ihs` | Pay for UK healthcare (IHS) | 2024-11-28 | 5 | 2,095 |
| `evisa` | eVisas | 2025-07-15 | 7 | 3,409 |
| `brp` | Biometric residence permits | 2024-10-31 | 5 | 499 |
| `atas` | Academic Technology Approval Scheme | 2026-03-24 | 15 | 3,794 |
| `visa-processing-times` | Visa processing times | 2026-09-09 | 11 | 1,024 |
| `financial-evidence` | Financial evidence for Student applicants | 2025-12-31 | 6 | 1,367 |
| `immigration-rules-student` | Immigration Rules: Appendix Student | 2026-10-08 | 10 | 8,955 |

The **Immigration Rules: Appendix Student** source is included deliberately:
the consumer-facing Student visa page says work "depends on what you're
studying", while the 20-hour rule lives in the rules. A credible system must
retrieve the rule, not paraphrase from memory.

## Query set

- **50 queries**: 42 answerable, 8 unanswerable
- Languages: 42 English, 8 Chinese
- Categories: 10 exact, 10 paraphrase, 8 numeric, 6 multi-hop, 8 unanswerable,
  8 Chinese
- Split: **25 dev / 25 test**, stratified by category and language
- Relevance: **chunk-level**; 1.74 relevant chunks per answerable query on
  average (min 1, max 4; 23 queries have more than one)
- Every answerable query also carries a short `reference_answer` for error
  analysis; it is not used by retrieval metrics.

### How queries were authored

1. Read the official source sections and write user-shaped questions **before**
   looking at retrieval results. Exact, paraphrased, numeric, multi-hop and
   Chinese variants were written intentionally.
2. Locate answer-bearing chunks by matching short fact phrases against the
   source text, restricted to the relevant section when a phrase is common.
3. Review the first-pass matches: remove chunks that merely mention the topic
   (e.g. only "expired BRP" without the usage rule), keep all chunks that
   contain an answer (e.g. both the guide and the rules when both state it).
4. Freeze the labels and hash `queries.jsonl` + `labels.jsonl` into the report.

Limitations of the labels:

- Single annotator; no inter-annotator agreement.
- Some questions have more than one valid answer source (guide vs Immigration
  Rules). The label set tries to include all answer-bearing chunks, but a
  retriever returning an equally valid unlabelled chunk is scored as a miss.
- Relevance is binary, not graded.

### Unanswerable set

Eight questions are deliberately outside the snapshot: pets, IELTS bands,
Child Student visa fee, police registration, Skilled Worker salary, British
citizenship, University of York tuition, and private health insurance. These are
used only for gate/refusal evaluation, never for recall.

## Licensing and provenance

gov.uk content is published under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
The snapshot keeps source URLs and update dates so attribution and freshness are
visible. The project is independent and is **not** immigration advice.

## Reproduce

```bash
python scripts/build_data.py --offline
python scripts/annotate_eval.py --write
python scripts/finalize_eval.py
python scripts/run_eval.py --rerank --split both
python scripts/calibrate_gate.py --rerank
```

The first command rebuilds sections and chunks from the raw snapshot; the second
and third regenerate labels and the versioned eval files; the last two produce
the retrieval report and gate calibration.

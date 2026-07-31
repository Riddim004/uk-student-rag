# 🔎 ragkit — a hybrid-retrieval RAG engine (with retrieval evals)

[![CI](https://github.com/Riddim004/UK-Student-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Riddim004/UK-Student-RAG/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

<p align="center">
  <b>English</b> | <a href="#-中文说明">中文</a>
</p>

A small, **fully-tested** Retrieval-Augmented Generation engine that does the
three things most demo RAGs skip:

1. **Hybrid retrieval** — BM25 (lexical) + dense (semantic), fused with
   **Reciprocal Rank Fusion**, not just a single vector search.
2. **Retrieval evaluation** — standard IR metrics (recall@k, precision@k, MRR,
   hit-rate, MAP) against a labelled query set, so retriever quality is
   *measured*, not eyeballed.
3. **Grounded, cited answers with refusal** — every claim carries a `[n]`
   citation mapped to a real chunk, and the system says "I don't know" when
   retrieval finds nothing.

The bundled corpus is UK student-visa guidance (I'm an international student, so
I'm the first user), but the engine is domain-agnostic.

> ⚠️ **Not immigration advice.** The bundled documents are illustrative samples;
> always verify with the official guidance on [gov.uk](https://www.gov.uk/student-visa).

## ✨ Why this is different

- 🧪 **Runs offline in CI.** A deterministic `HashingEmbedder` and a pure-Python
  BM25 mean the **retriever, the pipeline and the whole eval suite run with no
  API key and no network** — 22 tests, plus an offline scorecard on every push.
- 🔀 **Real hybrid retrieval.** RRF fuses *rankings* (scale-free) from BM25 and
  dense retrievers, which have complementary strengths.
- 📊 **Honest, measurable retrieval.** The eval compares bm25 / dense / hybrid
  head-to-head — see the [scorecard](#-retrieval-evaluation) and its honest
  interpretation.
- 🧩 **Dependency-injected.** Embedder and chat model are swappable protocols;
  that's what makes the tests hermetic.
- 🚫 **Refuses instead of hallucinating.** Empty retrieval ⇒ refusal *before*
  any model call.

## 🧠 Architecture

```
                    ingest                         query time
  ┌───────────┐   chunk+embed   ┌────────────┐   ┌──────────────────────────┐
  │ documents │ ───────────────▶│  corpus    │   │  BM25  ┐                  │
  │  (*.md)   │                 │ chunks +   │──▶│  dense ┼─▶ RRF fusion ─▶ top-k
  └───────────┘                 │ vectorstore│   │        ┘   (HybridRetriever)
                                └────────────┘   └───────────────┬──────────┘
                                                                  ▼
                                                        ┌───────────────────┐
                                                        │ generator          │
                                                        │ grounded + cited   │
                                                        │ or refuse if empty │
                                                        └─────────┬─────────┘
                                                                  ▼
                                                    Answer(text, citations, refused)
```

## 🚀 Quick start

The retrieval evaluation needs **no API key** — try it first:

```bash
pip install -r requirements.txt

# Compare retrievers offline (deterministic hashing embedder):
python -m ragkit.evals

# Ask a question (needs an API key for real embeddings + generation):
cp .env.example .env      # add your key
python -m ragkit "How many hours can I work during term time?"
python -m ragkit "How long can I stay after graduating?" --retriever hybrid
```

## 📊 Retrieval evaluation

`python -m ragkit.evals` scores each retriever over a labelled query set. Actual
**offline** output (deterministic `HashingEmbedder`, k=5):

```
| Retriever | recall@5 | precision@5 | MRR   | hit@5 | MAP   |
|-----------|----------|-------------|-------|-------|-------|
| bm25      | 1.000    | 0.233       | 1.000 | 1.000 | 1.000 |
| dense     | 1.000    | 0.225       | 0.917 | 1.000 | 0.917 |
| hybrid    | 1.000    | 0.225       | 0.917 | 1.000 | 0.917 |
```

**How to read this honestly:** on this small, keyword-heavy question set, **BM25
wins** — the questions share exact vocabulary with the documents, which is BM25's
home turf, while the offline `HashingEmbedder` is a bag-of-words hash with no real
semantics, so dense (and therefore hybrid) trail. This is expected, and it's the
point of *measuring*: hybrid's advantage shows up with **real embeddings**
(`python -m ragkit.evals --openai`) and **paraphrased** queries, where lexical
overlap is low. `precision@5` is naturally capped low because each question has a
single relevant document. See [`docs/design.md`](docs/design.md) for the full
analysis.

## 🖥️ CLI

```bash
python -m ragkit "<question>" [--retriever bm25|dense|hybrid] [--data DIR]
python -m ragkit.evals [--openai] [-k 5] [-o scorecard.md]
```

## 🗂️ Project layout

```
ragkit/
  config.py         # Settings (env-driven, testable)
  types.py          # Document / Chunk / ScoredChunk / Citation / Answer
  chunking.py       # word-window chunking with overlap (pure)
  embeddings.py     # Embedder protocol + OpenAI + offline HashingEmbedder
  vectorstore.py    # in-memory cosine store (no numpy)
  retrievers/
    bm25.py         # hand-written Okapi BM25
    dense.py        # embedding retrieval
    hybrid.py       # Reciprocal Rank Fusion
  generator.py      # grounded, cited generation + refusal
  llm.py            # OpenAI chat wrapper
  pipeline.py       # retrieve -> generate
  __main__.py       # CLI
  evals/
    metrics.py      # recall@k / precision@k / MRR / hit-rate / MAP (pure)
    dataset.py      # labelled query set
    runner.py       # evaluate + compare retrievers
    __main__.py     # `python -m ragkit.evals`
data/uk_student_visa/*.md   # sample corpus (illustrative)
tests/              # 22 offline tests
```

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
pytest
```

22 tests cover chunking, BM25 relevance, RRF fusion order, the IR metrics,
citation extraction and refusal, and the end-to-end pipeline — all with injected
fakes, so they need no API key and no network. CI runs them on Python 3.10–3.12.

## 🛣️ Roadmap

- Paragraph-aware chunking and a cross-encoder reranking stage.
- Larger labelled eval set with paraphrased and adversarial queries.
- Swap the in-memory store for a persistent vector DB (Chroma/Qdrant) behind the same interface.
- Answer-level evals (faithfulness / citation accuracy) on top of retrieval evals.

## 📄 License

MIT — see [LICENSE](LICENSE).

---

# 📖 中文说明

<p align="center">
  <a href="#-ragkit--a-hybrid-retrieval-rag-engine-with-retrieval-evals">English</a> | <b>中文</b>
</p>

一个小而**完整测试**的 RAG（检索增强生成）引擎，做到了大多数 demo 级 RAG 都跳过的三件事：

1. **混合检索** — BM25（词法）+ 向量（语义），用 **Reciprocal Rank Fusion（RRF）** 融合，而不是只做单路向量检索。
2. **检索评估** — 用标准 IR 指标（recall@k、precision@k、MRR、hit-rate、MAP）在带标注的问题集上评测，检索质量是**量出来的**，不是靠眼看。
3. **带引用、可拒答的接地生成** — 每条论断都带 `[n]` 引用并映射到真实文本块；检索不到时系统会明确说"我不知道"。

内置语料是英国学生签证指引（我本人就是留学生，也是第一个用户），但引擎与领域无关。

> ⚠️ **非移民建议。** 内置文档为示例性质，请务必以 [gov.uk](https://www.gov.uk/student-visa) 官方指引为准。

## ✨ 差异点

- 🧪 **可在 CI 里离线运行。** 确定性的 `HashingEmbedder` + 纯 Python BM25，让**检索器、管线和整个评估套件都无需 API、无需联网**——22 个测试，外加每次推送跑一次离线评分卡。
- 🔀 **真正的混合检索。** RRF 融合 BM25 与向量检索的*排名*（与分数量纲无关），二者优势互补。
- 📊 **诚实、可度量的检索。** 评估把 bm25 / dense / hybrid 三者正面对比——见下方[评分卡](#-检索评估)及其诚实解读。
- 🧩 **依赖注入。** embedder 和对话模型都是可替换协议，这正是测试可离线的关键。
- 🚫 **宁可拒答也不幻觉。** 检索为空时，在调用模型**之前**就拒答。

## 🧠 架构

```
       ingest（切块+embedding）            查询期
  文档(*.md) ──────────────▶ 语料(chunks+向量库) ──▶ BM25 ┐
                                                    dense ┼─▶ RRF 融合 ─▶ top-k
                                                          ┘   (HybridRetriever)
                                                              │
                                                              ▼
                                              生成器：接地+引用，空则拒答
                                                              │
                                                              ▼
                                        Answer(text, citations, refused)
```

## 🚀 快速开始

检索评估**无需 API key**，先跑它：

```bash
pip install -r requirements.txt

# 离线对比三种检索器（确定性 hashing embedder）：
python -m ragkit.evals

# 提问（真实 embedding + 生成需要 API key）：
cp .env.example .env      # 填入你的 key
python -m ragkit "term time 每周能工作多少小时？"
python -m ragkit "毕业后能在英国停留多久？" --retriever hybrid
```

## 📊 检索评估

`python -m ragkit.evals` 在带标注问题集上给每种检索器打分。以下是**离线**实际输出（确定性 `HashingEmbedder`，k=5）：

```
| Retriever | recall@5 | precision@5 | MRR   | hit@5 | MAP   |
|-----------|----------|-------------|-------|-------|-------|
| bm25      | 1.000    | 0.233       | 1.000 | 1.000 | 1.000 |
| dense     | 1.000    | 0.225       | 0.917 | 1.000 | 0.917 |
| hybrid    | 1.000    | 0.225       | 0.917 | 1.000 | 0.917 |
```

**如何诚实解读：** 在这个小而"关键词重合度高"的问题集上，**BM25 胜出**——问题与文档共享精确词汇，正是 BM25 的主场；而离线的 `HashingEmbedder` 是词袋哈希、没有真正语义，所以 dense（及 hybrid）落后。这符合预期，也正是"要度量"的意义所在：hybrid 的优势要在**真实 embedding**（`python -m ragkit.evals --openai`）和**改写型**问题（词法重合低）下才会显现。`precision@5` 天然偏低，因为每个问题只有一个相关文档。完整分析见 [`docs/design.md`](docs/design.md)。

## 🖥️ 命令行

```bash
python -m ragkit "<问题>" [--retriever bm25|dense|hybrid] [--data DIR]
python -m ragkit.evals [--openai] [-k 5] [-o scorecard.md]
```

## 🧪 测试

```bash
pip install -r requirements-dev.txt
pytest
```

22 个测试覆盖切块、BM25 相关性、RRF 融合顺序、IR 指标、引用抽取与拒答、以及端到端管线，全部使用注入的假对象，无需 API、无需联网。CI 在 Python 3.10–3.12 上运行。

## 🛣️ 路线图

- 段落感知切块 + cross-encoder 重排阶段。
- 更大的标注评估集，含改写型与对抗型问题。
- 用持久化向量库（Chroma/Qdrant）替换内存库，接口不变。
- 在检索评估之上叠加答案级评估（faithfulness / 引用准确率）。

## 📄 许可证

MIT，见 [LICENSE](LICENSE)。

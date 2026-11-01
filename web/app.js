/* Evidence Desk — no framework, no build step. */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const I18N = {
  en: {
    railLede: "A UK student-visa answer desk built on a frozen gov.uk snapshot. Every claim carries an exhibit.",
    status: "Status",
    gate: "Gate",
    corpus: "Corpus",
    chunks: "Chunks",
    startup: "Cold start",
    caseQuestions: "Case questions",
    railFoot: "Not legal advice. The corpus is a 2026-11-01 snapshot; rules and fees change. Open the cited gov.uk page before acting.",
    kicker: "Case file / Student route",
    titleA: "Answers with",
    titleB: "receipts.",
    deck: "Hybrid retrieval over official guidance, a cross-encoder that decides when to refuse, and a visible trail for every citation. No black box, no invented policy.",
    questionLabel: "Question under review",
    placeholder: "How many hours can I work during term time?",
    ask: "Check the file",
    hint: "⌘ / Ctrl + Enter",
    hintTwo: "50 hand-labelled questions · EN / 中文",
    stageRetrieve: "Hybrid retrieval",
    stageGate: "Admissibility gate",
    stageGenerate: "Drafting",
    finding: "Finding",
    copyLink: "Copy link",
    exhibitsCited: "Exhibits cited",
    exhibits: "Exhibits",
    why: "Why this answer",
    refusalExplain: "This question falls outside the frozen corpus. The closest passages are shown beside this card, but none cleared the admissibility gate. Check the official pages below rather than treating a nearby passage as an answer.",
    howItReads: "How the desk reads",
    howA: "Questions are embedded locally and matched against 167 chunks of gov.uk guidance. BM25 and dense results are fused by rank; a multilingual cross-encoder then decides whether the file can answer at all.",
    whatYouSee: "What you see",
    howB: "The answer, the exact passages behind each citation, dense/sparse/rerank scores, the gate verdict, and stage timings. If the evidence is weak, the desk refuses instead of guessing.",
    howToCheck: "How to check it",
    howC: "The score strip comes from a 50-question eval with 25 held-out test queries and 8 unanswerable ones. Reports, calibration, incidents and benchmark commands live in the repository.",
    answered: "Answered",
    refused: "Not in the file",
    serviceError: "Service error",
    modeLlm: "LLM grounded",
    modeExtractive: "Offline demo · extractive",
    modeConnecting: "connecting",
    modeError: "unavailable",
    gateScalar: "scalar",
    gateRerank: "rerank",
    closest: "Closest passages, not cited",
    contexts: "exhibits",
    candidates: "candidates inspected",
    copied: "Link copied",
    copyFailed: "Copy failed",
    emptyAnswer: "The model returned no text. Try again.",
    degradedNote: "The chat model failed after retries. This answer was degraded to a cited extractive fallback — the quotes are real, but the prose is not composed.",
    gatePassed: "gate passed",
    gateRefused: "gate refused",
    gateDisclosed: "best rerank score",
    threshold: "threshold",
    readExcerpt: "Read excerpt",
    openSource: "gov.uk ↗",
    density: "dense",
    lexical: "sparse",
    cross: "rerank",
    rankDense: "dense rank",
    rankSparse: "sparse rank",
    timings: "stage timings",
    retrievalTrail: "retrieval trail",
    corpusLine: "corpus",
    questionMark: "question",
    answerMode: "mode",
    sourceLang: "source language EN",
    disclaimer: "Snapshot evidence, not legal advice.",
    quotedEvidence: "Quoted evidence · copied verbatim from the cited passage",
    groundedAnswer: "Grounded answer · composed from the cited passages",
  },
  zh: {
    railLede: "一份基于 gov.uk 冻结快照的英国学生签证问答台。每个结论都带出处。",
    status: "状态",
    gate: "门控",
    corpus: "语料",
    chunks: "切片",
    startup: "冷启动",
    caseQuestions: "示例问题",
    railFoot: "非法律意见。语料为 2026-11-01 快照，规则与费用会变化；行动前请打开引用的 gov.uk 页面核对。",
    kicker: "案卷 / 学生签证",
    titleA: "答案，",
    titleB: "带凭据。",
    deck: "在官方指南上做混合检索，用交叉编码器判断该不该拒答；每个引用都有可见轨迹。没有黑箱，也不编造政策。",
    questionLabel: "待查问题",
    placeholder: "学期期间我每周能工作多少小时？",
    ask: "查卷",
    hint: "⌘ / Ctrl + Enter",
    hintTwo: "50 道人工标注问题 · 中 / EN",
    stageRetrieve: "混合检索",
    stageGate: "可答性门控",
    stageGenerate: "起草",
    finding: "结论",
    copyLink: "复制链接",
    exhibitsCited: "引用证据",
    exhibits: "证据",
    why: "为什么是这个答案",
    refusalExplain: "这个问题不在当前语料范围内。右侧列出的是最接近的段落，但都没有通过可答性门控。请以 gov.uk 官方页面为准，不要把相近段落当成答案。",
    howItReads: "如何检索",
    howA: "问题在本地向量化，与 167 个 gov.uk 指南切片匹配。BM25 与向量结果按排名融合，再由多语交叉编码器判断语料究竟能不能回答。",
    whatYouSee: "你会看到",
    howB: "答案、每条引用对应的原文、dense/sparse/rerank 分数、门控判定和各阶段耗时。证据不足时，系统会拒答而不是猜测。",
    howToCheck: "如何核查",
    howC: "顶部分数来自 50 道题的评测：25 道留出测试集，8 道不可答。报告、阈值标定、事故记录与基准测试命令都在代码仓库中。",
    answered: "已答复",
    refused: "不在卷内",
    serviceError: "服务错误",
    modeLlm: "LLM · 有据生成",
    modeExtractive: "离线演示 · 摘录式",
    modeConnecting: "连接中",
    modeError: "不可用",
    gateScalar: "标量门控",
    gateRerank: "重排门控",
    closest: "最接近的段落（未引用）",
    contexts: "条证据",
    candidates: "条候选",
    copied: "链接已复制",
    copyFailed: "复制失败",
    emptyAnswer: "模型没有返回文本，请重试。",
    degradedNote: "对话模型在重试后仍不可用，答案已降级为带引用的摘录式回退——引文是真实的，但措辞未经模型组织。",
    gatePassed: "门控通过",
    gateRefused: "门控拒答",
    gateDisclosed: "最高重排分",
    threshold: "阈值",
    readExcerpt: "展开原文",
    openSource: "gov.uk ↗",
    density: "dense",
    lexical: "sparse",
    cross: "rerank",
    rankDense: "向量排名",
    rankSparse: "词法排名",
    timings: "阶段耗时",
    retrievalTrail: "检索轨迹",
    corpusLine: "语料",
    questionMark: "问题",
    answerMode: "模式",
    sourceLang: "原文语言 EN",
    disclaimer: "快照证据，非法律意见。",
    quotedEvidence: "原文摘录 · 逐字引自引用段落",
    groundedAnswer: "有据回答 · 基于引用段落组织",
  },
};

const CATEGORY = {
  en: { exact: "exact", paraphrase: "paraphrase", numeric: "numeric", multi_hop: "multi-hop", chinese: "Chinese", unanswerable: "unanswerable" },
  zh: { exact: "精确", paraphrase: "改写", numeric: "数字", multi_hop: "多跳", chinese: "中文", unanswerable: "不可答" },
};

const state = {
  lang: "en",
  health: null,
  examples: [],
  controller: null,
  loading: false,
  last: null,
  startedAt: 0,
  requestId: 0,
};

const t = (key) => (I18N[state.lang] && I18N[state.lang][key]) || I18N.en[key] || key;

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function shortHash(value) {
  return value ? String(value).slice(0, 12) : "—";
}

function fmtMs(ms) {
  if (ms === null || ms === undefined || Number.isNaN(Number(ms))) return "—";
  const value = Number(ms);
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

function fmtScore(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function setLang(lang) {
  state.lang = lang;
  document.documentElement.lang = lang === "zh" ? "zh" : "en";
  $$("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  $$("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  $$("[data-lang]").forEach((button) => button.classList.toggle("is-on", button.dataset.lang === lang));
  renderExamples();
  renderHealth();
  if (state.last) renderAnswer(state.last);
  if (!state.last) setVerdictIdle();
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("is-on");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => node.classList.remove("is-on"), 1800);
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  setLang("en");
  wireEvents();
  const question = new URLSearchParams(window.location.search).get("q");
  if (question) {
    $("#question").value = question.slice(0, 500);
    updateCount();
  }
  try {
    const [healthRes, examplesRes] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/examples"),
    ]);
    if (!healthRes.ok) throw new Error(`health ${healthRes.status}`);
    state.health = await healthRes.json();
    const payload = examplesRes.ok ? await examplesRes.json() : { examples: [] };
    state.examples = payload.examples || [];
    renderHealth();
    renderExamples();
    if (question) ask(question);
  } catch (error) {
    const dot = $("#mode-dot");
    dot.className = "dot is-error";
    $("#mode-label").textContent = t("modeError");
    $("#rail-warning").hidden = false;
    $("#rail-warning").textContent = String(error.message || error);
  }
}

function renderHealth() {
  if (!state.health) return;
  const health = state.health;
  const mode = health.product_mode;
  const dot = $("#mode-dot");
  dot.className = `dot ${mode === "llm" ? "is-live" : "is-demo"}`;
  $("#mode-label").textContent = mode === "llm" ? t("modeLlm") : t("modeExtractive");

  const facts = $$("#facts > div");
  const gateLabel = health.gate_mode === "rerank"
    ? `rerank ${fmtScore(health.settings.min_rerank_score)}`
    : t("gateScalar");
  const sources = health.corpus?.sources?.length ?? "—";
  const words = health.corpus?.total_words ? `${Math.round(health.corpus.total_words / 1000)}k words` : "";
  const values = [
    gateLabel,
    `${sources} src ${words}`.trim(),
    `${health.corpus?.total_chunks ?? "—"}`,
    fmtMs(health.startup_ms),
  ];
  facts.forEach((row, i) => { const dd = row.querySelector("dd"); if (dd) dd.textContent = values[i]; });

  const warning = $("#rail-warning");
  if (health.warnings && health.warnings.length) {
    warning.hidden = false;
    warning.textContent = health.warnings.join(" · ");
  } else {
    warning.hidden = true;
  }

  const evalData = health.eval || {};
  const hybrid = evalData.hybrid || {};
  const gate = evalData.gate || {};
  $("#score-recall").textContent = fmtScore(hybrid.recall, 3);
  $("#score-ndcg").textContent = fmtScore(hybrid.ndcg, 3);
  $("#score-gate").textContent = fmtScore(gate.gate_accuracy, 3);
  $("#score-n").textContent = evalData.dataset?.n_queries ?? "—";
  $("#score-hash").textContent = shortHash(health.corpus?.chunk_hash);
  $("#score-hash").title = health.corpus?.chunk_hash || "";
  $("#score-recall").classList.toggle("is-good", Number(hybrid.recall) >= 0.9);
  $("#score-gate").classList.toggle("is-good", Number(gate.gate_accuracy) >= 0.9);

  const generated = health.corpus?.generated_at || evalData.metadata?.generated_at;
  if (generated) $("#snapshot").textContent = `Snapshot ${generated.slice(0, 10)}`;
  $("#foot-corpus").textContent = `corpus ${shortHash(health.corpus?.corpus_hash)}`;
}

function renderExamples() {
  const holder = $("#samples");
  if (!holder) return;
  holder.innerHTML = "";
  state.examples.forEach((example) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample";
    button.dataset.q = example.question;
    button.innerHTML = `
      <span class="sample-cat">${esc(CATEGORY[state.lang][example.category] || example.category)}</span>
      <span class="sample-lang">${example.language === "zh" ? "中文" : "EN"}</span>
      <span class="sample-q">${esc(example.question)}</span>`;
    button.addEventListener("click", () => {
      $("#question").value = example.question;
      updateCount();
      ask(example.question);
    });
    holder.appendChild(button);
  });
}

/* ------------------------------------------------------------------ ask */

function setStages(stage) {
  const order = ["retrieve", "gate", "generate"];
  const index = stage ? order.indexOf(stage) : -1;
  $$("#process .stage").forEach((node) => {
    const stageIndex = order.indexOf(node.dataset.stage);
    node.classList.toggle("is-active", stageIndex === index);
    node.classList.toggle("is-done", index === -1 ? true : stageIndex < index);
    if (stageIndex === index) {
      node.querySelector(".stage-time").textContent = fmtMs(performance.now() - state.startedAt);
    }
  });
}

function applyServerTimings(timings) {
  const map = { retrieve: timings?.retrieve, gate: timings?.gate, generate: timings?.generate };
  $$("#process .stage").forEach((node) => {
    const value = map[node.dataset.stage];
    node.classList.remove("is-active");
    node.classList.add("is-done");
    node.querySelector(".stage-time").textContent = fmtMs(value);
  });
}

async function ask(question) {
  const trimmed = String(question || "").trim();
  if (!trimmed) { $("#question").focus(); return; }
  if (state.loading) state.controller?.abort();
  state.loading = true;
  state.last = null;
  state.startedAt = performance.now();
  $("#ask").disabled = true;
  $("#question").value = trimmed.slice(0, 500);
  updateCount();
  $("#empty").hidden = true;
  $("#verdict").hidden = true;
  $("#exhibit-panel").hidden = true;
  $("#process").hidden = false;
  $$("#process .stage").forEach((node) => {
    node.classList.remove("is-active", "is-done");
    node.querySelector(".stage-time").textContent = "—";
  });
  setStages("retrieve");
  window.history.replaceState({}, "", `${location.pathname}?q=${encodeURIComponent(trimmed)}`);

  const controller = new AbortController();
  const requestId = ++state.requestId;
  let gotAnswer = false;
  let gotError = false;
  state.controller = controller;
  try {
    const response = await fetch(`/api/ask/stream?q=${encodeURIComponent(trimmed)}`, {
      signal: controller.signal,
      headers: { Accept: "text/event-stream" },
    });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    await readSse(response.body, (event, data) => {
      if (event === "stage") {
        if (data.stage === "done") setStages(null);
        else setStages(data.stage);
      } else if (event === "answer") {
        gotAnswer = true;
        if (requestId !== state.requestId) return;
        state.last = data;
        applyServerTimings(data.timings_ms || {});
        renderAnswer(data);
      } else if (event === "error") {
        gotError = true;
        if (requestId === state.requestId) renderError(data);
      }
    });
  } catch (error) {
    if (error.name !== "AbortError" && requestId === state.requestId) {
      gotError = true;
      renderError({ message: String(error.message || error) });
    }
  } finally {
    if (requestId === state.requestId) {
      if (!gotAnswer && !gotError) {
        renderError({ message: "The stream ended before an answer was returned." });
      }
      state.loading = false;
      $("#ask").disabled = false;
    }
  }
}

async function readSse(body, handler) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (!raw.trim() || raw.trim().startsWith(":")) continue;
      let event = "message";
      const dataLines = [];
      raw.split("\n").forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      });
      if (dataLines.length) {
        try { handler(event, JSON.parse(dataLines.join("\n"))); }
        catch (error) { console.error("bad SSE payload", error, dataLines); }
      }
    }
  }
}

/* --------------------------------------------------------------- render */

function setVerdictIdle() {
  $("#verdict-state").textContent = "—";
  $("#verdict-state").className = "verdict-state";
  $("#answer-text").innerHTML = "";
  $("#verdict").hidden = true;
}

function renderError(error) {
  $("#process").hidden = false;
  $$("#process .stage").forEach((node) => node.classList.remove("is-active"));
  $("#verdict").hidden = false;
  $("#exhibit-panel").hidden = true;
  const stateNode = $("#verdict-state");
  stateNode.textContent = t("serviceError");
  stateNode.className = "verdict-state is-error";
  $("#verdict-meta").textContent = `${t("answerMode")} · ${fmtMs(performance.now() - state.startedAt)}`;
  $("#degraded-note").hidden = true;
  $("#refusal-extra").hidden = true;
  $("#citation-list").hidden = true;
  $("#answer-text").className = "answer";
  $("#answer-text").innerHTML = `<p>${esc(error.message || "Unknown error")}</p>`;
  $("#answer-mode").textContent = "error";
  $("#answer-time").textContent = "";
  $("#answer-gate").textContent = "";
}

function formatAnswer(text) {
  const safe = esc(text || "");
  const withCitations = safe.replace(/\[(\d+)\]/g, (match, marker) =>
    `<button type="button" class="cite" data-marker="${marker}" aria-label="Exhibit ${marker}">${marker}</button>`);
  const paragraphs = withCitations.split(/\n{2,}/).filter((part) => part.trim());
  return (paragraphs.length ? paragraphs : [withCitations]).map((part) => `<p>${part.replace(/\n/g, "<br>")}</p>`).join("");
}

function renderAnswer(data) {
  $("#process").hidden = false;
  $("#verdict").hidden = false;
  $("#empty").hidden = true;
  $("#exhibit-panel").hidden = false;

  const stateNode = $("#verdict-state");
  if (data.refused) {
    stateNode.textContent = t("refused");
    stateNode.className = "verdict-state is-refused";
  } else {
    stateNode.textContent = t("answered");
    stateNode.className = "verdict-state is-answered";
  }

  const gate = data.gate || {};
  const counts = data.counts || {};
  const gateLine = gate.accepted
    ? `${t("gatePassed")} · ${t("gateDisclosed")} ${fmtScore(gate.best_rerank)} ≥ ${fmtScore(gate.threshold)}`
    : `${t("gateRefused")} · ${t("gateDisclosed")} ${fmtScore(gate.best_rerank)} < ${fmtScore(gate.threshold)}`;
  $("#verdict-meta").textContent = `${counts.candidates ?? 0} ${t("candidates")} → ${counts.contexts ?? 0} ${t("contexts")} · ${gateLine}`;

  const kicker = $("#answer-kicker");
  if (data.refused) {
    kicker.hidden = true;
  } else {
    kicker.hidden = false;
    kicker.textContent = data.answer_mode === "extractive" ? t("quotedEvidence") : t("groundedAnswer");
  }

  const degraded = $("#degraded-note");
  if (data.degraded) {
    degraded.hidden = false;
    degraded.textContent = `${t("degradedNote")}${data.degraded_reason ? ` (${data.degraded_reason})` : ""}`;
  } else {
    degraded.hidden = true;
  }

  $("#refusal-extra").hidden = !data.refused;
  $("#answer-text").className = `answer${data.answer_mode === "extractive" && !data.refused ? " is-quote" : ""}`;

  let answerText = data.answer;
  if (state.lang === "zh") {
    if (data.refused) {
      answerText = "没有足够证据支持回答。";
    } else if (data.answer_mode === "extractive") {
      answerText = `以下为与问题最相关的原文摘录（${t("sourceLang")}）：\n\n${data.answer}`;
    }
  }
  $("#answer-text").innerHTML = formatAnswer(answerText);

  const citations = data.citations || [];
  const citationList = $("#citation-list");
  citationList.hidden = citations.length === 0;
  $("#citations").innerHTML = citations.map((citation) => `
    <li>
      <span class="marker">[${citation.marker}]</span>
      <div>
        <div class="cite-title">${esc(citation.title || citation.doc_id)}</div>
        <div class="cite-meta">${esc((citation.section_path || []).slice(-1)[0] || "")}${citation.updated_at ? ` · ${esc(citation.updated_at.slice(0, 10))}` : ""}</div>
        <a href="${esc(citation.source)}" target="_blank" rel="noopener">${esc(citation.source)}</a>
      </div>
    </li>`).join("");

  $("#answer-mode").textContent = `${t("answerMode")} ${data.answer_mode || (data.refused ? "refusal" : "—")}${data.degraded ? " · degraded" : ""}`;
  $("#answer-time").textContent = `${data.timings_ms?.total ?? "—"} ms total`;
  $("#answer-gate").textContent = `${t("gate")} ${gate.mode || "—"} · ${fmtScore(gate.best_rerank)} / ${fmtScore(gate.threshold)}`;

  renderExhibits(data);
  renderWhy(data);
  $("#exhibit-panel").scrollTop = 0;
}

function meter(label, value, display, width) {
  return `<div class="meter">
    <span>${esc(label)}</span>
    <span class="meter-bar"><i style="width:${Math.max(1, Math.min(100, width))}%"></i></span>
    <span class="meter-value">${esc(display)}</span>
  </div>`;
}

function renderExhibits(data) {
  const usingTrail = data.refused;
  const items = (usingTrail ? (data.trail || []).slice(0, 5) : (data.contexts || []));
  $("#exhibit-count").textContent = String(items.length).padStart(2, "0");
  $("#exhibit-sub").textContent = usingTrail ? t("closest") : `${items.length} ${t("contexts")} · ${data.gate?.mode || ""}`;
  const holder = $("#exhibits");
  holder.innerHTML = "";

  items.forEach((item, index) => {
    const marker = index + 1;
    const components = item.components || {};
    const dense = components.dense;
    const sparse = components.sparse;
    const rerank = components.rerank;
    const ranks = [
      components.dense_rank ? `d#${components.dense_rank}` : "",
      components.sparse_rank ? `s#${components.sparse_rank}` : "",
    ].filter(Boolean).join(" · ");
    const card = document.createElement("article");
    card.className = "exhibit";
    card.id = `exhibit-${marker}`;
    card.dataset.marker = String(marker);
    card.innerHTML = `
      <div class="exhibit-top">
        <div class="exhibit-no">${String(marker).padStart(2, "0")}</div>
        <div>
          <h3 class="exhibit-title">${esc(item.title || item.doc_id)}</h3>
          <p class="exhibit-section">${esc(item.section || "")}${item.updated_at ? ` · ${esc(item.updated_at.slice(0, 10))}` : ""}${ranks ? ` · ${esc(ranks)}` : ""}</p>
        </div>
      </div>
      <p class="exhibit-snippet">${esc(item.snippet || "")}</p>
      <div class="meters">
        ${dense !== null && dense !== undefined ? meter(t("density"), dense, fmtScore(dense, 3), dense * 100) : ""}
        ${sparse !== null && sparse !== undefined ? meter(t("lexical"), sparse, fmtScore(sparse, 2), (sparse / 30) * 100) : ""}
        ${rerank !== null && rerank !== undefined ? meter(t("cross"), rerank, fmtScore(rerank, 2), ((rerank + 5) / 15) * 100) : ""}
      </div>
      <div class="exhibit-links">
        <details>
          <summary>${t("readExcerpt")}</summary>
          <p class="exhibit-full">${esc(item.text || item.snippet || "")}</p>
        </details>
        <a href="${esc(item.source)}" target="_blank" rel="noopener">${t("openSource")}</a>
      </div>`;
    card.addEventListener("click", (event) => {
      if (event.target.closest("a, summary, details")) return;
      focusExhibit(marker);
    });
    holder.appendChild(card);
  });

}

function focusExhibit(marker) {
  const card = $(`#exhibit-${marker}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("is-target");
  setTimeout(() => card.classList.remove("is-target"), 1600);
}

function renderWhy(data) {
  const gate = data.gate || {};
  const timings = data.timings_ms || {};
  const trail = data.trail || [];
  const threshold = gate.threshold;
  const best = gate.best_rerank;
  const canPlot = threshold !== null && threshold !== undefined && best !== null && best !== undefined;
  const scale = (value) => Math.max(0, Math.min(100, ((value + 5) / 15) * 100));
  const meter = canPlot ? `
    <div class="gate-meter">
      <div class="why-row"><span>${t("gateDisclosed")}</span><b>${fmtScore(best)}</b></div>
      <div class="why-row"><span>${t("threshold")}</span><b>${fmtScore(threshold)}</b></div>
      <div class="gate-scale">
        <span class="gate-tick" style="left:${scale(threshold)}%" data-label="${fmtScore(threshold)}"></span>
        <span class="gate-best" style="left:${scale(best)}%" title="best ${fmtScore(best)}"></span>
      </div>
    </div>` : "";

  const rows = [
    [t("answerMode"), `${data.answer_mode || "—"}${data.degraded ? " · degraded" : ""}`],
    [t("gate"), `${gate.mode || "—"} · ${gate.accepted ? t("gatePassed") : t("gateRefused")}`],
    [t("candidates"), String(data.counts?.candidates ?? "—")],
    [t("timings"), ""],
    ...Object.entries(timings).map(([key, value]) => [`  ${key}`, fmtMs(value)]),
    [t("corpusLine"), shortHash(data.corpus?.corpus_hash)],
  ];

  $("#why-body").innerHTML = `
    ${meter}
    ${rows.map(([label, value]) => `<div class="why-row"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join("")}
    <div class="why-row" style="margin-top:8px"><span>${t("retrievalTrail")}</span><b>top ${trail.length}</b></div>
    ${trail.map((item, index) => {
      const c = item.components || {};
      const score = c.rerank ?? c.dense ?? c.sparse;
      return `<div class="trail-row">
        <span class="trail-rank">${String(index + 1).padStart(2, "0")}</span>
        <span class="trail-id" title="${esc(item.chunk_id)}">${esc(item.chunk_id)}</span>
        <span class="trail-score">${fmtScore(score)}</span>
      </div>`;
    }).join("")}`;
}

/* ------------------------------------------------------------------ UI */

function growQuestion() {
  const field = $("#question");
  field.style.height = "auto";
  field.style.height = `${Math.min(field.scrollHeight, 190)}px`;
}

function updateCount() {
  $("#char-count").textContent = String($("#question").value.length);
  growQuestion();
}

function wireEvents() {
  $("#ask").addEventListener("click", () => ask($("#question").value));
  $("#question").addEventListener("input", updateCount);
  $("#question").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      ask($("#question").value);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== $("#question")) {
      event.preventDefault();
      $("#question").focus();
    }
    if (event.key === "Escape" && document.activeElement === $("#question")) {
      $("#question").blur();
    }
  });
  $("#answer-text").addEventListener("click", (event) => {
    const marker = event.target.closest("[data-marker]")?.dataset.marker;
    if (marker) focusExhibit(marker);
  });
  $("#citations").addEventListener("click", (event) => {
    const title = event.target.closest("li")?.querySelector(".marker")?.textContent || "";
    const marker = title.replace(/[^\d]/g, "");
    if (marker) focusExhibit(marker);
  });
  $("#share").addEventListener("click", async () => {
    const url = `${location.origin}${location.pathname}?q=${encodeURIComponent($("#question").value.trim())}`;
    try {
      await navigator.clipboard.writeText(url);
      toast(t("copied"));
    } catch {
      toast(t("copyFailed"));
    }
  });
  $$("[data-lang]").forEach((button) => {
    button.addEventListener("click", () => setLang(button.dataset.lang));
  });
}

boot();

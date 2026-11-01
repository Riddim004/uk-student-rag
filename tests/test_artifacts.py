"""Evidence artifacts must stay consistent with the code and data.

These tests fail when the committed reports drift from the corpus hashes, the
eval set, or the shipped thresholds — the failure mode that made the original
project's README impossible to trust.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragkit.config import Settings
from ragkit.evals.dataset import load_eval_set

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evidence" / "eval"
CORPUS = ROOT / "data" / "corpus"


def _json(name: str) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def test_report_hashes_match_the_bundled_artifacts() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    _queries, eval_hash = load_eval_set(EVAL)
    report = (EVAL / "report.md").read_text(encoding="utf-8")
    results = _json("results.json")

    for split in ("dev", "test"):
        meta = results["splits"][split]["metadata"]
        assert meta["corpus_hash"] == manifest["corpus_hash"]
        assert meta["chunk_hash"] == manifest["chunk_hash"]
        assert meta["eval_hash"] == eval_hash

    assert manifest["corpus_hash"] in report
    assert manifest["chunk_hash"] in report
    assert eval_hash in report


def test_dataset_stats_match_the_query_and_label_files() -> None:
    stats = _json("dataset_stats.json")
    queries = [
        json.loads(line)
        for line in (EVAL / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [
        json.loads(line)
        for line in (EVAL / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert stats["n_queries"] == len(queries) == len(labels)
    assert stats["n_queries_answerable"] == sum(1 for q in queries if q["answerable"])
    assert stats["n_queries_unanswerable"] == sum(1 for q in queries if not q["answerable"])
    assert stats["splits"] == {
        "dev": sum(1 for q in queries if q["split"] == "dev"),
        "test": sum(1 for q in queries if q["split"] == "test"),
    }
    assert len({q["id"] for q in queries}) == len(queries)
    assert {q["split"] for q in queries} == {"dev", "test"}


def test_all_labelled_chunks_exist_in_the_corpus() -> None:
    chunk_ids = {
        json.loads(line)["id"]
        for line in (CORPUS / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for line in (EVAL / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        missing = set(label["relevant_chunk_ids"]) - chunk_ids
        assert not missing, f"{label['query_id']} references missing chunks: {missing}"


def test_reported_quality_is_above_the_shipped_regression_floor() -> None:
    results = _json("results.json")["splits"]
    for split in ("dev", "test"):
        hybrid = results[split]["systems"]["hybrid"]
        gate = results[split]["systems"]["hybrid+rerank-gate"]["refusal"]

        assert hybrid["metrics"]["recall"] >= 0.90
        assert hybrid["metrics"]["hit"] >= 0.95
        assert gate["gate_accuracy"] >= 0.90
        assert gate["answerable_over_refusal_rate"] <= 0.05

    dev_gate = results["dev"]["systems"]["hybrid+rerank-gate"]["refusal"]
    assert dev_gate["unanswerable_refusal_rate"] >= 0.70


def test_reported_gate_thresholds_match_settings() -> None:
    settings = Settings()
    report = (EVAL / "report.md").read_text(encoding="utf-8")
    assert f"'min_rerank_score': {settings.min_rerank_score}" in report
    calibration = (EVAL / "gate_calibration.md").read_text(encoding="utf-8")
    assert f"| rerank | {settings.min_rerank_score:.3f} |" in calibration


def test_failure_analysis_covers_every_test_query_that_failed() -> None:
    results = _json("results.json")["splits"]["test"]
    errors = (EVAL / "errors.md").read_text(encoding="utf-8")
    for system in ("bm25", "dense", "hybrid", "hybrid+rerank", "hybrid+rerank-gate"):
        rows = results["systems"][system]["per_query"]
        failures = [
            row
            for row in rows
            if row["error_type"] not in {"hit", "correct_refusal"}
        ]
        for row in failures:
            assert row["query_id"] in errors, f"{row['query_id']} missing from errors.md"

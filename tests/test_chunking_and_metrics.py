"""Unit tests for chunking and IR metrics — pure functions, exact expectations."""

from __future__ import annotations

import pytest

from ragkit.chunking import chunk_document, chunk_text
from ragkit.evals.metrics import (
    average_precision,
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
)
from ragkit.types import Document


# -- chunking -----------------------------------------------------------

def test_chunk_text_windows_with_overlap():
    words = " ".join(str(i) for i in range(10))  # "0 1 ... 9"
    chunks = chunk_text(words, chunk_size=4, overlap=1)
    # step = 3 -> starts at 0,3,6; the window at 6 already covers the end,
    # so no orphan trailing chunk is emitted.
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunk_text_empty():
    assert chunk_text("   ", chunk_size=5, overlap=0) == []


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=2, overlap=2)


def test_chunk_document_carries_metadata_and_ids():
    doc = Document(id="doc", text="a b c d e f", title="T", source="s")
    chunks = chunk_document(doc, chunk_size=2, overlap=0)
    assert [c.id for c in chunks] == ["doc#0", "doc#1", "doc#2"]
    assert all(c.doc_id == "doc" and c.title == "T" and c.source == "s" for c in chunks)


# -- IR metrics ---------------------------------------------------------

RETRIEVED = ["a", "b", "c", "d"]  # ranked best-first


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, {"b", "z"}, k=4) == 0.5
    assert recall_at_k(RETRIEVED, {"b"}, k=1) == 0.0  # b is at rank 2
    assert recall_at_k(RETRIEVED, set(), k=4) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, {"a", "c"}, k=4) == 0.5
    assert precision_at_k(RETRIEVED, {"a"}, k=1) == 1.0


def test_mrr_uses_first_relevant_rank():
    assert mrr(RETRIEVED, {"c"}) == pytest.approx(1 / 3)
    assert mrr(RETRIEVED, {"z"}) == 0.0


def test_hit_rate():
    assert hit_rate(RETRIEVED, {"d"}, k=4) == 1.0
    assert hit_rate(RETRIEVED, {"d"}, k=2) == 0.0


def test_average_precision():
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    ap = average_precision(RETRIEVED, {"a", "c"})
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)

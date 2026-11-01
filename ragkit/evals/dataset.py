"""Versioned evaluation set loading.

Queries and chunk-level labels live in separate JSONL files so the question set
can be authored before labels are assigned. The loader joins them and validates
the invariants the metrics rely on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuery:
    id: str
    question: str
    category: str
    language: str
    answerable: bool
    reference_answer: str
    split: str = "test"
    relevant_chunk_ids: tuple[str, ...] = ()
    relevant_section_ids: tuple[str, ...] = ()


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_eval_set(eval_dir: str | Path) -> tuple[list[EvalQuery], str]:
    """Load ``queries.jsonl`` + ``labels.jsonl`` and return (queries, hash)."""
    eval_dir = Path(eval_dir)
    queries_path = eval_dir / "queries.jsonl"
    labels_path = eval_dir / "labels.jsonl"
    queries = _load_jsonl(queries_path)
    labels = {row["query_id"]: row for row in _load_jsonl(labels_path)}

    if len({row["id"] for row in queries}) != len(queries):
        raise ValueError("duplicate query ids in queries.jsonl")
    missing = [row["id"] for row in queries if row["id"] not in labels]
    if missing:
        raise ValueError(f"queries without labels: {missing}")

    out: list[EvalQuery] = []
    for row in queries:
        label = labels[row["id"]]
        chunk_ids = tuple(label.get("relevant_chunk_ids") or ())
        section_ids = tuple(label.get("relevant_section_ids") or ())
        answerable = bool(row["answerable"])
        if answerable and not chunk_ids:
            raise ValueError(f"{row['id']}: answerable query has no chunk labels")
        if not answerable and chunk_ids:
            raise ValueError(f"{row['id']}: unanswerable query has chunk labels")
        out.append(
            EvalQuery(
                id=row["id"],
                question=row["question"],
                category=row["category"],
                language=row["language"],
                answerable=answerable,
                reference_answer=row.get("reference_answer", ""),
                split=row.get("split", "test"),
                relevant_chunk_ids=chunk_ids,
                relevant_section_ids=section_ids,
            )
        )
    out.sort(key=lambda q: q.id)

    digest = hashlib.sha256()
    digest.update(queries_path.read_bytes())
    digest.update(labels_path.read_bytes())
    return out, digest.hexdigest()

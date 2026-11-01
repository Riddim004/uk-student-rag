"""Validate the reviewed labels and write the runtime eval files."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "evidence" / "eval"
CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"

# Stratified dev/test split. Gate thresholds are tuned on dev; final numbers
# are reported on test.
DEV_IDS = {
    "q01", "q03", "q05", "q07", "q09",
    "q11", "q13", "q15", "q17", "q19",
    "q21", "q23", "q25", "q27",
    "q29", "q31", "q33",
    "q35", "q36", "q37", "q38",
    "q43", "q45", "q47", "q49",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    notes = json.loads((EVAL_DIR / "annotation_notes.json").read_text(encoding="utf-8"))
    labels = load_jsonl(EVAL_DIR / "labels.firstpass.jsonl")
    chunks = load_jsonl(CHUNKS_PATH)
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}

    notes_by_id = {note["id"]: note for note in notes["queries"]}
    labels_by_id = {row["query_id"]: row for row in labels}
    if set(notes_by_id) != set(labels_by_id):
        missing = sorted(set(notes_by_id) - set(labels_by_id))
        extra = sorted(set(labels_by_id) - set(notes_by_id))
        raise SystemExit(f"query/label mismatch: missing={missing} extra={extra}")

    errors: list[str] = []
    queries_out: list[dict] = []
    labels_out: list[dict] = []
    for qid, note in notes_by_id.items():
        row = labels_by_id[qid]
        chunk_ids = row["relevant_chunk_ids"]
        section_ids = row["relevant_section_ids"]
        if note["answerable"] and not chunk_ids:
            errors.append(f"{qid}: answerable but no relevant chunks")
        if not note["answerable"] and chunk_ids:
            errors.append(f"{qid}: unanswerable but has relevant chunks")
        for chunk_id in chunk_ids:
            if chunk_id not in chunk_by_id:
                errors.append(f"{qid}: unknown chunk id {chunk_id}")
        queries_out.append(
            {
                "id": qid,
                "question": note["question"],
                "category": note["category"],
                "language": note["language"],
                "answerable": note["answerable"],
                "reference_answer": note["reference_answer"],
                "split": "dev" if qid in DEV_IDS else "test",
            }
        )
        labels_out.append(
            {
                "query_id": qid,
                "relevant_chunk_ids": chunk_ids,
                "relevant_section_ids": section_ids,
            }
        )

    if errors:
        raise SystemExit("\n".join(errors))

    queries_out.sort(key=lambda row: row["id"])
    labels_out.sort(key=lambda row: row["query_id"])
    with (EVAL_DIR / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in queries_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (EVAL_DIR / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in labels_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    categories = Counter(row["category"] for row in queries_out)
    label_counts = [len(row["relevant_chunk_ids"]) for row in labels_out if row["relevant_chunk_ids"]]
    stats = {
        "n_queries": len(queries_out),
        "n_queries_answerable": sum(1 for row in queries_out if row["answerable"]),
        "n_queries_unanswerable": sum(1 for row in queries_out if not row["answerable"]),
        "categories": dict(sorted(categories.items())),
        "splits": {
            "dev": sum(1 for row in queries_out if row["split"] == "dev"),
            "test": sum(1 for row in queries_out if row["split"] == "test"),
        },
        "labels": {
            "min_chunks_per_answerable_query": min(label_counts),
            "max_chunks_per_answerable_query": max(label_counts),
            "mean_chunks_per_answerable_query": round(sum(label_counts) / len(label_counts), 2),
            "queries_with_multiple_chunks": sum(1 for n in label_counts if n > 1),
        },
    }
    (EVAL_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

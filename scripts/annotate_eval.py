"""Locate answer-bearing chunks for the draft eval queries.

This is an annotation aid, not part of retrieval. It normalises markdown, finds
chunks containing each key phrase inside the hinted sections, and writes a
first-pass labels file for human review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.ingest import load_documents_jsonl  # noqa: E402

CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"
NOTES_PATH = ROOT / "evidence" / "eval" / "annotation_notes.json"
OUT_PATH = ROOT / "evidence" / "eval" / "labels.firstpass.jsonl"

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_MARKS = re.compile(r"[*_`>#|]+")


def normalise(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_MARKS.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("\xa0", " ")
    return " ".join(text.lower().split())


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def in_hints(section_id: str, hints: list[str]) -> bool:
    if not hints:
        return True
    return any(section_id.startswith(hint) for hint in hints)


def find_matches(chunks: list[dict], note: dict) -> list[dict]:
    phrases = [normalise(p) for p in note.get("key_phrases") or []]
    hints = note.get("section_hints") or []
    matches = []
    for chunk in chunks:
        if not in_hints(chunk["section_id"], hints):
            continue
        text = normalise(chunk["text"])
        hit = [p for p in phrases if p and p in text]
        if hit:
            matches.append(
                {
                    "chunk_id": chunk["id"],
                    "section_id": chunk["section_id"],
                    "section_path": chunk["section_path"],
                    "matched_phrases": hit,
                    "snippet": chunk["text"][:280].replace("\n", " "),
                }
            )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write first-pass labels")
    args = parser.parse_args()

    notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    chunks = load_chunks()
    rows = []
    problems = 0

    for note in notes["queries"]:
        matches = find_matches(chunks, note)
        if note["answerable"] and not matches:
            problems += 1
            status = "MISSING"
        elif not note["answerable"] and matches:
            problems += 1
            status = "UNEXPECTED_MATCH"
        else:
            status = "ok"
        print(f"\n[{note['id']}] {status} {note['question']}")
        print(f"  category={note['category']} hints={note.get('section_hints')}")
        for match in matches[:8]:
            print(f"  - {match['chunk_id']} | {' > '.join(match['section_path'])}")
            print(f"    matched={match['matched_phrases']} | {match['snippet'][:180]}")
        rows.append(
            {
                "query_id": note["id"],
                "relevant_chunk_ids": sorted({m["chunk_id"] for m in matches}),
                "relevant_section_ids": sorted({m["section_id"] for m in matches}),
                "status": status,
            }
        )

    if args.write:
        with OUT_PATH.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"\nwrote {OUT_PATH}")
    print(f"\nqueries={len(rows)} problems={problems}")


if __name__ == "__main__":
    main()

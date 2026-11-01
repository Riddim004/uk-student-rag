"""Fetch the gov.uk snapshot and build the section/chunk corpus.

Usage:
    python scripts/build_data.py            # fetch live, then build
    python scripts/build_data.py --offline  # rebuild from data/raw/govuk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.chunking import chunk_documents  # noqa: E402
from ragkit.govuk import (  # noqa: E402
    SOURCE_SPECS,
    extract_documents,
    fetch_source,
    load_raw,
    read_raw_manifest,
    save_raw,
    utc_now,
    write_corpus,
    write_raw_manifest,
    write_utf8,
)
from ragkit.types import Document  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "govuk"
CORPUS_DIR = ROOT / "data" / "corpus"


def _fallback_fetched_at(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.replace(microsecond=0).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="do not fetch; use raw snapshot")
    parser.add_argument("--chunk-size", type=int, default=260)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    previous = read_raw_manifest(RAW_DIR)
    entries: list[dict[str, object]] = []
    documents: list[Document] = []

    for spec in SOURCE_SPECS:
        if args.offline:
            content = load_raw(RAW_DIR, spec)
            entry = previous.get(spec.doc_id)
            if entry is None:
                raw_path = RAW_DIR / f"{spec.doc_id}.json"
                entry = {
                    "doc_id": spec.doc_id,
                    "label": spec.label,
                    "base_path": spec.base_path,
                    "api_url": f"https://www.gov.uk/api/content{spec.base_path}",
                    "public_url": f"https://www.gov.uk{spec.base_path}",
                    "title": content.get("title"),
                    "document_type": content.get("document_type"),
                    "content_id": content.get("content_id"),
                    "public_updated_at": content.get("public_updated_at"),
                    "first_published_at": content.get("first_published_at"),
                    "fetched_at": _fallback_fetched_at(raw_path),
                    "content_hash": hashlib.sha256(
                        json.dumps(
                            content, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    "raw_file": raw_path.name,
                }
            fetched_at = str(entry.get("fetched_at") or "")
        else:
            fetched_at = utc_now()
            content = fetch_source(spec)
            entry = save_raw(spec, content, RAW_DIR, fetched_at)

        docs = extract_documents(content, spec, fetched_at)
        if not docs:
            print(f"warning: no documents extracted for {spec.doc_id}", file=sys.stderr)
        documents.extend(docs)
        entries.append(entry)
        print(f"{spec.doc_id:22s} sections={len(docs):2d} words={sum(len(d.text.split()) for d in docs):5d}")

    if not args.offline:
        write_raw_manifest(RAW_DIR, entries)

    manifest = write_corpus(documents, CORPUS_DIR, entries)
    chunks = chunk_documents(documents, args.chunk_size, args.overlap)
    chunks_jsonl = "\n".join(
        json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) for chunk in chunks
    ) + ("\n" if chunks else "")
    write_utf8(CORPUS_DIR / "chunks.jsonl", chunks_jsonl)

    manifest.update(
        {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.overlap,
            "total_chunks": len(chunks),
            "chunk_hash": hashlib.sha256(chunks_jsonl.encode("utf-8")).hexdigest(),
        }
    )
    write_utf8(
        CORPUS_DIR / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    print(
        f"\ncorpus: {manifest['total_documents']} sections, "
        f"{manifest['total_words']} words, {len(chunks)} chunks\n"
        f"hash: {manifest['corpus_hash']}\n"
        f"output: {CORPUS_DIR}"
    )


if __name__ == "__main__":
    main()

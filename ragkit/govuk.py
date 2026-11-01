"""gov.uk Content API acquisition and section extraction.

The snapshot is deliberately boring and reproducible: raw API JSON is kept on
disk, extracted sections are emitted as JSONL, and every source records its
URL, publication update date, fetch time and content hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Document

CONTENT_API = "https://www.gov.uk/api/content"
PUBLIC_SITE = "https://www.gov.uk"
USER_AGENT = (
    "ragkit-evidence/1.0 (portfolio reproducibility project; "
    "https://github.com/Riddim004/uk-student-rag)"
)


@dataclass(frozen=True)
class SourceSpec:
    doc_id: str
    base_path: str
    label: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("student-visa", "/student-visa", "Student visa"),
    SourceSpec("graduate-visa", "/graduate-visa", "Graduate visa"),
    SourceSpec("ihs", "/healthcare-immigration-application", "Immigration health surcharge"),
    SourceSpec("evisa", "/evisa", "eVisa"),
    SourceSpec("brp", "/biometric-residence-permits", "Biometric residence permits"),
    SourceSpec(
        "atas",
        "/guidance/academic-technology-approval-scheme",
        "Academic Technology Approval Scheme",
    ),
    SourceSpec(
        "visa-processing-times",
        "/guidance/visa-processing-times-applications-outside-the-uk",
        "Visa processing times",
    ),
    SourceSpec(
        "financial-evidence",
        "/guidance/financial-evidence-for-student-and-child-student-route-applicants",
        "Financial evidence for Student applicants",
    ),
    SourceSpec(
        "immigration-rules-student",
        "/guidance/immigration-rules/immigration-rules-appendix-student",
        "Immigration Rules: Appendix Student",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def content_hash(data: Any) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fetch_source(spec: SourceSpec, timeout: float = 30.0) -> dict[str, Any]:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    response = session.get(f"{CONTENT_API}{spec.base_path}", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"unexpected response for {spec.base_path}: {type(data)}")
    return data


def slugify(text: str, max_length: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text or "section")[:max_length].strip("-")


def _unique_slug(slug: str, seen: set[str]) -> str:
    candidate = slug
    n = 2
    while candidate in seen:
        candidate = f"{slug}-{n}"
        n += 1
    seen.add(candidate)
    return candidate


def markdown_from_html(html: str) -> str:
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return markdownify(str(soup), heading_style="ATX", bullets="-")


_BOILERPLATE = re.compile(
    r"^\s*(?:print this page|back to top|contents|skip to main content)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_EMPTY_LINK = re.compile(r"\[[^\]]*\]\(\s*\)")


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = _IMAGE.sub("", text)
    text = _EMPTY_LINK.sub("", text)
    text = _BOILERPLATE.sub("", text)
    text = re.sub(r"\]\((/[^)]+)\)", r"](https://www.gov.uk\1)", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_h2(markdown: str) -> list[tuple[str | None, str]]:
    """Split markdown into an intro plus one piece per H2 section."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in markdown.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), [line]))
        else:
            sections[-1][1].append(line)
    out: list[tuple[str | None, str]] = []
    for title, lines in sections:
        body = "\n".join(lines).strip()
        if len(body) >= 40:
            out.append((title, body))
    return out


def extract_documents(
    content: dict[str, Any],
    spec: SourceSpec,
    fetched_at: str,
) -> list[Document]:
    """Turn one Content API response into section-level Documents."""
    doc_title = (content.get("title") or spec.label).strip()
    page_url = f"{PUBLIC_SITE}{content.get('base_path') or spec.base_path}"
    updated_at = content.get("public_updated_at") or ""
    source_type = content.get("document_type") or ""
    details = content.get("details") or {}
    parts = details.get("parts") or []

    documents: list[Document] = []
    if parts:
        for i, part in enumerate(parts):
            text = clean_markdown(markdown_from_html(part.get("body") or ""))
            if len(text) < 40:
                continue
            part_title = (part.get("title") or "").strip() or f"Part {i + 1}"
            part_slug = (part.get("slug") or slugify(part_title)).strip()
            url = page_url if part_slug == "overview" else f"{page_url}/{part_slug}"
            section_id = f"{spec.doc_id}__{part_slug}"
            documents.append(
                Document(
                    id=section_id,
                    text=text,
                    title=doc_title,
                    source=url,
                    doc_id=spec.doc_id,
                    section_id=section_id,
                    section_path=(doc_title, part_title),
                    updated_at=updated_at,
                    fetched_at=fetched_at,
                    source_type=source_type,
                    ordinal=i,
                )
            )
        return documents

    text = clean_markdown(markdown_from_html(details.get("body") or ""))
    if len(text) < 40:
        return documents

    seen: set[str] = set()
    for i, (h2_title, section_text) in enumerate(split_h2(text)):
        section_title = h2_title or doc_title
        section_slug = _unique_slug(slugify(section_title), seen)
        section_id = f"{spec.doc_id}__{section_slug}"
        path = (doc_title, section_title) if h2_title else (doc_title,)
        documents.append(
            Document(
                id=section_id,
                text=section_text,
                title=doc_title,
                source=page_url,
                doc_id=spec.doc_id,
                section_id=section_id,
                section_path=path,
                updated_at=updated_at,
                fetched_at=fetched_at,
                source_type=source_type,
                ordinal=i,
            )
        )
    return documents


def write_utf8(path: Path, text: str) -> None:
    """Write UTF-8 bytes with LF newlines on every platform.

    ``Path.write_text`` translates ``\n`` to ``\r\n`` on Windows, which would
    make the file bytes differ from the hashed string. Hashes must identify the
    artifact that is actually on disk, on every OS.
    """
    path.write_bytes(text.encode("utf-8"))


def save_raw(
    spec: SourceSpec,
    content: dict[str, Any],
    raw_dir: Path,
    fetched_at: str,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{spec.doc_id}.json"
    write_utf8(
        path,
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "doc_id": spec.doc_id,
        "label": spec.label,
        "base_path": spec.base_path,
        "api_url": f"{CONTENT_API}{spec.base_path}",
        "public_url": f"{PUBLIC_SITE}{spec.base_path}",
        "title": content.get("title"),
        "document_type": content.get("document_type"),
        "content_id": content.get("content_id"),
        "public_updated_at": content.get("public_updated_at"),
        "first_published_at": content.get("first_published_at"),
        "fetched_at": fetched_at,
        "content_hash": content_hash(content),
        "raw_file": path.name,
    }


def load_raw(raw_dir: Path, spec: SourceSpec) -> dict[str, Any]:
    path = raw_dir / f"{spec.doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing raw snapshot for {spec.doc_id}: {path}. Run without --offline first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_raw_manifest(raw_dir: Path) -> dict[str, dict[str, Any]]:
    path = raw_dir / "manifest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["doc_id"]: entry for entry in data.get("sources", [])}


def write_raw_manifest(raw_dir: Path, entries: list[dict[str, Any]]) -> None:
    payload = {"generated_at": utc_now(), "sources": entries}
    write_utf8(
        raw_dir / "manifest.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def documents_jsonl(documents: list[Document]) -> str:
    lines = [
        json.dumps(asdict(doc), ensure_ascii=False, sort_keys=True) for doc in documents
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_corpus(
    documents: list[Document],
    corpus_dir: Path,
    source_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = corpus_dir / "sources"
    sources_dir.mkdir(exist_ok=True)

    jsonl = documents_jsonl(documents)
    write_utf8(corpus_dir / "documents.jsonl", jsonl)

    by_source: dict[str, list[Document]] = {}
    for doc in documents:
        by_source.setdefault(doc.doc_id, []).append(doc)
    for doc_id, docs in by_source.items():
        header = (
            f"# {docs[0].title}\n\n"
            f"Source: {docs[0].source}\n"
            f"Published/updated: {docs[0].updated_at}\n"
            f"Fetched: {docs[0].fetched_at}\n\n---\n\n"
        )
        body = "\n\n---\n\n".join(doc.text for doc in docs)
        write_utf8(sources_dir / f"{doc_id}.md", header + body + "\n")

    entry_by_id = {entry["doc_id"]: entry for entry in source_entries}
    source_summary = []
    for doc_id, docs in by_source.items():
        entry = entry_by_id.get(doc_id, {})
        source_summary.append(
            {
                "doc_id": doc_id,
                "title": docs[0].title,
                "url": docs[0].source,
                "document_type": docs[0].source_type,
                "public_updated_at": docs[0].updated_at,
                "fetched_at": docs[0].fetched_at,
                "content_hash": entry.get("content_hash"),
                "documents": len(docs),
                "words": sum(len(doc.text.split()) for doc in docs),
            }
        )

    manifest = {
        "generated_at": utc_now(),
        "total_documents": len(documents),
        "total_words": sum(len(doc.text.split()) for doc in documents),
        "corpus_hash": hashlib.sha256(jsonl.encode("utf-8")).hexdigest(),
        "sources": source_summary,
    }
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest

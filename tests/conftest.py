"""Shared test fixtures: a tiny corpus and a fake chat model. No API, no network."""

from __future__ import annotations

import pytest

from ragkit.config import Settings
from ragkit.embeddings import HashingEmbedder
from ragkit.ingest import build_corpus
from ragkit.types import Document

# A small, well-separated corpus so retrieval behaviour is predictable.
DOCS = [
    Document(
        id="working",
        title="Working during your studies",
        text=(
            "Students on a full-time degree course may work up to 20 hours per week "
            "during term time and full time during holidays. Term-time hours are a "
            "strict weekly limit and cannot be averaged. Self-employment is not allowed."
        ),
    ),
    Document(
        id="graduate_route",
        title="Graduate Route",
        text=(
            "The Graduate Route lets graduates stay in the UK after completing a course. "
            "Undergraduate and master's graduates get two years; PhD graduates get three "
            "years. No job offer or minimum salary is required and it cannot be extended."
        ),
    ),
    Document(
        id="finance",
        title="Financial requirement",
        text=(
            "Applicants must show maintenance funds to cover fees and living costs. "
            "The money must be held for a continuous 28 day period before applying. "
            "Bank statements or a sponsorship letter are acceptable evidence."
        ),
    ),
    Document(
        id="dependants",
        title="Bringing dependants",
        text=(
            "A dependant is your partner or child. Postgraduate students on longer "
            "courses may bring dependants, who each pay the health surcharge and show "
            "extra maintenance funds. Partners must show a genuine relationship."
        ),
    ),
]


@pytest.fixture()
def settings() -> Settings:
    # Small chunk size so multi-chunk documents actually occur in tests.
    return Settings(chunk_size=30, chunk_overlap=5, top_k=5, candidate_k=10)


@pytest.fixture()
def corpus(settings: Settings):
    return build_corpus(DOCS, HashingEmbedder(dim=128), settings)


class FakeChatModel:
    """Records the last prompt and returns a configurable reply."""

    def __init__(self, reply: str = "The answer is grounded [1].") -> None:
        self.reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.reply

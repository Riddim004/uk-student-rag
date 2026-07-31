"""The labelled evaluation set: questions with their relevant document ids.

Relevance is labelled at the document level (which doc *should* answer this),
which is cheap to author and stable across chunking changes. Keep this in sync
with the files under ``data/uk_student_visa/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuery:
    question: str
    relevant_doc_ids: frozenset[str] = field(default_factory=frozenset)


DEFAULT_EVAL_SET: tuple[EvalQuery, ...] = (
    EvalQuery(
        "How many hours can I work during term time on a Student visa?",
        frozenset({"working"}),
    ),
    EvalQuery(
        "How long can I stay in the UK after graduating?",
        frozenset({"graduate_route"}),
    ),
    EvalQuery(
        "How much money do I need to show for maintenance?",
        frozenset({"finance"}),
    ),
    EvalQuery(
        "Can I bring my partner or child as a dependant?",
        frozenset({"dependants"}),
    ),
    EvalQuery(
        "When should I apply and how long does a decision take?",
        frozenset({"applying"}),
    ),
    EvalQuery(
        "Do I need to prove my English language ability?",
        frozenset({"eligibility"}),
    ),
)

"""Language-aware tokenisation for lexical retrieval.

English words are kept as words. CJK text has no spaces, so a word regex would
drop it entirely; here each CJK character becomes a unigram and adjacent
characters also become bigrams. Bigrams are the smallest useful unit for
Chinese lexical matching without pulling in a segmentation dependency.
"""

from __future__ import annotations

import re
import unicodedata

_LATIN = re.compile(r"[a-z0-9]+(?:['’\-][a-z0-9]+)*")
_CJK_RUN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+"
)


def tokenize(text: str) -> list[str]:
    """Return lower-cased Latin words plus CJK unigrams and bigrams."""
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = _LATIN.findall(text)
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens

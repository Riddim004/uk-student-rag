"""Exception taxonomy.

Failures are grouped by what a caller can do about them:

* ``DataError`` — the corpus or eval set is missing/corrupt; rebuild the data.
* ``ConfigurationError`` — an invalid setting or backend name; fix the config.
* ``ModelError`` — an embedding/rerank/chat backend failed.
  - ``TransientModelError`` is worth retrying (timeout, 429, 5xx).
  - ``GenerationError`` means the answer stage failed after retries; callers
    must surface it instead of pretending the model refused.
"""

from __future__ import annotations


class RagkitError(Exception):
    """Base class for all expected, user-actionable failures."""


class ConfigurationError(RagkitError):
    """Invalid settings or an unknown backend."""


class DataError(RagkitError):
    """The corpus or evaluation data is missing, empty, or malformed."""


class ModelError(RagkitError):
    """An embedding, rerank, or chat model failed."""


class TransientModelError(ModelError):
    """A model call failed in a way that may succeed on retry."""


class EmbeddingError(ModelError):
    """The embedding backend returned an unusable result."""


class GenerationError(ModelError):
    """The answer stage failed after retries."""


_TRANSIENT_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
    "TimeoutError",
    "ConnectionError",
}


def is_transient(exc: BaseException) -> bool:
    """Best-effort classification for network/backend errors.

    Works by class name so the OpenAI SDK is not required at import time.
    """
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


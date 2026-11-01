"""Small retry helper with exponential backoff.

Deliberately dependency-free and injectable (``sleep``/``rng``) so retry
behaviour can be tested without waiting.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.1,
    retry_if: Callable[[BaseException], bool] = lambda _exc: True,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call ``fn`` until it succeeds, ``attempts`` is exhausted, or
    ``retry_if`` rejects the exception.

    The final failure is re-raised unchanged so callers keep the original
    traceback and error type.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    rng = rng or random.Random()
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not retry_if(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 1.0 + rng.uniform(-jitter, jitter)
            sleep(max(0.0, delay))
    raise AssertionError("unreachable")

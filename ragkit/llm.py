"""OpenAI-compatible chat model with explicit timeouts and bounded retries."""

from __future__ import annotations

from .config import Settings
from .errors import GenerationError, TransientModelError, is_transient
from .retry import retry_call


class OpenAIChatModel:
    """Thin wrapper over an OpenAI-compatible chat completions endpoint.

    The SDK's own retries are disabled so this class owns the policy: a bounded
    number of attempts, exponential backoff, and a clear ``GenerationError``
    once the budget is exhausted. Non-transient failures (bad request, auth)
    are never retried.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        from openai import OpenAI  # imported lazily so tests need no dependency

        self._settings = settings or Settings.from_env()
        kwargs = {
            "timeout": self._settings.request_timeout_s,
            "max_retries": 0,
        }
        if self._settings.api_base:
            kwargs["base_url"] = self._settings.api_base
        self._client = OpenAI(**kwargs)

    def complete(self, system: str, user: str) -> str:
        settings = self._settings

        def call() -> str:
            try:
                resp = self._client.chat.completions.create(
                    model=settings.chat_model,
                    temperature=settings.temperature,
                    max_tokens=settings.max_output_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as exc:
                if is_transient(exc):
                    raise TransientModelError(
                        f"chat request failed: {type(exc).__name__}"
                    ) from exc
                raise GenerationError(
                    f"chat request rejected: {type(exc).__name__}: {exc}"
                ) from exc
            text = resp.choices[0].message.content or ""
            if not text.strip():
                raise TransientModelError("chat model returned an empty response")
            return text.strip()

        try:
            return retry_call(
                call,
                attempts=settings.max_retries + 1,
                base_delay=settings.retry_base_delay_s,
                retry_if=lambda exc: isinstance(exc, TransientModelError),
            )
        except TransientModelError as exc:
            raise GenerationError(
                f"chat model unavailable after {settings.max_retries + 1} attempts: {exc}"
            ) from exc

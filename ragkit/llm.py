"""OpenAI-compatible chat model implementing the ``ChatModel`` protocol."""

from __future__ import annotations

from .config import Settings


class OpenAIChatModel:
    """Thin wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        from openai import OpenAI  # imported lazily so tests need no dependency

        self._settings = settings or Settings.from_env()
        self._client = (
            OpenAI(base_url=self._settings.api_base)
            if self._settings.api_base
            else OpenAI()
        )

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._settings.chat_model,
            temperature=self._settings.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

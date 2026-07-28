"""Ollama LLM provider implementation.

This provider wraps the ``ollama`` Python SDK and implements the
``BaseLLMProvider`` interface.  It is stateless with respect to
conversation history: each ``chat()`` call receives a ``ChatRequest``
and yields text chunks.

The model name is sourced from the ``ChatRequest.model`` field —
never hardcoded — so any Ollama-compatible model can be used by
changing configuration only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import ollama

from src.exceptions import (
    MissingCredentialsError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from src.models.chat import ChatRequest
from src.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Provider that communicates with a local Ollama instance.

    Parameters
    ----------
    settings:
        A validated ``Settings`` instance containing *ollama_host* and
        *ollama_model* values.  The model from settings serves only as
        the default; the active model is taken from each
        ``ChatRequest.model`` at call time so different requests can
        use different models.
    """

    def __init__(self, settings: object) -> None:
        self._host: str = getattr(settings, "ollama_host", "http://localhost:11434")
        self._default_model: str = getattr(settings, "ollama_model", "llama3.1:8b")
        self._client: ollama.Client = ollama.Client(host=self._host)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        """Return ``"ollama"`` as the stable provider identifier."""
        return "ollama"

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    def validate_configuration(self) -> None:
        """Verify that the Ollama host and model name are present.

        This is a local-only check — no network call is made.
        Network reachability is tested by :meth:`is_available`.
        """
        if not self._host:
            raise MissingCredentialsError(
                self.provider_name,
                safe_message="Ollama host is not configured. Set OLLAMA_HOST in your .env file.",
            )
        if not self._default_model:
            raise MissingCredentialsError(
                self.provider_name,
                safe_message="Ollama model is not configured. Set OLLAMA_MODEL in your .env file.",
            )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Uses a lightweight ``list()`` call (no model loading) so it is
        safe to call on every sidebar render.  Returns ``False`` on any
        connection issue without raising.
        """
        try:
            self._client.list()
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Ollama availability check failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Core chat method
    # ------------------------------------------------------------------
    def chat(self, request: ChatRequest) -> Iterator[str]:
        """Stream a response from Ollama.

        Parameters
        ----------
        request:
            A fully populated provider-neutral chat request.

        Yields
        ------
        str
            Normalised non-empty text chunks.

        Raises
        ------
        ProviderConnectionError
            If Ollama cannot be reached.
        ProviderRateLimitError
            If Ollama returns a rate-limit-like error.
        ProviderResponseError
            If the response is unexpected or malformed.
        """
        model = request.model or self._default_model
        ollama_messages = self._build_ollama_messages(request)

        yielded_any = False

        try:
            stream = self._client.chat(
                model=model,
                messages=ollama_messages,
                stream=True,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            )

            for chunk in stream:
                fragment = self._extract_text(chunk)
                if fragment.strip():
                    yielded_any = True
                    yield fragment

        except ollama.ResponseError as exc:
            logger.error(
                "Ollama response error (status=%s): %s",
                exc.status_code,
                exc.error,
            )
            if exc.status_code == 429:
                raise ProviderRateLimitError(
                    self.provider_name,
                    safe_message="Ollama is rate-limiting requests. Please wait and try again.",
                ) from exc
            raise ProviderResponseError(
                self.provider_name,
                safe_message=f"Ollama returned an error (status {exc.status_code}). Check the logs for details.",
            ) from exc
        except ConnectionError as exc:
            logger.error("Ollama connection error: %s", exc)
            raise ProviderConnectionError(
                self.provider_name,
                safe_message="Cannot connect to Ollama. Make sure it is installed and running.",
            ) from exc
        except Exception as exc:
            logger.error("Unexpected Ollama error: %s", exc, exc_info=True)
            raise ProviderResponseError(
                self.provider_name,
                safe_message="An unexpected error occurred while communicating with Ollama.",
            ) from exc

        if not yielded_any:
            raise ProviderResponseError(
                self.provider_name,
                safe_message="Provider returned an empty response.",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ollama_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        """Convert a ``ChatRequest`` into the Ollama SDK message format.

        The system prompt is placed as the first message with role
        ``system`` if non-empty.  Remaining messages are mapped by
        role (``user``, ``assistant``).
        """
        ollama_messages: list[dict[str, str]] = []

        # System prompt (first, if non-empty)
        if request.system_prompt:
            ollama_messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )

        # Conversation messages
        for msg in request.messages:
            ollama_messages.append(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                }
            )

        return ollama_messages

    @staticmethod
    def _extract_text(chunk: Any) -> str:
        """Extract and normalise the text fragment from a stream chunk.

        Handles **both** ``ollama.ChatResponse`` / ``ollama.Message``
        objects (returned by ollama>=0.6) **and** plain ``dict``
        responses (older SDK behaviour / test doubles).

        Precedence:
        1. ``chunk.message.content`` (``ChatResponse`` object)
        2. ``chunk.content`` (``Message`` object)
        3. ``chunk["message"]["content"]`` (dict)
        4. ``chunk.get("content", "")`` (dict fallback)
        """
        # 1. ChatResponse object -> .message -> .content
        msg = getattr(chunk, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content

        # 2. Message object -> .content
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content

        # 3. Dict with "message" -> {"content": ...}
        if isinstance(chunk, dict):
            msg = chunk.get("message")
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content

            # 4. Direct "content" key (dict fallback)
            content = chunk.get("content", "")
            if isinstance(content, str):
                return content

        return ""

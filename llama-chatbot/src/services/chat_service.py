"""Chat orchestration — the bridge between the UI and LLM providers.

``ChatService`` owns the conversation flow:
- validates user input
- adds user messages to ``Conversation``
- creates ``ChatRequest`` objects
- calls the selected LLM provider
- collects streamed response chunks
- adds the assistant message to ``Conversation``
- returns the full response text

It depends on ``BaseLLMProvider``, ``Conversation``, and ``ChatRequest``.
It does not depend on any UI framework or provider SDK.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from src.exceptions import (
    ChatbotError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
)
from src.models.chat import ChatRequest, Conversation
from src.prompts.system_prompts import get_default_prompt
from src.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates the chat flow between user and LLM provider.

    Parameters
    ----------
    provider:
        An LLM provider instance implementing ``BaseLLMProvider``.
    model:
        The model identifier (e.g. ``"llama3.1:8b"``, ``"llama-3.1-70b-versatile"``).
        This value is passed into every ``ChatRequest``.
    system_prompt:
        Optional override for the default system prompt.
    temperature:
        Sampling temperature (0.0 - 2.0).  Default 0.7.
    max_tokens:
        Maximum tokens in the response.  Default 2048.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        if not isinstance(provider, BaseLLMProvider):
            raise TypeError(
                f"Expected a BaseLLMProvider instance, got {type(provider).__name__}"
            )
        self._provider: BaseLLMProvider = provider
        self._model: str = model
        self._system_prompt: str = system_prompt or get_default_prompt()
        self._temperature: float = temperature
        self._max_tokens: int = max_tokens

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def provider(self) -> BaseLLMProvider:
        """The currently selected LLM provider."""
        return self._provider

    @property
    def model(self) -> str:
        """The model name sent with each request."""
        return self._model

    @property
    def system_prompt(self) -> str:
        """The system prompt used for chat requests."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Update the system prompt for subsequent requests."""
        self._system_prompt = value or get_default_prompt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_message(self, conversation: Conversation, content: str) -> str:
        """Send a user message and return the full assistant response.

        This method:
        1. Validates the user input (non-empty).
        2. Adds the user message to *conversation*.
        3. Creates a ``ChatRequest`` with the current settings.
        4. Calls ``provider.chat(request)`` and collects all chunks.
        5. Adds the assistant message to *conversation*.
        6. Returns the combined response text.

        Parameters
        ----------
        conversation:
            The conversation to append messages to.
        content:
            The user's message text.

        Returns
        -------
        str
            The complete assistant response.

        Raises
        ------
        ConfigurationError
            If the user input is empty.
        ProviderConnectionError
            If the provider cannot be reached.
        ProviderResponseError
            If the provider returns an unexpected response.
        ChatbotError
            For any other application-level error.
        """
        self._validate_input(content)

        # 1. Add user message
        conversation.add_user_message(content)

        # 2. Build request
        request = self._build_request(conversation)

        # 3. Stream and combine
        full_response: str = ""
        try:
            for chunk in self._provider.chat(request):
                full_response += chunk
        except (ProviderConnectionError, ProviderResponseError):
            raise
        except ChatbotError:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error during provider.chat(): %s", exc, exc_info=True
            )
            raise ProviderResponseError(
                self._provider.provider_name,
                safe_message="An unexpected error occurred while generating a response.",
            ) from exc

        # 4. Add assistant message
        conversation.add_assistant_message(full_response)

        return full_response

    def stream_message(self, conversation: Conversation, content: str) -> Iterator[str]:
        """Send a user message and yield response chunks as they arrive.

        This method:
        1. Validates the user input (non-empty).
        2. Adds the user message to *conversation*.
        3. Creates a ``ChatRequest`` with the current settings.
        4. Yields each non-empty chunk from ``provider.chat(request)``.
        5. After all chunks have been consumed, adds the combined
           assistant message to *conversation*.

        Parameters
        ----------
        conversation:
            The conversation to append messages to.
        content:
            The user's message text.

        Yields
        ------
        str
            Non-empty text chunks from the provider.

        Raises
        ------
        ConfigurationError
            If the user input is empty.
        ProviderConnectionError
            If the provider cannot be reached.
        ProviderResponseError
            If the provider returns an unexpected response.
        """
        self._validate_input(content)

        # 1. Add user message
        conversation.add_user_message(content)

        # 2. Build request
        request = self._build_request(conversation)

        # 3. Stream and yield
        full_response: str = ""
        try:
            for chunk in self._provider.chat(request):
                if chunk.strip():
                    full_response += chunk
                    yield chunk
        except (ProviderConnectionError, ProviderResponseError):
            raise
        except ChatbotError:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error during provider.chat(): %s", exc, exc_info=True
            )
            raise ProviderResponseError(
                self._provider.provider_name,
                safe_message="An unexpected error occurred while generating a response.",
            ) from exc

        # 4. Add assistant message after streaming completes
        conversation.add_assistant_message(full_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_input(self, content: str) -> None:
        """Ensure user input is non-empty.

        Parameters
        ----------
        content:
            The user message to validate.

        Raises
        ------
        ConfigurationError
            If *content* is empty or whitespace-only.
        """
        if not content or not content.strip():
            raise ConfigurationError(
                "User message cannot be empty.",
                safe_message="Please enter a message before sending.",
            )

    def _build_request(self, conversation: Conversation) -> ChatRequest:
        """Build a ``ChatRequest`` from the conversation and current settings.

        Parameters
        ----------
        conversation:
            The current conversation (user + assistant messages only).

        Returns
        -------
        ChatRequest
            A ready-to-send request with the system prompt in its
            own field (not duplicated into *messages*).
        """
        return ChatRequest(
            messages=conversation.messages,
            model=self._model,
            system_prompt=self._system_prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

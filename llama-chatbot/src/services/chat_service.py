"""Chat orchestration — the bridge between the UI and LLM providers.

``ChatService`` owns the conversation flow:
- validates user input
- optionally retrieves RAG context via ``RAGService``
- adds user messages to ``Conversation``
- creates ``ChatRequest`` objects
- calls the selected LLM provider
- collects streamed response chunks
- adds the assistant message to ``Conversation``
- returns the full response text

It depends on ``BaseLLMProvider``, ``Conversation``, and ``ChatRequest``.
When RAG is enabled, it also depends on ``RAGService`` (optional).
It does not depend on any UI framework or provider SDK.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from src.exceptions import (
    ChatbotError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
)
from src.models.chat import ChatRequest, Conversation, SourceRef
from src.prompts.system_prompts import get_default_prompt
from src.providers.base import BaseLLMProvider
from src.rag.citations import build_citations_section, build_source_refs

if TYPE_CHECKING:
    from src.rag.rag_service import RAGService

logger = logging.getLogger(__name__)

#: Template for prepending RAG context to the user message.
_RAG_CONTEXT_TEMPLATE: str = "Relevant context:\n{context}\n\nQuestion:\n{query}"


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
    rag_service:
        Optional ``RAGService`` instance for retrieval-augmented generation.
        When provided, retrieved document context is prepended to the user
        message before sending to the LLM.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        rag_service: RAGService | None = None,
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
        self._rag_service: RAGService | None = rag_service
        #: The most recent ``RAGResult`` captured during RAG enrichment.
        #: ``None`` when RAG is disabled, failed, or not yet run.  Retained
        #: for internal debugging / RAG evaluation (document_id, chunk_index,
        #: score, page_number, filename).
        self._last_rag_result: Any = None

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

    @property
    def rag_service(self) -> RAGService | None:
        """The optional ``RAGService`` instance, or ``None`` if RAG is
        not enabled."""
        return self._rag_service

    @property
    def last_rag_result(self) -> Any:
        """The most recent ``RAGResult`` captured during RAG enrichment.

        Returns ``None`` when RAG is disabled, failed, or not yet run.
        The result preserves full retrieval metadata (``document_id``,
        ``filename``, ``chunk_index``, ``page_number``, ``score``) for
        internal debugging / RAG evaluation.
        """
        return self._last_rag_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_message(self, conversation: Conversation, content: str) -> str:
        """Send a user message and return the full assistant response.

        This method:
        1. Validates the user input (non-empty).
        2. Optionally retrieves RAG context and prepends it to the user
           message.
        3. Adds the user message to *conversation*.
        4. Creates a ``ChatRequest`` with the current settings.
        5. Calls ``provider.chat(request)`` and collects all chunks.
        6. Adds the assistant message (with structured sources, if any)
           to *conversation*.
        7. Returns the combined response text.

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

        # 1. Optionally retrieve RAG context
        enriched_content = self._enrich_with_context(content)

        # 2. Add user message
        conversation.add_user_message(enriched_content)

        # 3. Build request
        request = self._build_request(conversation)

        # 4. Stream and combine
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

        # 5. Append backend-generated citations (from retrieved metadata)
        final_response, sources = self._build_cited_response(full_response)

        # 6. Add assistant message (with structured sources when present)
        conversation.add_assistant_message(final_response, sources=sources)

        return final_response

    def stream_message(self, conversation: Conversation, content: str) -> Iterator[str]:
        """Send a user message and yield response chunks as they arrive.

        This method:
        1. Validates the user input (non-empty).
        2. Optionally retrieves RAG context and prepends it to the user
           message.
        3. Adds the user message to *conversation*.
        4. Creates a ``ChatRequest`` with the current settings.
        5. Yields each non-empty chunk from ``provider.chat(request)``.
        6. After all chunks have been consumed, adds the combined
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

        # 1. Optionally retrieve RAG context
        enriched_content = self._enrich_with_context(content)

        # 2. Add user message
        conversation.add_user_message(enriched_content)

        # 3. Build request
        request = self._build_request(conversation)

        # 4. Stream and yield
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

        # 5. Append backend-generated citations (from retrieved metadata)
        final_response, sources = self._build_cited_response(full_response)

        # 6. Add assistant message after streaming completes
        conversation.add_assistant_message(final_response, sources=sources)

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

    def _enrich_with_context(self, content: str) -> str:
        """Optionally retrieve RAG context and prepend it to the user query.

        When ``rag_service`` is configured, this method retrieves
        relevant document context and formats it into the user message.
        When ``rag_service`` is ``None``, the original content is
        returned unchanged.

        Parameters
        ----------
        content:
            The original user message.

        Returns
        -------
        str
            The enriched message (with context) if RAG is enabled, or
            the original message if not.
        """
        if self._rag_service is None:
            self._last_rag_result = None
            return content

        try:
            rag_result = self._rag_service.query(content)
            self._last_rag_result = rag_result
            if rag_result.context:
                return _RAG_CONTEXT_TEMPLATE.format(
                    context=rag_result.context,
                    query=content.strip(),
                )
        except ChatbotError:
            self._last_rag_result = None
            logger.warning("RAG query failed, falling back to original message")
        except Exception as exc:
            self._last_rag_result = None
            logger.warning(
                "Unexpected error during RAG query: %s", exc, exc_info=True
            )

        return content

    def _build_cited_response(
        self,
        response: str,
    ) -> tuple[str, list[SourceRef] | None]:
        """Append a backend-generated ``Sources:`` section and build sources.

        Citations are derived from the metadata of the most recent RAG
        retrieval (filename + optional page number) — **not** from the
        LLM output.  The section and source refs are omitted when:

        - RAG is disabled (``rag_service`` is ``None``)
        - the RAG query failed
        - retrieval returned no chunks

        Internal metadata (``document_id``, ``chunk_index``, Qdrant point
        IDs, similarity scores) is intentionally not shown to the user.

        Parameters
        ----------
        response:
            The raw LLM response text.

        Returns
        -------
        tuple[str, list[SourceRef] | None]
            A ``(text, sources)`` pair.  ``text`` is the response with the
            citations block appended when retrieved chunks exist, otherwise
            the response unchanged.  ``sources`` is the structured list of
            source refs (with chunk text and score kept internally), or
            ``None`` when no usable chunks exist.
        """
        rag_result = self._last_rag_result
        if rag_result is None:
            return response, None

        retrieval_result = getattr(rag_result, "retrieval_result", None)
        if retrieval_result is None:
            return response, None

        chunks = getattr(retrieval_result, "chunks", None)
        sources = build_source_refs(chunks)
        if not sources:
            return response, None

        citations = build_citations_section(chunks)
        if not citations:
            return response, sources

        return f"{response}{citations}", sources

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

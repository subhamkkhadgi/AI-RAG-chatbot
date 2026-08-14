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
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from src.exceptions import (
    ChatbotError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
)
from src.models.chat import ChatRequest, ChatRole, Conversation, SourceRef
from src.prompts.system_prompts import get_default_prompt
from src.providers.base import BaseLLMProvider
from src.rag.citations import build_citations_section, build_source_refs
from src.rag.confidence import is_confident
from src.rag.evidence import filter_supporting_chunks
from src.services.response_formatter import strip_attribution

if TYPE_CHECKING:
    from src.rag.rag_service import RAGService

logger = logging.getLogger(__name__)

#: Template for prepending RAG context to the user message.
_RAG_CONTEXT_TEMPLATE: str = "<retrieved_context>\nThe following is reference material from documents.\nUse only the factual information.\nDo not copy the document's formatting, numbering style, headings, or list markers — especially malformed list symbols like *.\nWrite any list with each item on its own separate line using normal Markdown bullets `- `, for example \"- item\". Never join list items with \"*\" or other symbols. Preserve the actual information from the retrieved context.\n\n{context}\n</retrieved_context>\n\nQuestion:\n{query}"


#: Generic words that carry little topical signal.  A question containing
#: almost only these is treated as a generic/referential follow-up.
_GENERIC_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "can", "could", "did",
        "do", "does", "for", "from", "how", "in", "is", "of", "on", "or",
        "that", "the", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "why", "will", "with", "would", "should",
    }
)

#: Minimum number of topical (non-generic) words a question must carry to
#: be considered self-contained.  Below this (with a prior user question)
#: the question is treated as a generic referential follow-up.
_MIN_TOPIC_WORDS: int = 3

#: Regular expression splitting text into lowercase alphanumeric tokens.
_TOKEN_RE: "re.Pattern[str]" = re.compile(r"[a-z0-9]+")

#: Separator used to combine a previous user question with the current
#: follow-up when building a contextualized retrieval query.
_RETRIEVAL_JOIN: str = " — "

#: Marker separating the RAG context block from the original question in a
#: stored (enriched) user message.  Used by :func:`_most_recent_user_question`
#: to recover the original question so the previous context is never polluted.
_RAG_QUESTION_MARKER: str = "\n\nQuestion:\n"


def build_retrieval_query(
    content: str,
    conversation: "Conversation | None",
) -> str:
    """Build the query to send to RAG for *content*.

    Explicit / self-contained questions are returned unchanged.  For a
    clearly generic / referential follow-up (one that lacks enough topic
    information on its own **and** has a prior user message), the most
    recent previous **user** question is combined with the current
    question so retrieval can identify the document/topic being discussed.

    Only the query sent to RAG is changed; the user message stored in the
    conversation is never altered.  Previous assistant answers are never
    used, and the full conversation is never included.

    Parameters
    ----------
    content:
        The current user message (the question to send).
    conversation:
        The conversation to read the most recent previous user question
        from, or ``None`` (treated as empty).

    Returns
    -------
    str
        The retrieval query — the original *content* when it is explicit
        or self-contained, or a contextualized combination otherwise.
    """
    current = content.strip()
    if not current:
        return content

    previous_question = _most_recent_user_question(conversation)
    if previous_question is None:
        # No prior user message — nothing to contextualize against.
        return content

    if _is_self_contained(current):
        # The current question already carries its own topic information;
        # do not inject context (keeps explicit / independent questions
        # byte-for-byte unchanged).
        return content

    # The current question is generic / referential and a prior user
    # question exists.  Combine the previous question (for topic context)
    # with the current one (to preserve the current intent).
    return f"{previous_question.strip()}{_RETRIEVAL_JOIN}{current}"


def _most_recent_user_question(
    conversation: "Conversation | None",
) -> str | None:
    """Return the most recent previous *original* user question, or ``None``.

    Scans the conversation from the end backwards and returns the content
    of the last message whose role is ``USER``.  Assistant answers and any
    other roles are ignored.  Because a stored user message may be the
    RAG-enriched form (``<retrieved_context>...\\n\\nQuestion:\\n<original>``),
    the original question is recovered by keeping only the text after the
    ``Question:`` marker so the previous context is never polluted.
    """
    if conversation is None:
        return None
    for msg in reversed(conversation.messages):
        if getattr(msg, "role", None) == ChatRole.USER:
            content = msg.content
            marker = content.rfind(_RAG_QUESTION_MARKER)
            if marker != -1:
                return content[marker + len(_RAG_QUESTION_MARKER):]
            return content
    return None


def _is_self_contained(content: str) -> bool:
    """Decide whether *content* carries enough topic information on its own.

    A question is considered self-contained (and therefore **not**
    rewritten) when it contains at least :data:`_MIN_TOPIC_WORDS` topical
    (non-generic) words.  Short generic questions such as "What
    technologies are used?" carry almost no topical words and are treated
    as referential, while questions that name a topic (enough content
    words) are left unchanged.

    Parameters
    ----------
    content:
        The candidate question text.

    Returns
    -------
    bool
        ``True`` when the question carries enough topic information to be
        self-contained; ``False`` when it is likely a generic follow-up.
    """
    if not content:
        return True
    tokens = _TOKEN_RE.findall(content.strip().lower())
    topic_words = [t for t in tokens if t not in _GENERIC_WORDS]
    return len(topic_words) >= _MIN_TOPIC_WORDS


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
        message before sending to the LLM (only when retrieval confidence
        meets :attr:`confidence_threshold`).
    confidence_threshold:
        Optional minimum retrieval confidence (highest chunk similarity
        score) required to inject retrieved context into the prompt.
        ``None`` restores the pre-Sprint-9B behaviour — retrieved context
        is always injected when available.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        rag_service: RAGService | None = None,
        confidence_threshold: float | None = None,
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
        #: Minimum retrieval confidence required to use retrieved context.
        self._confidence_threshold: float | None = confidence_threshold
        #: The most recent ``RAGResult`` captured during RAG enrichment.
        #: ``None`` when RAG is disabled, failed, or not yet run.  Retained
        #: for internal debugging / RAG evaluation (document_id, chunk_index,
        #: score, page_number, filename).
        self._last_rag_result: Any = None
        #: Whether the most recent RAG context was actually injected into
        #: the prompt.  ``False`` when the context failed confidence gating.
        self._last_context_injected: bool = False

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
    def confidence_threshold(self) -> float | None:
        """The minimum retrieval confidence required to use retrieved context."""
        return self._confidence_threshold

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
        enriched_content = self._enrich_with_context(content, conversation)

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

        # 6. Clean retrieval-style attribution wording from the displayed
        #    answer.  Applied after citation processing so citation filtering
        #    still sees the original raw answer.
        final_response = strip_attribution(final_response)

        # 7. Add assistant message (with structured sources when present)
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
        enriched_content = self._enrich_with_context(content, conversation)

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

        # 6. Clean retrieval-style attribution wording from the displayed
        #    answer.  Applied after citation processing so citation filtering
        #    still sees the original raw answer.
        final_response = strip_attribution(final_response)

        # 7. Add assistant message after streaming completes
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

    def _enrich_with_context(
        self,
        content: str,
        conversation: Conversation,
    ) -> str:
        """Optionally retrieve RAG context and prepend it to the user query.

        When ``rag_service`` is configured, this method retrieves
        relevant document context and formats it into the user message.
        When ``rag_service`` is ``None``, the original content is
        returned unchanged.

        The query sent to retrieval may be contextualized for an ambiguous
        follow-up via :func:`build_retrieval_query`; the *content* itself
        (the user message) is never altered.

        Parameters
        ----------
        content:
            The original user message (stored in the conversation
            unchanged).
        conversation:
            The current conversation, used to contextualize a generic
            follow-up for retrieval.

        Returns
        -------
        str
            The enriched message (with context) if RAG is enabled and the
            retrieval confidence meets :attr:`confidence_threshold`, or
            the original message if not.
        """
        if self._rag_service is None:
            self._last_rag_result = None
            self._last_context_injected = False
            return content

        # Build the retrieval query: explicit / self-contained questions
        # stay unchanged; ambiguous follow-ups are combined with the most
        # recent previous user question.
        retrieval_query = build_retrieval_query(content, conversation)

        try:
            rag_result = self._rag_service.query(retrieval_query)
            self._last_rag_result = rag_result
            # Confidence-aware RAG: only inject context when the retrieval
            # confidence meets the configured threshold.  When confidence
            # cannot be determined (e.g. mock results in tests) we fall
            # back to the previous behaviour so existing flows are preserved.
            retrieval_result = getattr(rag_result, "retrieval_result", None)
            # --- TEMP DIAGNOSTIC LOGGING (missing-citations investigation) ---
            _diag_chunks = (
                getattr(retrieval_result, "chunks", None)
                if retrieval_result is not None
                else None
            )
            _diag_chunk_list = (
                _diag_chunks if isinstance(_diag_chunks, (list, tuple)) else []
            )
            _diag_scores = [
                float(getattr(c, "score"))
                for c in _diag_chunk_list
                if isinstance(getattr(c, "score", None), (int, float))
            ]
            logger.debug(
                "RAG_DIAG[chat] pre-injection | query=%r chunk_count=%d "
                "chunks=%r max_score=%r confidence_threshold=%r",
                content,
                len(_diag_chunk_list),
                [
                    (getattr(c, "filename", ""), getattr(c, "score", None))
                    for c in _diag_chunk_list
                ],
                max(_diag_scores) if _diag_scores else None,
                self._confidence_threshold,
            )
            if retrieval_result is not None and is_confident(
                retrieval_result, self._confidence_threshold
            ):
                self._last_context_injected = True
                logger.debug(
                    "RAG_DIAG[chat] injection decision | injected=True query=%r",
                    content,
                )
                if rag_result.context:
                    return _RAG_CONTEXT_TEMPLATE.format(
                        context=rag_result.context,
                        query=content.strip(),
                    )
            else:
                self._last_context_injected = False
                logger.debug(
                    "RAG_DIAG[chat] injection decision | injected=False query=%r",
                    content,
                )
        except ChatbotError:
            self._last_rag_result = None
            self._last_context_injected = False
            logger.warning("RAG query failed, falling back to original message")
        except Exception as exc:
            self._last_rag_result = None
            self._last_context_injected = False
            logger.warning(
                "Unexpected error during RAG query: %s", exc, exc_info=True
            )

        return content

    def _build_cited_response(
        self,
        response: str,
    ) -> tuple[str, list[SourceRef] | None]:
        """Build the assistant response text and structured sources.

        The structured sources are derived from the metadata of the most
        recent RAG retrieval (filename + optional page number) — **not**
        from the LLM output.  The structured ``sources`` are returned so
        the UI can render them as the expandable Sources card.  The
        plain-text ``Sources:`` block is intentionally **not** appended to
        the response text so citations are not displayed twice.

        Sources are omitted when:
        - RAG is disabled (``rag_service`` is ``None``)
        - the RAG query failed
        - retrieval returned no chunks
        - no retrieved chunk lexically supports the generated answer
        - the retrieval confidence fell below :attr:`confidence_threshold`
          (context was not injected into the prompt)

        Internal metadata (``document_id``, ``chunk_index``, Qdrant point
        IDs, similarity scores) is intentionally not shown to the user.

        Parameters
        ----------
        response:
            The raw LLM response text.

        Returns
        -------
        tuple[str, list[SourceRef] | None]
            A ``(text, sources)`` pair.  ``text`` is the raw response text
            unchanged.  ``sources`` is the structured list of source refs
            (with chunk text and score kept internally), or ``None`` when
            no usable chunks exist.
        """
        rag_result = self._last_rag_result
        if rag_result is None:
            return response, None

        retrieval_result = getattr(rag_result, "retrieval_result", None)
        if retrieval_result is None:
            return response, None

        # Only cite sources when the retrieved context was actually injected
        # into the prompt (i.e. it passed the confidence gate).  When the
        # context was gated out, the LLM answered without retrieval and we
        # must not show citations for it.
        if not self._last_context_injected:
            logger.debug(
                "RAG_DIAG[chat] cited-response | early_return=True reason=context_not_injected "
                "last_context_injected=False"
            )
            return response, None

        chunks = getattr(retrieval_result, "chunks", None)

        # Answer-aware citation filtering: keep only the retrieved chunks
        # that lexically support the generated answer.  Chunks that do not
        # support the answer are removed from the citation set.  This is a
        # provider-neutral, lexical layer (no embeddings, no extra LLM call).
        supporting_chunks = filter_supporting_chunks(response, chunks)

        sources = build_source_refs(supporting_chunks)
        if not sources:
            logger.debug(
                "RAG_DIAG[chat] cited-response | early_return=True reason=no_sources "
                "last_context_injected=%r supporting_chunk_count=%d sources_exist=False",
                self._last_context_injected,
                len(supporting_chunks),
            )
            return response, None

        # The plain-text Sources block is not appended to the response
        # content.  The structured ``sources`` metadata is attached
        # separately and rendered by the UI as the expandable Sources card,
        # so citations appear exactly once.
        citations = build_citations_section(supporting_chunks)
        if not citations:
            logger.debug(
                "RAG_DIAG[chat] cited-response | early_return=True reason=citations_empty "
                "last_context_injected=%r supporting_chunk_count=%d sources_exist=True",
                self._last_context_injected,
                len(supporting_chunks),
            )
            return response, sources

        logger.debug(
            "RAG_DIAG[chat] cited-response | returning_sources=True "
            "last_context_injected=%r supporting_chunk_count=%d sources_exist=True",
            self._last_context_injected,
            len(supporting_chunks),
        )
        return response, sources

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

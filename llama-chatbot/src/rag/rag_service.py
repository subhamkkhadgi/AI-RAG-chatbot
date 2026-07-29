"""RAG orchestration — the bridge between retrieval and context formatting.

``RAGService`` orchestrates the retrieval-augmented generation pipeline:

1. Accept a user query.
2. Retrieve relevant document chunks via ``DocumentRetriever``.
3. Build a formatted context string via ``ContextBuilder``.
4. Return a ``RAGResult`` containing both the structured result and the
   formatted context.

It does **not** call any LLM, create prompts, or manage conversation
history — those responsibilities belong to higher-level services.

The service is provider-neutral: it never imports concrete providers,
embedding classes, or vector-store implementations.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.exceptions import ChatbotError, ConfigurationError
from src.rag.context_builder import ContextBuilder
from src.retrieval.models import RetrievalResult
from src.retrieval.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class RAGResult(BaseModel):
    """The complete result of a RAG query.

    Attributes:
        query: The original user query.
        retrieval_result: The full ``RetrievalResult`` from the
            ``DocumentRetriever``, containing the ordered list of
            ``RetrievedChunk`` objects with metadata.
        context: A formatted, LLM-ready context string built by the
            ``ContextBuilder`` from the retrieved chunks.
    """

    query: str
    retrieval_result: RetrievalResult
    context: str

    model_config = {"frozen": True}


class RAGService:
    """Orchestrate retrieval and context building for RAG.

    This service wires together the existing ``DocumentRetriever`` and
    ``ContextBuilder``, providing a single entry point for executing a
    RAG query.

    Parameters
    ----------
    retriever:
        A ``DocumentRetriever`` instance configured with the desired
        embedding provider and vector store.
    context_builder:
        A ``ContextBuilder`` instance that formats retrieved chunks
        into an LLM-ready context string.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        context_builder: ContextBuilder,
    ) -> None:
        if not isinstance(retriever, DocumentRetriever):
            raise TypeError(
                f"Expected a DocumentRetriever instance, got {type(retriever).__name__}"
            )
        if not isinstance(context_builder, ContextBuilder):
            raise TypeError(
                f"Expected a ContextBuilder instance, got {type(context_builder).__name__}"
            )
        self._retriever: DocumentRetriever = retriever
        self._context_builder: ContextBuilder = context_builder

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def retriever(self) -> DocumentRetriever:
        """The configured ``DocumentRetriever`` instance."""
        return self._retriever

    @property
    def context_builder(self) -> ContextBuilder:
        """The configured ``ContextBuilder`` instance."""
        return self._context_builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(self, text: str, limit: int | None = None) -> RAGResult:
        """Execute a RAG query, returning retrieved chunks and formatted context.

        This method:
        1. Validates the input (non-empty).
        2. Retrieves relevant chunks using ``DocumentRetriever.retrieve()``.
        3. Builds a formatted context string using ``ContextBuilder.build()``.
        4. Returns a ``RAGResult`` with the structured and formatted results.

        Parameters
        ----------
        text:
            The user's search query.
        limit:
            Maximum number of chunks to retrieve. Falls back to the
            retriever's ``default_limit`` if ``None``.

        Returns
        -------
        RAGResult
            A frozen result object containing the query, retrieval
            result, and formatted context string.

        Raises
        ------
        ConfigurationError
            If the query is empty or whitespace-only.
        ChatbotError
            If the context builder fails to format the result.
        """
        # 1. Validate input (let DocumentRetriever raise ConfigurationError)
        #    We do a basic check here to fail fast; DocumentRetriever also validates.
        stripped = text.strip()
        if not stripped:
            raise ConfigurationError(
                "RAG query must not be empty.",
                safe_message="Please enter a query before searching documents.",
            )

        # 2. Retrieve relevant chunks
        logger.info(
            "RAG query (length=%d, limit=%s)", len(stripped), str(limit)
        )
        try:
            retrieval_result: RetrievalResult = self._retriever.retrieve(
                stripped, limit=limit
            )
        except (ConfigurationError, ChatbotError):
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error during retrieval: %s", exc, exc_info=True
            )
            raise ChatbotError(
                "An unexpected error occurred during document retrieval.",
                safe_message="Failed to search documents. Please try again.",
            ) from exc

        # 3. Build formatted context
        try:
            context: str = self._context_builder.build(retrieval_result)
        except Exception as exc:
            logger.error(
                "Unexpected error during context building: %s", exc, exc_info=True
            )
            raise ChatbotError(
                "An unexpected error occurred while building context.",
                safe_message="Failed to prepare document context. Please try again.",
            ) from exc

        # 4. Return result
        return RAGResult(
            query=stripped,
            retrieval_result=retrieval_result,
            context=context,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"retriever={self._retriever!r}, "
            f"context_builder={self._context_builder!r})"
        )


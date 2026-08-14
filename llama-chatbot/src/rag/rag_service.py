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
from src.retrieval.reranker import ChunkReranker, select_page_diverse
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
    reranker:
        An optional ``ChunkReranker``.  When ``None`` (default), the
        existing retrieval pipeline runs unchanged.  When provided, the
        candidate pool is retrieved (``candidate_limit``), re-ranked by
        the reranker, and page-diverse selected down to ``final_limit``
        chunks before the context is built.
    candidate_limit:
        Size of the candidate pool fetched from the vector store when a
        reranker is active.  ``None`` falls back to the retriever's
        ``default_limit``.
    final_limit:
        Number of chunks kept after reranking + page-diverse selection.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        context_builder: ContextBuilder,
        *,
        reranker: ChunkReranker | None = None,
        candidate_limit: int | None = None,
        final_limit: int = 3,
    ) -> None:
        if not isinstance(retriever, DocumentRetriever):
            raise TypeError(
                f"Expected a DocumentRetriever instance, got {type(retriever).__name__}"
            )
        if not isinstance(context_builder, ContextBuilder):
            raise TypeError(
                f"Expected a ContextBuilder instance, got {type(context_builder).__name__}"
            )
        if final_limit < 1:
            raise ValueError(
                f"final_limit must be a positive integer, got {final_limit}"
            )
        if candidate_limit is not None and candidate_limit < 1:
            raise ValueError(
                f"candidate_limit must be a positive integer, got {candidate_limit}"
            )
        self._retriever: DocumentRetriever = retriever
        self._context_builder: ContextBuilder = context_builder
        self._reranker: ChunkReranker | None = reranker
        self._candidate_limit: int | None = candidate_limit
        self._final_limit: int = final_limit

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

    @property
    def reranker(self) -> ChunkReranker | None:
        """The optional ``ChunkReranker``, or ``None`` when disabled."""
        return self._reranker

    @property
    def candidate_limit(self) -> int | None:
        """The candidate pool size used when a reranker is active."""
        return self._candidate_limit

    @property
    def final_limit(self) -> int:
        """The number of chunks kept after reranking + page-diverse selection."""
        return self._final_limit

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
            retriever's ``default_limit`` if ``None``.  When a reranker
            is configured, the configured ``candidate_limit`` pool
            governs retrieval instead (the reranker needs the full
            candidate pool), and the *context* is selected down to
            ``final_limit`` chunks.

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
            If the context builder fails to format the result, or the
            reranker/selection stage fails.
        """
        # 1. Validate input (let DocumentRetriever raise ConfigurationError)
        #    We do a basic check here to fail fast; DocumentRetriever also validates.
        stripped = text.strip()
        if not stripped:
            raise ConfigurationError(
                "RAG query must not be empty.",
                safe_message="Please enter a query before searching documents.",
            )

        # 2. Determine the retrieval limit.  With an active reranker we
        #    fetch the wider candidate pool so reranking sees enough
        #    cross-page/cross-document alternatives.
        effective_limit = limit
        if self._reranker is not None:
            effective_limit = (
                self._candidate_limit
                if self._candidate_limit is not None
                else effective_limit
            )

        # 3. Retrieve relevant chunks
        logger.info(
            "RAG query (length=%d, limit=%s)", len(stripped), str(effective_limit)
        )
        try:
            retrieval_result: RetrievalResult = self._retriever.retrieve(
                stripped, limit=effective_limit
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

        # 4. Optional reranker + page-diverse selection stage.  Only
        #    operates on the in-memory retrieved chunks (no Qdrant writes,
        #    no stored data changes).  Skipped entirely when no reranker
        #    is configured, preserving the exact existing behaviour.
        if self._reranker is not None and retrieval_result.chunks:
            try:
                reranked = self._reranker.rank(stripped, retrieval_result.chunks)
                selected = select_page_diverse(reranked, self._final_limit)
            except ChatbotError:
                raise
            except Exception as exc:
                logger.error(
                    "Unexpected error during reranking/selection: %s",
                    exc,
                    exc_info=True,
                )
                raise ChatbotError(
                    "An unexpected error occurred during document reranking.",
                    safe_message="Failed to refine document search. Please try again.",
                ) from exc

            retrieval_result = RetrievalResult(
                query=stripped,
                chunks=selected,
                total_results=len(selected),
            )

        # 5. Build formatted context
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

        # 6. Return result
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
            f"context_builder={self._context_builder!r}, "
            f"reranker={self._reranker!r}, "
            f"candidate_limit={self._candidate_limit!r}, "
            f"final_limit={self._final_limit!r})"
        )


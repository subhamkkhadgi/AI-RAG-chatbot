"""Provider-neutral document retriever.

The retriever accepts a query, generates an embedding using
``BaseEmbeddingProvider``, searches via ``BaseVectorStore``, and
returns a ``RetrievalResult`` containing ``RetrievedChunk`` objects.

It never imports concrete providers (Ollama, Qdrant, …) — only the
abstract interfaces.
"""

from __future__ import annotations

import logging
from typing import Any

from src.embeddings.base import BaseEmbeddingProvider
from src.exceptions import ConfigurationError, ProviderResponseError
from src.retrieval.models import RetrievedChunk, RetrievalResult
from src.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieve relevant document chunks for a query.

    Parameters
    ----------
    embedding_provider:
        An embedding provider that implements ``BaseEmbeddingProvider``.
    vector_store:
        A vector store that implements ``BaseVectorStore``.
    default_limit:
        Default number of results to return when ``limit`` is not
        specified (default 3).
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        default_limit: int = 3,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_limit = default_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, limit: int | None = None) -> RetrievalResult:
        """Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query:
            The user's search query.
        limit:
            Maximum number of results to return.  Falls back to
            ``default_limit`` if ``None``.

        Returns
        -------
        RetrievalResult
            An ordered list of retrieved chunks with metadata.

        Raises
        ------
        ConfigurationError
            If the query is empty or whitespace-only.
        ProviderResponseError
            If the embedding provider or vector store returns an
            unexpected response.
        """
        # 1. Validate query
        stripped = query.strip()
        if not stripped:
            raise ConfigurationError("Query must not be empty.")

        resolved_limit = limit if limit is not None else self._default_limit

        # 2. Generate query embedding
        logger.info("Generating embedding for query (length=%d)", len(stripped))
        query_vector = self._embedding_provider.embed(stripped)

        if not query_vector:
            raise ProviderResponseError(
                self._embedding_provider.provider_name,
                safe_message="Embedding provider returned an empty vector for the query.",
            )

        # 3. Search vector store
        logger.info("Searching vector store (limit=%d)", resolved_limit)
        raw_results = self._vector_store.search(query_vector, limit=resolved_limit)

        # 4. Convert results to RetrievedChunk models
        chunks = self._build_chunks(raw_results)

        # 5. Return result
        return RetrievalResult(
            query=stripped,
            chunks=chunks,
            total_results=len(chunks),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_chunks(raw_results: list[dict[str, Any]]) -> list[RetrievedChunk]:
        """Convert raw vector store results to ``RetrievedChunk`` objects.

        Parameters
        ----------
        raw_results:
            A list of result dicts from ``BaseVectorStore.search()``,
            each containing ``id``, ``score``, and ``payload`` keys.

        Returns
        -------
        list[RetrievedChunk]
            An ordered list of chunk models.
        """
        chunks: list[RetrievedChunk] = []

        for raw in raw_results:
            payload = raw.get("payload") or {}

            chunk = RetrievedChunk(
                chunk_id=str(raw.get("id", "")),
                document_id=str(payload.get("document_id", "")),
                filename=str(payload.get("filename", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=str(payload.get("text", "")),
                score=float(raw.get("score", 0.0)),
                page_number=(
                    int(payload["page_number"])
                    if payload.get("page_number") is not None
                    else None
                ),
                created_at=str(payload.get("created_at", "")),
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        """Return the configured embedding provider."""
        return self._embedding_provider

    @property
    def vector_store(self) -> BaseVectorStore:
        """Return the configured vector store."""
        return self._vector_store

    @property
    def default_limit(self) -> int:
        """Return the default result limit."""
        return self._default_limit

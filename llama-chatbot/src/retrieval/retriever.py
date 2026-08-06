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
from src.retrieval.models import DocumentScope, RetrievedChunk, RetrievalResult
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
        min_score: float | None = None,
        max_documents: int = 0,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_limit = default_limit

        #: Minimum cosine-similarity score. ``None`` means no threshold.
        #: Defaults to ``settings.retrieval_min_score`` when not supplied.
        self._min_score: float | None = self._resolve_min_score(min_score)

        #: Maximum number of documents in a single result set.
        #: ``0`` means unlimited (threshold-only filtering).
        #: Defaults to ``settings.retrieval_max_documents`` when not supplied.
        self._max_documents: int = self._resolve_max_documents(max_documents)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        limit: int | None = None,
        scope: DocumentScope | None = None,
    ) -> RetrievalResult:
        """Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query:
            The user's search query.
        limit:
            Maximum number of results to return.  Falls back to
            ``default_limit`` if ``None``.
        scope:
            Optional ``DocumentScope`` restricting retrieval to specific
            documents.  ``None`` means unrestricted retrieval across all
            documents.

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

        # 3. Build the filter dict from the optional scope.
        filter_dict = self._build_scope_filter(scope)

        # 4. Search vector store (optionally scoped to documents).
        logger.info("Searching vector store (limit=%d)", resolved_limit)
        raw_results = self._vector_store.search(
            query_vector,
            limit=resolved_limit,
            filter_dict=filter_dict,
        )

        # 5. Convert results to RetrievedChunk models
        chunks = self._build_chunks(raw_results)

        # 6. Apply retrieval-precision filters:
        #    (a) drop chunks below the minimum similarity score, and
        #    (b) keep only the top-`max_documents` documents (ranked by
        #        their best chunk score) when a positive limit is set.
        chunks = self._apply_precision_filters(chunks)

        # 7. Apply a strict scope guard (defence in depth): even if the
        #    underlying store ignored the filter, drop any chunk that is
        #    not part of the selected documents.
        chunks = self._apply_scope_guard(chunks, scope)

        # 8. Return result
        return RetrievalResult(
            query=stripped,
            chunks=chunks,
            total_results=len(chunks),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_scope_filter(scope: DocumentScope | None) -> dict | None:
        """Build a payload filter dict from an optional document scope.

        Only a single ``document_id`` scope is expressible as a simple
        equality filter.  When the scope is empty or contains multiple
        documents, ``None`` is returned so the caller falls back to
        unrestricted retrieval (the strict scope guard still applies).

        Parameters
        ----------
        scope:
            The optional ``DocumentScope`` to translate into a filter.

        Returns
        -------
        dict | None
            A ``{"document_id": value}`` filter, or ``None`` when no
            filter can be applied.
        """
        if scope is None or not scope.document_ids:
            return None
        if len(scope.document_ids) == 1:
            return {"document_id": scope.document_ids[0]}
        return None

    @staticmethod
    def _apply_scope_guard(
        chunks: list[RetrievedChunk],
        scope: DocumentScope | None,
    ) -> list[RetrievedChunk]:
        """Drop chunks that fall outside the requested document scope.

        This is a defence-in-depth guard applied after retrieval so that
        even a vector store that ignores the filter never leaks chunks
        from unselected documents.

        Parameters
        ----------
        chunks:
            The retrieved chunks.
        scope:
            The optional ``DocumentScope``.  ``None`` or an empty scope
            means no restriction.

        Returns
        -------
        list[RetrievedChunk]
            The chunks restricted to the requested documents.
        """
        if scope is None or not scope.document_ids:
            return chunks

        allowed = set(scope.document_ids)
        return [c for c in chunks if c.document_id in allowed]

    def _apply_precision_filters(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Filter retrieved chunks by similarity threshold and document cap.

        Steps
        -----
        1. If ``min_score`` is set, drop every chunk with
           ``score < min_score``.
        2. If ``max_documents`` is positive, group the surviving chunks by
           ``document_id`` and keep only the top ``max_documents``
           documents ranked by their best chunk score.  All chunks from
           lower-ranked documents are removed.

        The remaining chunks stay in their original (score-descending)
        order.

        Parameters
        ----------
        chunks:
            Retrieved chunks, ordered by descending relevance.

        Returns
        -------
        list[RetrievedChunk]
            The filtered chunks.
        """
        if not chunks:
            return []

        # 1. Score threshold filter (provider-neutral, post-search)
        if self._min_score is not None:
            before = len(chunks)
            chunks = [c for c in chunks if c.score >= self._min_score]
            dropped = before - len(chunks)
            if dropped:
                logger.info(
                    "Dropped %d chunk(s) below min_score=%.4f",
                    dropped,
                    self._min_score,
                )

        # 2. Document cap filter (limits cross-document false positives)
        if self._max_documents > 0 and chunks:
            chunks = self._limit_documents(chunks, self._max_documents)

        return chunks

    @staticmethod
    def _limit_documents(
        chunks: list[RetrievedChunk], max_documents: int
    ) -> list[RetrievedChunk]:
        """Keep only chunks from the top-``max_documents`` documents.

        Documents are ranked by their highest-scoring chunk (first
        occurrence order breaks ties).  All chunks belonging to documents
        outside the top ``max_documents`` are removed.
        """
        if max_documents <= 0:
            return chunks

        # Best score per document, preserving first-seen order.
        best_by_doc: dict[str, float] = {}
        for chunk in chunks:
            doc_id = chunk.document_id or chunk.filename
            best = best_by_doc.get(doc_id, float("-inf"))
            if chunk.score > best:
                best_by_doc[doc_id] = chunk.score

        # Rank documents by best score descending; ties keep insertion order.
        ranked_docs = sorted(
            best_by_doc.keys(),
            key=lambda doc_id: best_by_doc[doc_id],
            reverse=True,
        )
        allowed_docs = set(ranked_docs[:max_documents])

        return [c for c in chunks if (c.document_id or c.filename) in allowed_docs]

    def _resolve_min_score(self, min_score: float | None) -> float | None:
        """Resolve the similarity threshold.

        Falls back to ``Settings.retrieval_min_score`` when *min_score*
        is ``None``.  Returns ``None`` (no threshold) when the settings
        value is unavailable.
        """
        if min_score is not None:
            return float(min_score)
        try:
            from src.config import get_settings

            return float(get_settings().retrieval_min_score)
        except Exception:  # noqa: BLE001
            return None

    def _resolve_max_documents(self, max_documents: int) -> int:
        """Resolve the maximum-documents cap.

        Falls back to ``Settings.retrieval_max_documents`` when
        *max_documents* is ``0`` (the sentinel for "use default").
        Returns ``0`` (unlimited) when the settings value is unavailable.
        """
        if max_documents > 0:
            return int(max_documents)
        try:
            from src.config import get_settings

            return int(get_settings().retrieval_max_documents)
        except Exception:  # noqa: BLE001
            return 0

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

    @property
    def min_score(self) -> float | None:
        """Return the configured minimum similarity threshold.

        ``None`` means no threshold (all retrieved chunks are kept).
        """
        return self._min_score

    @property
    def max_documents(self) -> int:
        """Return the maximum number of documents allowed in a result set.

        ``0`` means unlimited.
        """
        return self._max_documents

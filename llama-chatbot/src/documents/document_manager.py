"""Document management service for listing and deleting uploaded documents.

This module provides a ``DocumentManager`` that relies solely on the
vector store's payload data — no separate metadata database is used.
All document metadata (document_id, filename, chunk_count, created_at)
is extracted from Qdrant payloads via ``scroll()``.
"""

from __future__ import annotations

import logging
from typing import Any

from src.exceptions import ChatbotError, ProviderResponseError
from src.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document record model (informal — plain dict for simplicity)
# ---------------------------------------------------------------------------
# Each document record has the shape:
#   {
#       "document_id": str,
#       "filename": str,
#       "chunk_count": int,
#       "created_at": str,
#   }


# ---------------------------------------------------------------------------
# Document manager
# ---------------------------------------------------------------------------
class DocumentManager:
    """Manage uploaded documents via the vector store.

    This service queries Qdrant payloads to build a list of unique
    documents and provides deletion at the document level.

    Parameters
    ----------
    vector_store:
        A vector store that implements ``BaseVectorStore``.
    """

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_documents(self) -> list[dict[str, Any]]:
        """Return a deduplicated list of uploaded documents.

        Scrolls all points in the collection, groups by
        ``document_id``, and returns one record per document with
        metadata from the first chunk payload.

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains ``document_id``, ``filename``,
            ``chunk_count``, and ``created_at``.  Ordered by
            ``created_at`` descending (most recent first).  Returns
            an empty list if no documents exist.
        """
        try:
            points = self._vector_store.scroll(limit=9999)
        except ProviderResponseError as exc:
            logger.error("Failed to scroll vector store: %s", exc)
            raise ChatbotError(
                "Failed to list documents. The vector store may be unavailable.",
                safe_message="Could not retrieve document list. Please try again.",
            ) from exc

        # Group points by document_id, keeping the first payload for each
        document_map: dict[str, dict[str, Any]] = {}

        for point in points:
            payload = point.get("payload", {})
            doc_id = payload.get("document_id", "")
            if not doc_id:
                continue

            if doc_id not in document_map:
                document_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": payload.get("filename", "unknown"),
                    "chunk_count": payload.get("chunk_count", 0),
                    "created_at": payload.get("created_at", ""),
                }

        # Sort by created_at descending (most recent first)
        documents = list(document_map.values())
        documents.sort(key=lambda d: d.get("created_at", ""), reverse=True)

        return documents

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Parameters
        ----------
        document_id:
            The unique identifier of the document to delete.

        Returns
        -------
        int
            The number of chunks (points) deleted.

        Raises
        ------
        ChatbotError
            If the deletion fails.
        """
        if not document_id or not document_id.strip():
            raise ChatbotError(
                "Document ID must not be empty.",
                safe_message="Invalid document ID.",
            )

        try:
            deleted_count = self._vector_store.delete_by_document_id(document_id)
            logger.info(
                "Deleted document '%s' (%d chunk(s)).",
                document_id,
                deleted_count,
            )
            return deleted_count
        except ProviderResponseError as exc:
            logger.error(
                "Failed to delete document '%s': %s",
                document_id,
                exc,
            )
            raise ChatbotError(
                f"Failed to delete document '{document_id}'.",
                safe_message="Could not delete the document. Please try again.",
            ) from exc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vector_store(self) -> BaseVectorStore:
        """Return the configured vector store."""
        return self._vector_store


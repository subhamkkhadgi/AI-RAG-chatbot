"""Qdrant vector store provider implementation.

This provider wraps the ``qdrant-client`` SDK and implements the
``BaseVectorStore`` interface.  It is the only module in the codebase
that may import ``qdrant_client`` — all other layers depend only on
the abstract ``BaseVectorStore``.

The host, port, and collection name are sourced from Settings via
``qdrant_host``, ``qdrant_port``, and ``qdrant_collection`` —
never hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.exceptions import (
    ConfigurationError,
    MissingCredentialsError,
    ProviderConnectionError,
    ProviderResponseError,
)
from src.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Provider that communicates with a Qdrant vector database instance.

    Parameters
    ----------
    settings:
        A validated ``Settings`` instance containing *qdrant_host*,
        *qdrant_port*, and *qdrant_collection* values.
    """

    def __init__(self, settings: object) -> None:
        self._host: str = getattr(settings, "qdrant_host", "localhost")
        self._port: int = int(getattr(settings, "qdrant_port", 6333))
        self._collection: str = getattr(settings, "qdrant_collection", "")
        self._client: QdrantClient | None = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        """Return ``"qdrant"`` as the stable provider identifier."""
        return "qdrant"

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""
        return self._collection

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    def validate_configuration(self) -> None:
        """Verify that the Qdrant host, port, and collection are present.

        This is a local-only check — no network call is made.
        Network reachability is tested by :meth:`is_available`.
        """
        if not self._host:
            raise MissingCredentialsError(
                self.provider_name,
                safe_message="Qdrant host is not configured. Set QDRANT_HOST in your .env file.",
            )
        if not self._port:
            raise MissingCredentialsError(
                self.provider_name,
                safe_message="Qdrant port is not configured. Set QDRANT_PORT in your .env file.",
            )
        if not self._collection:
            raise MissingCredentialsError(
                self.provider_name,
                safe_message="Qdrant collection is not configured. Set QDRANT_COLLECTION in your .env file.",
            )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Check whether the Qdrant server is reachable.

        Uses a lightweight ``get_collections()`` call so it is safe to
        call frequently.  Returns ``False`` on any connection issue
        without raising.
        """
        try:
            client = self._get_client()
            client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Qdrant availability check failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def create_collection(self, vector_size: int) -> None:
        """Create a Qdrant collection with the given vector size.

        If the collection already exists, this is a no-op.
        """
        client = self._get_client()

        try:
            if client.collection_exists(self._collection):
                logger.debug(
                    "Collection %r already exists — skipping creation.",
                    self._collection,
                )
                return

            client.create_collection(
                collection_name=self._collection,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info("Created collection %r with vector size %d.", self._collection, vector_size)
        except Exception as exc:
            logger.error("Failed to create collection %r: %s", self._collection, exc)
            raise ProviderResponseError(
                self.provider_name,
                safe_message=f"Failed to create collection '{self._collection}'. Check the logs for details.",
            ) from exc

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------
    def upsert(
        self,
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """Insert or update vectors with associated payloads.

        Parameters
        ----------
        vectors:
            A list of dense vectors to store.
        payloads:
            A list of metadata dicts, one per vector, in the same order.
        """
        if len(vectors) != len(payloads):
            raise ConfigurationError(
                "Mismatch between number of vectors and payloads: "
                f"{len(vectors)} vectors vs {len(payloads)} payloads.",
            )

        if not vectors:
            return

        client = self._get_client()

        points = [
            qdrant_models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload,
            )
            for vector, payload in zip(vectors, payloads)
        ]

        try:
            client.upsert(
                collection_name=self._collection,
                points=points,
            )
        except Exception as exc:
            logger.error("Qdrant upsert error: %s", exc)
            raise ProviderResponseError(
                self.provider_name,
                safe_message="Failed to insert vectors into Qdrant. Check the logs for details.",
            ) from exc

    def search(
        self,
        vector: list[float],
        limit: int = 10,
    ) -> list[dict]:
        """Search for the nearest neighbours of a query vector.

        Parameters
        ----------
        vector:
            The query vector to search with.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            A list of result dicts, each containing ``id``, ``score``,
            and ``payload`` keys.
        """
        client = self._get_client()

        try:
            response = client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=limit,
            )
            results = response.points
        except Exception as exc:
            logger.error("Qdrant search error: %s", exc)
            raise ProviderResponseError(
                self.provider_name,
                safe_message="Failed to search in Qdrant. Check the logs for details.",
            ) from exc

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {},
            }
            for point in results
        ]

    def delete(self, document_id: str) -> None:
        """Delete a document from the collection by its ID.

        Parameters
        ----------
        document_id:
            The unique identifier of the document to delete.
        """
        client = self._get_client()

        try:
            client.delete(
                collection_name=self._collection,
                points_selector=qdrant_models.PointIdsList(
                    points=[document_id],
                ),
            )
        except Exception as exc:
            logger.error("Qdrant delete error: %s", exc)
            raise ProviderResponseError(
                self.provider_name,
                safe_message=f"Failed to delete document '{document_id}' from Qdrant. Check the logs for details.",
            ) from exc

    def scroll(
        self,
        limit: int = 100,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Scroll through points in the collection with an optional filter.

        Parameters
        ----------
        limit:
            Maximum number of points to return.
        filter_dict:
            Optional payload filter to narrow results.

        Returns
        -------
        list[dict]
            A list of point dicts, each containing ``id``, ``payload``,
            and optionally ``vector``.
        """
        client = self._get_client()

        try:
            qdrant_filter = None
            if filter_dict:
                conditions = [
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                    for key, value in filter_dict.items()
                ]
                qdrant_filter = qdrant_models.Filter(must=conditions)

            records, _ = client.scroll(
                collection_name=self._collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                scroll_filter=qdrant_filter,
            )

            return [
                {
                    "id": str(record.id),
                    "payload": record.payload or {},
                }
                for record in records
            ]
        except Exception as exc:
            logger.error("Qdrant scroll error: %s", exc)
            raise ProviderResponseError(
                self.provider_name,
                safe_message="Failed to scroll points in Qdrant. Check the logs for details.",
            ) from exc

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all points belonging to a document.

        Uses a payload filter on ``document_id`` to find and remove
        all chunks associated with the document.

        Parameters
        ----------
        document_id:
            The unique identifier of the document whose points should
            be deleted.

        Returns
        -------
        int
            The number of points deleted.
        """
        client = self._get_client()

        try:
            # First, scroll to count and collect point IDs
            records, _ = client.scroll(
                collection_name=self._collection,
                limit=9999,
                with_payload=False,
                with_vectors=False,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        ),
                    ],
                ),
            )

            if not records:
                logger.info(
                    "No points found for document_id='%s' — nothing to delete.",
                    document_id,
                )
                return 0

            point_ids = [record.id for record in records]

            client.delete(
                collection_name=self._collection,
                points_selector=qdrant_models.PointIdsList(
                    points=point_ids,
                ),
            )

            deleted_count = len(point_ids)
            logger.info(
                "Deleted %d point(s) for document_id='%s'",
                deleted_count,
                document_id,
            )
            return deleted_count

        except Exception as exc:
            logger.error(
                "Qdrant delete_by_document_id error for '%s': %s",
                document_id,
                exc,
            )
            raise ProviderResponseError(
                self.provider_name,
                safe_message=f"Failed to delete document '{document_id}' from Qdrant. Check the logs for details.",
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_client(self) -> QdrantClient:
        """Return a lazily-initialised QdrantClient.

        The client is created once and cached for the lifetime of the
        provider instance.
        """
        if self._client is None:
            self._client = QdrantClient(
                host=self._host,
                port=self._port,
            )
        return self._client

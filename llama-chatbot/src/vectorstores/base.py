"""Abstract base class for all vector store providers.

Every provider must implement the methods described below.
Providers are **stateless**: they store and retrieve vectors, and do not
mutate any external state beyond the vector database they connect to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """Abstract interface for a vector database provider.

    Subclasses must define:

    - ``provider_name``          — stable lowercase identifier (e.g. ``"qdrant"``).
    - ``collection_name``        — name of the collection/ index being operated on.
    - ``create_collection``      — create a collection with the given vector size.
    - ``upsert``                 — insert or update vectors with payloads.
    - ``search``                 — search for nearest neighbours.
    - ``delete``                 — delete a document by its ID.
    - ``validate_configuration`` — verify credentials / configuration.
    - ``is_available``           — lightweight reachability check.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider's stable lowercase identifier.

        This value is used for registry lookups and should match the
        name used in :meth:`register_vector_store`.
        Example: ``"qdrant"``.
        """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Return the name of the collection this store operates on."""

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    @abstractmethod
    def create_collection(self, vector_size: int) -> None:
        """Create a collection for storing vectors.

        Parameters
        ----------
        vector_size:
            The dimensionality of the vectors to be stored.

        Raises
        ------
        ConfigurationError
            If the collection configuration is invalid.
        ProviderConnectionError
            If the vector database cannot be reached.
        ProviderResponseError
            If the vector database returns an unexpected response.
        """

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------
    @abstractmethod
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

        Raises
        ------
        ProviderConnectionError
            If the vector database cannot be reached.
        ProviderResponseError
            If the vector database returns an unexpected response.
        """

    @abstractmethod
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
            A list of result dicts, each containing at minimum ``id``,
            ``score``, and ``payload`` keys.

        Raises
        ------
        ProviderConnectionError
            If the vector database cannot be reached.
        ProviderResponseError
            If the vector database returns an unexpected response.
        """

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Delete a document from the collection by its ID.

        Parameters
        ----------
        document_id:
            The unique identifier of the document to delete.

        Raises
        ------
        ProviderConnectionError
            If the vector database cannot be reached.
        ProviderResponseError
            If the vector database returns an unexpected response.
        """

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate provider-specific configuration.

        Checks that required credentials and settings are present and
        syntactically valid.  This method should *not* perform network
        calls; network reachability is checked by :meth:`is_available`.

        Raises
        ------
        MissingCredentialsError
            If a required host, port, or collection name is missing.
        ConfigurationError
            If a setting has an invalid value.
        """

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    @abstractmethod
    def is_available(self) -> bool:
        """Perform a lightweight reachability check.

        Returns ``True`` if the vector database can be reached, ``False``
        otherwise.  This method should be suitable for periodic health
        checks.  It must not raise exceptions for routine connection
        failures; errors should be logged and ``False`` returned.

        .. caution::
           Providers should implement this as a simple ping / health
           endpoint call.  Avoid expensive collection-loading operations.
        """

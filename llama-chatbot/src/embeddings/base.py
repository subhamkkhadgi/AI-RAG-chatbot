"""Abstract base class for all embedding providers.

Every provider must implement the methods described below.
Providers are **stateless**: they accept text, return vectors, and do not
mutate any external state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract interface for an embedding provider.

    Subclasses must define:

    - ``provider_name``          — stable lowercase identifier (e.g. ``"ollama"``).
    - ``embed``                  — embed a single text string into a vector.
    - ``embed_batch``            — embed a list of texts into a list of vectors.
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
        name used in :meth:`register_embedding_provider`.
        Example: ``"ollama"``.
        """

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Parameters
        ----------
        text:
            The input text to embed.

        Returns
        -------
        list[float]
            A dense vector representation of the input text.

        Raises
        ------
        MissingCredentialsError
            If required credentials are missing.
        ProviderConnectionError
            If the provider cannot be reached.
        ProviderResponseError
            If the provider returns an unexpected or malformed response.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of vectors.

        Parameters
        ----------
        texts:
            A list of input text strings to embed.

        Returns
        -------
        list[list[float]]
            A list of dense vectors, one per input text, in the same order.

        Raises
        ------
        MissingCredentialsError
            If required credentials are missing.
        ProviderConnectionError
            If the provider cannot be reached.
        ProviderResponseError
            If the provider returns an unexpected or malformed response.
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
            If a required API key, host, or model is missing.
        ConfigurationError
            If a setting has an invalid value.
        """

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    @abstractmethod
    def is_available(self) -> bool:
        """Perform a lightweight reachability check.

        Returns ``True`` if the provider can be reached, ``False``
        otherwise.  This method should be suitable for periodic health
        checks.  It must not raise exceptions for routine connection
        failures; errors should be logged and ``False`` returned.

        .. caution::
           Providers should implement this as a simple ping / health
           endpoint call.  Avoid expensive model-loading operations.
        """


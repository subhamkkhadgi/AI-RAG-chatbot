"""Abstract base class for all LLM providers.

Every provider must implement the four methods described below.
Providers are **stateless with respect to conversation history**: they
accept a ``ChatRequest``, yield response chunks, and do not mutate the
request or any external state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.models.chat import ChatRequest


class BaseLLMProvider(ABC):
    """Abstract interface for an LLM inference provider.

    Subclasses must define:

    - ``provider_name``     — stable lowercase identifier (e.g. ``"groq"``).
    - ``chat``              — stream response tokens for a chat request.
    - ``validate_configuration`` — verify credentials / configuration.
    - ``is_available``      — lightweight reachability check.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider's stable lowercase identifier.

        This value is used for registry lookups and should match the
        name used in :meth:`register_provider`.  Example: ``"groq"``.
        """

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    @abstractmethod
    def chat(self, request: ChatRequest) -> Iterator[str]:
        """Yield normalised text chunks from the LLM.

        Parameters
        ----------
        request:
            A fully populated, provider-neutral chat request.

        Yields
        ------
        str
            Non-empty text chunks as they are received from the
            underlying API.  Chunks are normalised so that callers
            receive plain strings regardless of whether the provider
            uses server-sent events, WebSockets, or polling.

        Raises
        ------
        MissingCredentialsError
            If required credentials are missing.
        ProviderConnectionError
            If the provider cannot be reached.
        ProviderRateLimitError
            If the provider returns a rate-limit response.
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
        checks (e.g. sidebar status indicator).  It must not raise
        exceptions for routine connection failures; errors should be
        logged and ``False`` returned.

        .. caution::
           Providers should implement this as a simple ping / health
           endpoint call.  Avoid expensive model-loading operations.
        """

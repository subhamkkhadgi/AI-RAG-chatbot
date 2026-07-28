"""Embedding provider factory.

Usage
-----
    from src.embeddings import create_embedding_provider

    provider = create_embedding_provider("ollama", settings)

The factory is provider-agnostic.  Adding a new provider requires only
registering it in ``src/embeddings/__init__.py`` — no factory changes
needed.
"""

from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.registry import _registry, list_registered_embedding_providers
from src.exceptions import ProviderNotFoundError


def create_embedding_provider(
    provider_name: str,
    settings: object,
) -> BaseEmbeddingProvider:
    """Construct and return an embedding provider instance.

    Parameters
    ----------
    provider_name:
        The registered name of the provider (case-insensitive).
    settings:
        A validated ``Settings`` instance (from ``src.config``).

    Returns
    -------
    BaseEmbeddingProvider
        A fully initialised embedding provider instance.

    Raises
    ------
    ProviderNotFoundError
        If *provider_name* is not in the registry.
    """
    normalized = provider_name.strip().lower()
    provider_class = _registry.get(normalized)

    if provider_class is None:
        supported = (
            ", ".join(list_registered_embedding_providers())
            if _registry
            else "(none registered)"
        )
        raise ProviderNotFoundError(
            normalized,
            safe_message=(
                f"Unsupported embedding provider: {normalized!r}. "
                f"Supported providers: {supported}."
            ),
        )

    instance = provider_class(settings)
    instance.validate_configuration()
    return instance


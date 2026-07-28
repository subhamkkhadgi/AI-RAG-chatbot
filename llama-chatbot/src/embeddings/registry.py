"""Embedding provider registry.

Usage
-----
    from src.embeddings.registry import register_embedding_provider

    register_embedding_provider("ollama", OllamaEmbeddingProvider)

The registry is an explicit ``dict`` — no dynamic discovery or
metaclass magic is used.
"""

from src.embeddings.base import BaseEmbeddingProvider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_registry: dict[str, type[BaseEmbeddingProvider]] = {}


def register_embedding_provider(
    name: str,
    provider_class: type[BaseEmbeddingProvider],
    *,
    allow_overwrite: bool = False,
) -> None:
    """Register an embedding provider class under *name*.

    Parameters
    ----------
    name:
        Lowercase identifier for the provider (e.g. ``"ollama"``).
    provider_class:
        A concrete subclass of :class:`BaseEmbeddingProvider`.
    allow_overwrite:
        If ``True``, silently replace an existing registration.
        If ``False`` (default), raise :exc:`ValueError` when *name*
        is already registered.

    Raises
    ------
    ValueError
        If *name* is empty, or if it is already registered and
        *allow_overwrite* is ``False``.
    TypeError
        If *provider_class* is not a subclass of :class:`BaseEmbeddingProvider`.
    """
    normalized = name.strip().lower()

    if not normalized:
        raise ValueError("Provider name must not be empty")

    if not (
        isinstance(provider_class, type)
        and issubclass(provider_class, BaseEmbeddingProvider)
    ):
        raise TypeError(
            f"{provider_class!r} must be a subclass of BaseEmbeddingProvider"
        )

    if normalized in _registry and not allow_overwrite:
        raise ValueError(
            f"Provider {normalized!r} is already registered "
            f"(use allow_overwrite=True to replace)"
        )

    _registry[normalized] = provider_class


def list_registered_embedding_providers() -> list[str]:
    """Return a sorted list of all registered embedding provider names."""
    return sorted(_registry.keys())


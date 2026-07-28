"""Vector store provider registry.

Usage
-----
    from src.vectorstores.registry import register_vector_store

    register_vector_store("qdrant", QdrantVectorStore)

The registry is an explicit ``dict`` — no dynamic discovery or
metaclass magic is used.
"""

from src.vectorstores.base import BaseVectorStore

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_registry: dict[str, type[BaseVectorStore]] = {}


def register_vector_store(
    name: str,
    store_class: type[BaseVectorStore],
    *,
    allow_overwrite: bool = False,
) -> None:
    """Register a vector store provider class under *name*.

    Parameters
    ----------
    name:
        Lowercase identifier for the provider (e.g. ``"qdrant"``).
    store_class:
        A concrete subclass of :class:`BaseVectorStore`.
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
        If *store_class* is not a subclass of :class:`BaseVectorStore`.
    """
    normalized = name.strip().lower()

    if not normalized:
        raise ValueError("Vector store provider name must not be empty")

    if not (
        isinstance(store_class, type)
        and issubclass(store_class, BaseVectorStore)
    ):
        raise TypeError(
            f"{store_class!r} must be a subclass of BaseVectorStore"
        )

    if normalized in _registry and not allow_overwrite:
        raise ValueError(
            f"Vector store provider {normalized!r} is already registered "
            f"(use allow_overwrite=True to replace)"
        )

    _registry[normalized] = store_class


def list_registered_vector_stores() -> list[str]:
    """Return a sorted list of all registered vector store provider names."""
    return sorted(_registry.keys())

"""Vector store provider factory.

Usage
-----
    from src.vectorstores import create_vector_store

    store = create_vector_store("qdrant", settings)

The factory is provider-agnostic.  Adding a new provider requires only
registering it in ``src/vectorstores/__init__.py`` — no factory changes
needed.
"""

from src.exceptions import ProviderNotFoundError
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.registry import _registry, list_registered_vector_stores


def create_vector_store(
    provider_name: str,
    settings: object,
) -> BaseVectorStore:
    """Construct and return a vector store provider instance.

    Parameters
    ----------
    provider_name:
        The registered name of the provider (case-insensitive).
    settings:
        A validated ``Settings`` instance (from ``src.config``).

    Returns
    -------
    BaseVectorStore
        A fully initialised vector store instance.

    Raises
    ------
    ProviderNotFoundError
        If *provider_name* is not in the registry.
    """
    normalized = provider_name.strip().lower()
    store_class = _registry.get(normalized)

    if store_class is None:
        supported = (
            ", ".join(list_registered_vector_stores())
            if _registry
            else "(none registered)"
        )
        raise ProviderNotFoundError(
            normalized,
            safe_message=(
                f"Unsupported vector store provider: {normalized!r}. "
                f"Supported providers: {supported}."
            ),
        )

    instance = store_class(settings)
    instance.validate_configuration()
    return instance

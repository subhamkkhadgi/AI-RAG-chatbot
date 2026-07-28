"""Vector store provider abstractions and factory.

The public API currently exposes:

- ``BaseVectorStore`` — abstract base class for all vector store providers.
- ``create_vector_store`` — factory that constructs and validates a provider.
- ``register_vector_store`` — registration function for adding new providers.
- ``list_registered_vector_stores`` — return sorted list of registered provider names.

Concrete implementations register themselves here so that importing
``src.vectorstores`` is sufficient to populate the registry.
"""

from .base import BaseVectorStore
from .factory import create_vector_store
from .registry import list_registered_vector_stores, register_vector_store
from .qdrant_vector_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# Register available vector store providers
# ---------------------------------------------------------------------------
# Each provider class is registered here so the factory can look it up by
# name.  Adding a new provider requires two steps:
#   1. Create a class implementing BaseVectorStore in a new module.
#   2. Import it here and call register_vector_store("name", ProviderClass).
#
# The registry is populated once at import time — no lazy discovery or
# metaclass magic is used.  This keeps the factory itself provider-agnostic.
# ---------------------------------------------------------------------------
register_vector_store("qdrant", QdrantVectorStore)

__all__ = [
    "BaseVectorStore",
    "create_vector_store",
    "list_registered_vector_stores",
    "register_vector_store",
]

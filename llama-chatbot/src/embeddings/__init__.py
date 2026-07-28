"""Embedding provider abstractions and factory.

The public API currently exposes:

- ``BaseEmbeddingProvider`` — abstract base class for all embedding providers.
- ``create_embedding_provider`` — factory that constructs and validates a provider.
- ``register_embedding_provider`` — registration function for adding new providers.
- ``list_registered_embedding_providers`` — return sorted list of registered provider names.

Concrete implementations register themselves here so that importing
``src.embeddings`` is sufficient to populate the registry.
"""

from .base import BaseEmbeddingProvider
from .factory import create_embedding_provider
from .ollama_embedding_provider import OllamaEmbeddingProvider
from .registry import (
    list_registered_embedding_providers,
    register_embedding_provider,
)

# ---------------------------------------------------------------------------
# Register available embedding providers
# ---------------------------------------------------------------------------
# Each provider class is registered here so the factory can look it up by
# name.  Adding a new provider requires two steps:
#   1. Create a class implementing BaseEmbeddingProvider in a new module.
#   2. Import it here and call register_embedding_provider("name", ProviderClass).
#
# The registry is populated once at import time — no lazy discovery or
# metaclass magic is used.  This keeps the factory itself provider-agnostic.
# ---------------------------------------------------------------------------
register_embedding_provider("ollama", OllamaEmbeddingProvider)

__all__ = [
    "BaseEmbeddingProvider",
    "create_embedding_provider",
    "list_registered_embedding_providers",
    "register_embedding_provider",
]

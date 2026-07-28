"""LLM provider abstractions and factory.

The public API currently exposes:

- ``BaseLLMProvider`` — abstract base class for all providers.
- ``create_provider`` — factory that constructs and validates a provider.
- ``register_provider`` — registration function for adding new providers.
- ``list_registered_providers`` — return sorted list of registered provider names.

Concrete implementations register themselves here so that importing
``src.providers`` is sufficient to populate the registry.
"""

from .base import BaseLLMProvider
from .factory import (
    create_provider,
    list_registered_providers,
    register_provider,
)
from .ollama_provider import OllamaProvider

# ---------------------------------------------------------------------------
# Register available providers
# ---------------------------------------------------------------------------
# Each provider class is registered here so the factory can look it up by
# name.  Adding a new provider requires two steps:
#   1. Create a class implementing BaseLLMProvider in a new module.
#   2. Import it here and call register_provider("name", ProviderClass).
#
# The registry is populated once at import time — no lazy discovery or
# metaclass magic is used.  This keeps the factory itself provider-agnostic.
# ---------------------------------------------------------------------------
register_provider("ollama", OllamaProvider)

__all__ = [
    "BaseLLMProvider",
    "create_provider",
    "list_registered_providers",
    "register_provider",
]

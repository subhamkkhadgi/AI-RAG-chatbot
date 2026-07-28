"""Provider registry and factory.

Usage
-----
    from src.providers import register_provider, create_provider

    register_provider("groq", GroqProvider)
    provider = create_provider("groq", settings)

The registry is an explicit ``dict`` — no dynamic discovery or
metaclass magic is used.
"""

from src.exceptions import ProviderNotFoundError
from src.providers.base import BaseLLMProvider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_registry: dict[str, type[BaseLLMProvider]] = {}


def register_provider(
    name: str,
    provider_class: type[BaseLLMProvider],
    *,
    allow_overwrite: bool = False,
) -> None:
    """Register an LLM provider class under *name*.

    Parameters
    ----------
    name:
        Lowercase identifier for the provider (e.g. ``"groq"``).
    provider_class:
        A concrete subclass of :class:`BaseLLMProvider`.
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
        If *provider_class* is not a subclass of :class:`BaseLLMProvider`.
    """
    normalized = name.strip().lower()

    if not normalized:
        raise ValueError("Provider name must not be empty")

    if not (
        isinstance(provider_class, type) and issubclass(provider_class, BaseLLMProvider)
    ):
        raise TypeError(f"{provider_class!r} must be a subclass of BaseLLMProvider")

    if normalized in _registry and not allow_overwrite:
        raise ValueError(
            f"Provider {normalized!r} is already registered "
            f"(use allow_overwrite=True to replace)"
        )

    _registry[normalized] = provider_class


def create_provider(
    provider_name: str,
    settings: object,
) -> BaseLLMProvider:
    """Construct and return a provider instance.

    Parameters
    ----------
    provider_name:
        The registered name of the provider (case-insensitive).
    settings:
        A validated ``Settings`` instance (from ``src.config``).

    Returns
    -------
    BaseLLMProvider
        A fully initialised provider instance.

    Raises
    ------
    ProviderNotFoundError
        If *provider_name* is not in the registry.
    """
    normalized = provider_name.strip().lower()
    provider_class = _registry.get(normalized)

    if provider_class is None:
        supported = (
            ", ".join(sorted(_registry.keys())) if _registry else "(none registered)"
        )
        raise ProviderNotFoundError(
            normalized,
            safe_message=(
                f"Unsupported provider: {normalized!r}. "
                f"Supported providers: {supported}."
            ),
        )

    instance = provider_class(settings)
    instance.validate_configuration()
    return instance


def list_registered_providers() -> list[str]:
    """Return a sorted list of all registered provider names."""
    return sorted(_registry.keys())

"""Application-specific exception hierarchy.

All custom exceptions inherit from ``ChatbotError`` so that top-level
handlers can catch them uniformly.  Each exception may carry a
*safe_message* suitable for display in the Streamlit UI, while the
original cause is preserved via ``raise ... from exc`` for logging.
"""

from __future__ import annotations


class ChatbotError(Exception):
    """Base exception for all application-level errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        *,
        safe_message: str = "An unexpected error occurred. Please try again.",
    ) -> None:
        self.safe_message = safe_message
        super().__init__(message)


class ConfigurationError(ChatbotError):
    """Raised when the application configuration is invalid or incomplete."""

    def __init__(self, message: str, *, safe_message: str | None = None) -> None:
        super().__init__(
            message,
            safe_message=safe_message or "Application configuration is invalid.",
        )


class ProviderNotFoundError(ChatbotError):
    """Raised when an unknown provider name is requested."""

    def __init__(self, provider_name: str, *, safe_message: str | None = None) -> None:
        super().__init__(
            f"Unknown provider: {provider_name!r}",
            safe_message=safe_message
            or f"Unsupported provider: {provider_name!r}. Choose 'groq' or 'ollama'.",
        )
        self.provider_name = provider_name


class MissingCredentialsError(ChatbotError):
    """Raised when required credentials for a provider are missing."""

    def __init__(self, provider_name: str, *, safe_message: str | None = None) -> None:
        super().__init__(
            f"Missing credentials for provider: {provider_name!r}",
            safe_message=safe_message
            or f"{provider_name.title()} requires additional configuration. Check the sidebar or .env file.",
        )
        self.provider_name = provider_name


class ProviderConnectionError(ChatbotError):
    """Raised when a connection to a provider fails."""

    def __init__(
        self,
        provider_name: str,
        *,
        safe_message: str | None = None,
    ) -> None:
        super().__init__(
            f"Connection to {provider_name!r} failed",
            safe_message=safe_message
            or f"Could not reach {provider_name.title()}. Check your network or server status.",
        )
        self.provider_name = provider_name


class ProviderRateLimitError(ChatbotError):
    """Raised when a provider returns a rate-limit response."""

    def __init__(
        self,
        provider_name: str,
        *,
        safe_message: str | None = None,
    ) -> None:
        super().__init__(
            f"Rate limit exceeded for provider: {provider_name!r}",
            safe_message=safe_message
            or f"{provider_name.title()} rate limit reached. Please wait a moment and try again.",
        )
        self.provider_name = provider_name


class ProviderResponseError(ChatbotError):
    """Raised when a provider returns an unexpected or malformed response."""

    def __init__(
        self,
        provider_name: str,
        *,
        safe_message: str | None = None,
    ) -> None:
        super().__init__(
            f"Unexpected response from provider: {provider_name!r}",
            safe_message=safe_message
            or f"{provider_name.title()} returned an unexpected response. Check the logs for details.",
        )
        self.provider_name = provider_name

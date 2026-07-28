"""Unit tests for OllamaProvider.

All tests mock the ``ollama`` SDK — no real Ollama instance is required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.exceptions import (
    MissingCredentialsError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from src.models.chat import ChatMessage, ChatRequest, ChatRole
from src.providers.ollama_provider import OllamaProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_settings() -> object:
    """Return a minimal settings-like object with required Ollama fields."""

    class FakeSettings:
        ollama_host: str = "http://localhost:11434"
        ollama_model: str = "llama3.1:8b"

    return FakeSettings()


@pytest.fixture
def empty_settings() -> object:
    """Return a settings object with empty fields for validation tests."""

    class EmptySettings:
        ollama_host: str = ""
        ollama_model: str = ""

    return EmptySettings()


@pytest.fixture
def sample_request() -> ChatRequest:
    """Return a standard chat request for streaming tests."""
    return ChatRequest(
        messages=[
            ChatMessage(role=ChatRole.USER, content="Hello"),
        ],
        model="llama3.1:8b",
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=100,
    )


# ---------------------------------------------------------------------------
# 1. Provider creation
# ---------------------------------------------------------------------------
class TestProviderCreation:
    def test_creation_with_valid_settings(self, mock_settings: object) -> None:
        """Provider should initialise without error given valid settings."""
        provider = OllamaProvider(mock_settings)
        assert provider.provider_name == "ollama"
        assert provider._host == "http://localhost:11434"
        assert provider._default_model == "llama3.1:8b"

    def test_creation_with_different_model_in_settings(self) -> None:
        """Changing OLLAMA_MODEL should only affect the default — no code change needed."""

        class AltSettings:
            ollama_host: str = "http://localhost:11434"
            ollama_model: str = "mistral"

        provider = OllamaProvider(AltSettings())
        assert provider._default_model == "mistral"

    def test_creation_with_custom_host(self) -> None:
        """Custom OLLAMA_HOST is accepted."""

        class CustomHost:
            ollama_host: str = "http://192.168.1.100:11434"
            ollama_model: str = "llama3.1:8b"

        provider = OllamaProvider(CustomHost())
        assert provider._host == "http://192.168.1.100:11434"


# ---------------------------------------------------------------------------
# 2. Configuration validation
# ---------------------------------------------------------------------------
class TestConfigurationValidation:
    def test_valid_configuration_passes(self, mock_settings: object) -> None:
        """Valid settings should not raise."""
        provider = OllamaProvider(mock_settings)
        provider.validate_configuration()  # should not raise

    def test_missing_host_raises(self, empty_settings: object) -> None:
        """Empty host should raise MissingCredentialsError."""
        provider = OllamaProvider(empty_settings)
        with pytest.raises(MissingCredentialsError) as exc_info:
            provider.validate_configuration()
        assert (
            "ollama" in str(exc_info.value).lower()
            or "host" in str(exc_info.value).lower()
        )

    def test_missing_model_raises(self) -> None:
        """Empty model should raise MissingCredentialsError."""

        class NoModel:
            ollama_host: str = "http://localhost:11434"
            ollama_model: str = ""

        provider = OllamaProvider(NoModel())
        with pytest.raises(MissingCredentialsError):
            provider.validate_configuration()


# ---------------------------------------------------------------------------
# 3. Different model names are accepted
# ---------------------------------------------------------------------------
class TestModelNameAcceptance:
    @pytest.mark.parametrize(
        "model_name",
        [
            "llama3.1:8b",
            "llama3.2:3b",
            "mistral",
            "qwen2.5:7b",
            "codellama:13b",
            "phi3:3.8b",
        ],
    )
    def test_different_model_names(
        self, mock_settings: object, model_name: str
    ) -> None:
        """Any model name should be accepted — no hardcoding."""
        request = ChatRequest(
            messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
            model=model_name,
            system_prompt="",
            temperature=0.7,
            max_tokens=50,
        )
        provider = OllamaProvider(mock_settings)

        with patch.object(provider._client, "chat") as mock_chat:
            mock_chat.return_value = [
                {"message": {"content": "response"}, "done": True}
            ]
            result = list(provider.chat(request))

        assert result == ["response"]
        # Verify the model from request was passed to the SDK
        _, kwargs = mock_chat.call_args
        assert kwargs["model"] == model_name


# ---------------------------------------------------------------------------
# 4. Successful streaming
# ---------------------------------------------------------------------------
class TestSuccessfulStreaming:
    def test_yields_text_chunks(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """Provider should yield text chunks from the streaming response."""
        provider = OllamaProvider(mock_settings)
        mock_stream = [
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": " world"}, "done": False},
            {"message": {"content": "!"}, "done": True},
        ]

        with patch.object(provider._client, "chat", return_value=mock_stream):
            result = list(provider.chat(sample_request))

        assert result == ["Hello", " world", "!"]

    def test_preserves_message_ordering(self, mock_settings: object) -> None:
        """Messages in the chat request should be sent in order."""
        provider = OllamaProvider(mock_settings)
        request = ChatRequest(
            messages=[
                ChatMessage(role=ChatRole.USER, content="First"),
                ChatMessage(role=ChatRole.ASSISTANT, content="Second"),
                ChatMessage(role=ChatRole.USER, content="Third"),
            ],
            model="llama3.1:8b",
            system_prompt="System instruction.",
            temperature=0.7,
            max_tokens=100,
        )

        with patch.object(provider._client, "chat") as mock_chat:
            mock_chat.return_value = [{"message": {"content": "ok"}, "done": True}]
            list(provider.chat(request))

        _, kwargs = mock_chat.call_args
        sent_messages = kwargs["messages"]
        # System first, then user/assistant in order
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "System instruction."
        assert sent_messages[1]["role"] == "user"
        assert sent_messages[1]["content"] == "First"
        assert sent_messages[2]["role"] == "assistant"
        assert sent_messages[2]["content"] == "Second"
        assert sent_messages[3]["role"] == "user"
        assert sent_messages[3]["content"] == "Third"


# ---------------------------------------------------------------------------
# 5. Empty chunk filtering
# ---------------------------------------------------------------------------
class TestEmptyChunkFiltering:
    def test_ignores_empty_chunks(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """Empty or whitespace-only chunks should be filtered out."""
        provider = OllamaProvider(mock_settings)
        mock_stream = [
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": ""}, "done": False},
            {"message": {"content": " world"}, "done": False},
            {"message": {"content": "  "}, "done": False},
            {"message": {"content": "!"}, "done": True},
        ]

        with patch.object(provider._client, "chat", return_value=mock_stream):
            result = list(provider.chat(sample_request))

        assert result == ["Hello", " world", "!"]

    def test_all_empty_chunks_raises(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """If all chunks are empty, the provider must raise, not silently return empty."""
        provider = OllamaProvider(mock_settings)
        mock_stream = [
            {"message": {"content": ""}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]

        with (
            patch.object(provider._client, "chat", return_value=mock_stream),
            pytest.raises(ProviderResponseError) as exc_info,
        ):
            list(provider.chat(sample_request))

        assert "empty" in exc_info.value.safe_message.lower()


# ---------------------------------------------------------------------------
# 6. Request immutability
# ---------------------------------------------------------------------------
class TestRequestImmutability:
    def test_request_not_modified(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """The chat request should not be modified by the provider."""
        original_messages = list(sample_request.messages)
        original_model = sample_request.model
        original_system = sample_request.system_prompt

        provider = OllamaProvider(mock_settings)
        with patch.object(provider._client, "chat") as mock_chat:
            mock_chat.return_value = [{"message": {"content": "ok"}, "done": True}]
            list(provider.chat(sample_request))

        assert sample_request.model == original_model
        assert sample_request.system_prompt == original_system
        assert list(sample_request.messages) == original_messages


# ---------------------------------------------------------------------------
# 7. Error handling — connection failure
# ---------------------------------------------------------------------------
class TestConnectionFailure:
    def test_connection_error_raises_provider_connection_error(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """ConnectionError should be translated to ProviderConnectionError."""
        provider = OllamaProvider(mock_settings)

        with patch.object(
            provider._client, "chat", side_effect=ConnectionError("refused")
        ):
            with pytest.raises(ProviderConnectionError) as exc_info:
                list(provider.chat(sample_request))

        assert (
            "ollama" in str(exc_info.value).lower()
            or "connect" in str(exc_info.value).lower()
        )

    def test_response_error_raises_provider_response_error(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """ollama.ResponseError should be translated to ProviderResponseError."""
        import ollama

        provider = OllamaProvider(mock_settings)
        api_error = ollama.ResponseError(status_code=500, error="Internal server error")

        with patch.object(provider._client, "chat", side_effect=api_error):
            with pytest.raises(ProviderResponseError):
                list(provider.chat(sample_request))

    def test_rate_limit_error_raises_provider_rate_limit_error(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """HTTP 429 should be translated to ProviderRateLimitError."""
        import ollama

        provider = OllamaProvider(mock_settings)
        rate_error = ollama.ResponseError(status_code=429, error="Rate limit exceeded")

        with patch.object(provider._client, "chat", side_effect=rate_error):
            with pytest.raises(ProviderRateLimitError):
                list(provider.chat(sample_request))


# ---------------------------------------------------------------------------
# 8. Malformed response handling
# ---------------------------------------------------------------------------
class TestMalformedResponse:
    def test_missing_message_field(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """Chunks without 'message' field should yield empty string (filtered)."""
        provider = OllamaProvider(mock_settings)
        mock_stream = [
            {"done": False},  # no "message" key
            {"message": {"content": "actual"}, "done": True},
        ]

        with patch.object(provider._client, "chat", return_value=mock_stream):
            result = list(provider.chat(sample_request))

        assert result == ["actual"]

    def test_missing_content_field(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """Chunks with message but no content should yield empty string (filtered)."""
        provider = OllamaProvider(mock_settings)
        mock_stream = [
            {"message": {"role": "assistant"}, "done": False},  # no "content" key
            {"message": {"content": "actual"}, "done": True},
        ]

        with patch.object(provider._client, "chat", return_value=mock_stream):
            result = list(provider.chat(sample_request))

        assert result == ["actual"]

    def test_non_dict_content(
        self, mock_settings: object, sample_request: ChatRequest
    ) -> None:
        """Non-string content in message should yield empty string (filtered)."""
        provider = OllamaProvider(mock_settings)
        mock_stream = [
            {"message": {"content": 12345}, "done": False},  # int, not str
            {"message": {"content": "actual"}, "done": True},
        ]

        with patch.object(provider._client, "chat", return_value=mock_stream):
            result = list(provider.chat(sample_request))

        assert result == ["actual"]


# ---------------------------------------------------------------------------
# 9. Availability checks
# ---------------------------------------------------------------------------
class TestAvailability:
    def test_availability_success(self, mock_settings: object) -> None:
        """is_available should return True when Ollama responds."""
        provider = OllamaProvider(mock_settings)

        with patch.object(provider._client, "list", return_value={"models": []}):
            assert provider.is_available() is True

    def test_availability_failure(self, mock_settings: object) -> None:
        """is_available should return False on connection error without raising."""
        provider = OllamaProvider(mock_settings)

        with patch.object(
            provider._client, "list", side_effect=ConnectionError("refused")
        ):
            assert provider.is_available() is False

    def test_availability_no_exception_leak(self, mock_settings: object) -> None:
        """is_available should never raise, even on unexpected errors."""
        provider = OllamaProvider(mock_settings)

        with patch.object(
            provider._client, "list", side_effect=RuntimeError("unexpected")
        ):
            # Should not raise — returns False
            assert provider.is_available() is False


# ---------------------------------------------------------------------------
# 10. Provider name matches registry expectations
# ---------------------------------------------------------------------------
class TestProviderIdentity:
    def test_provider_name_is_ollama(self, mock_settings: object) -> None:
        """provider_name must match the registry key used for Ollama."""
        provider = OllamaProvider(mock_settings)
        assert provider.provider_name == "ollama"

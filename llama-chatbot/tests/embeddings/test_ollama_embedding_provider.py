"""Unit tests for OllamaEmbeddingProvider.

All tests mock the ``ollama`` SDK — no real Ollama instance is required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider
from src.exceptions import (
    MissingCredentialsError,
    ProviderConnectionError,
    ProviderResponseError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_settings() -> object:
    """Return a minimal settings-like object with required fields."""

    class FakeSettings:
        ollama_host: str = "http://localhost:11434"
        embedding_model: str = "nomic-embed-text"

    return FakeSettings()


@pytest.fixture
def empty_settings() -> object:
    """Return a settings object with empty fields for validation tests."""

    class EmptySettings:
        ollama_host: str = ""
        embedding_model: str = ""

    return EmptySettings()


# ---------------------------------------------------------------------------
# 1. Provider creation
# ---------------------------------------------------------------------------
class TestProviderCreation:
    def test_creation_with_valid_settings(self, mock_settings: object) -> None:
        """Provider should initialise without error given valid settings."""
        provider = OllamaEmbeddingProvider(mock_settings)
        assert provider.provider_name == "ollama"
        assert provider._host == "http://localhost:11434"
        assert provider._model == "nomic-embed-text"

    def test_creation_with_different_model_in_settings(self) -> None:
        """Changing EMBEDDING_MODEL should only affect the model — no code change needed."""

        class AltSettings:
            ollama_host: str = "http://localhost:11434"
            embedding_model: str = "all-minilm"

        provider = OllamaEmbeddingProvider(AltSettings())
        assert provider._model == "all-minilm"

    def test_creation_with_custom_host(self) -> None:
        """Custom OLLAMA_HOST is accepted."""

        class CustomHost:
            ollama_host: str = "http://192.168.1.100:11434"
            embedding_model: str = "nomic-embed-text"

        provider = OllamaEmbeddingProvider(CustomHost())
        assert provider._host == "http://192.168.1.100:11434"

    def test_default_model_when_empty(self) -> None:
        """When embedding_model is empty string, provider stores empty (validated by Settings)."""

        class NoModel:
            ollama_host: str = "http://localhost:11434"
            embedding_model: str = ""

        provider = OllamaEmbeddingProvider(NoModel())
        assert provider._model == ""


# ---------------------------------------------------------------------------
# 2. Configuration validation
# ---------------------------------------------------------------------------
class TestConfigurationValidation:
    def test_valid_configuration_passes(self, mock_settings: object) -> None:
        """Valid settings should not raise."""
        provider = OllamaEmbeddingProvider(mock_settings)
        provider.validate_configuration()  # should not raise

    def test_missing_host_raises(self, empty_settings: object) -> None:
        """Empty host should raise MissingCredentialsError."""
        provider = OllamaEmbeddingProvider(empty_settings)
        with pytest.raises(MissingCredentialsError) as exc_info:
            provider.validate_configuration()
        assert (
            "ollama" in str(exc_info.value).lower()
            or "host" in str(exc_info.value).lower()
        )

    def test_missing_model_raises(self) -> None:
        """Empty embedding model should raise MissingCredentialsError."""

        class NoModel:
            ollama_host: str = "http://localhost:11434"
            embedding_model: str = ""

        provider = OllamaEmbeddingProvider(NoModel())
        with pytest.raises(MissingCredentialsError):
            provider.validate_configuration()


# ---------------------------------------------------------------------------
# 3. Different model names are accepted
# ---------------------------------------------------------------------------
class TestModelNameAcceptance:
    @pytest.mark.parametrize(
        "model_name",
        [
            "nomic-embed-text",
            "all-minilm",
            "mxbai-embed-large",
            "llama3.2:1b",
            "bge-m3",
        ],
    )
    def test_different_model_names(
        self, mock_settings: object, model_name: str
    ) -> None:
        """Any model name should be accepted — no hardcoding."""
        settings_with_model = mock_settings
        # Override model via __setattr__ for the test
        object.__setattr__(settings_with_model, "embedding_model", model_name)

        provider = OllamaEmbeddingProvider(settings_with_model)

        with patch.object(provider._client, "embed") as mock_embed:
            mock_embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
            result = provider.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        # Verify the model from settings was passed to the SDK
        _, kwargs = mock_embed.call_args
        assert kwargs["model"] == model_name


# ---------------------------------------------------------------------------
# 4. Successful embedding
# ---------------------------------------------------------------------------
class TestSuccessfulEmbedding:
    def test_embed_single_text(
        self, mock_settings: object
    ) -> None:
        """embed() should return a single vector."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with patch.object(provider._client, "embed") as mock_embed:
            mock_embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
            result = provider.embed("Hello world")

        assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_multiple_texts(
        self, mock_settings: object
    ) -> None:
        """embed_batch() should return a vector per input text."""
        provider = OllamaEmbeddingProvider(mock_settings)
        texts = ["Hello", "world", "test"]

        with patch.object(provider._client, "embed") as mock_embed:
            mock_embed.return_value = {
                "embeddings": [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                ]
            }
            result = provider.embed_batch(texts)

        assert len(result) == 3
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        # Verify input texts were passed
        _, kwargs = mock_embed.call_args
        assert kwargs["input"] == texts


# ---------------------------------------------------------------------------
# 5. Edge cases — empty inputs
# ---------------------------------------------------------------------------
class TestEmptyInputs:
    def test_empty_string_embed(self, mock_settings: object) -> None:
        """embed() with empty string should still call the API."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with patch.object(provider._client, "embed") as mock_embed:
            mock_embed.return_value = {"embeddings": [[0.0, 0.0]]}
            result = provider.embed("")

        assert result == [0.0, 0.0]

    def test_empty_list_embed_batch(self, mock_settings: object) -> None:
        """embed_batch() with empty list should return empty list."""
        provider = OllamaEmbeddingProvider(mock_settings)
        result = provider.embed_batch([])
        assert result == []

    def test_empty_response_raises(self, mock_settings: object) -> None:
        """Empty response from Ollama (no embeddings) should raise."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with (
            patch.object(provider._client, "embed") as mock_embed,
            pytest.raises(ProviderResponseError) as exc_info,
        ):
            mock_embed.return_value = {"embeddings": []}
            provider.embed("test")

        assert "empty" in exc_info.value.safe_message.lower()


# ---------------------------------------------------------------------------
# 6. Error handling — connection failure
# ---------------------------------------------------------------------------
class TestConnectionFailure:
    def test_connection_error_raises_provider_connection_error(
        self, mock_settings: object
    ) -> None:
        """ConnectionError should be translated to ProviderConnectionError."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with patch.object(
            provider._client, "embed", side_effect=ConnectionError("refused")
        ):
            with pytest.raises(ProviderConnectionError) as exc_info:
                provider.embed("test")

        assert (
            "ollama" in str(exc_info.value).lower()
            or "connect" in str(exc_info.value).lower()
        )

    def test_response_error_raises_provider_response_error(
        self, mock_settings: object
    ) -> None:
        """ollama.ResponseError should be translated to ProviderResponseError."""
        import ollama

        provider = OllamaEmbeddingProvider(mock_settings)
        api_error = ollama.ResponseError(status_code=500, error="Internal server error")

        with patch.object(provider._client, "embed", side_effect=api_error):
            with pytest.raises(ProviderResponseError):
                provider.embed("test")


# ---------------------------------------------------------------------------
# 7. Malformed response handling
# ---------------------------------------------------------------------------
class TestMalformedResponse:
    def test_missing_embeddings_field(self, mock_settings: object) -> None:
        """Response without 'embeddings' field should raise."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with (
            patch.object(provider._client, "embed") as mock_embed,
            pytest.raises(ProviderResponseError),
        ):
            mock_embed.return_value = {}
            provider.embed("test")

    def test_non_list_embeddings(self, mock_settings: object) -> None:
        """Response with non-list embeddings should raise."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with (
            patch.object(provider._client, "embed") as mock_embed,
            pytest.raises(ProviderResponseError),
        ):
            mock_embed.return_value = {"embeddings": "not_a_list"}
            provider.embed("test")


# ---------------------------------------------------------------------------
# 8. Object-style response handling (ollama>=0.4)
# ---------------------------------------------------------------------------
class TestObjectStyleResponse:
    def test_object_response_with_embeddings_attr(
        self, mock_settings: object
    ) -> None:
        """Response as object with .embeddings attribute should work."""
        provider = OllamaEmbeddingProvider(mock_settings)

        class FakeEmbedResponse:
            embeddings: list[list[float]] = [[0.1, 0.2, 0.3]]

        with patch.object(
            provider._client, "embed", return_value=FakeEmbedResponse()
        ):
            result = provider.embed("test")

        assert result == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# 9. Availability checks
# ---------------------------------------------------------------------------
class TestAvailability:
    def test_availability_success(self, mock_settings: object) -> None:
        """is_available should return True when Ollama responds."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with patch.object(provider._client, "list", return_value={"models": []}):
            assert provider.is_available() is True

    def test_availability_failure(self, mock_settings: object) -> None:
        """is_available should return False on connection error without raising."""
        provider = OllamaEmbeddingProvider(mock_settings)

        with patch.object(
            provider._client, "list", side_effect=ConnectionError("refused")
        ):
            assert provider.is_available() is False

    def test_availability_no_exception_leak(self, mock_settings: object) -> None:
        """is_available should never raise, even on unexpected errors."""
        provider = OllamaEmbeddingProvider(mock_settings)

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
        provider = OllamaEmbeddingProvider(mock_settings)
        assert provider.provider_name == "ollama"

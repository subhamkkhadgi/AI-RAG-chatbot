"""Unit tests for ChatService orchestration.

All tests use a mocked provider — no real API calls, no network,
no API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.exceptions import (
    ChatbotError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
)
from src.models.chat import ChatRequest, ChatRole, Conversation
from src.prompts.system_prompts import get_default_prompt
from src.providers.base import BaseLLMProvider
from src.services.chat_service import ChatService


# ======================================================================
# Helpers
# ======================================================================
def _make_mock_provider(
    provider_name: str = "mock",
    chunks: list[str] | None = None,
    raise_exc: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock that satisfies ``BaseLLMProvider``."""
    mock = MagicMock(spec=BaseLLMProvider)
    mock.provider_name = provider_name

    if raise_exc:
        mock.chat.side_effect = raise_exc
    else:
        _chunks = chunks or ["Response from mock!"]
        mock.chat.side_effect = lambda _request: iter(_chunks)

    mock.validate_configuration.return_value = None
    mock.is_available.return_value = True
    return mock


def _make_service(
    mock_provider: MagicMock | None = None,
    model: str = "mock-model",
    **kwargs,
) -> ChatService:
    provider = mock_provider or _make_mock_provider()
    return ChatService(provider=provider, model=model, **kwargs)


# ======================================================================
# Tests
# ======================================================================


class TestServiceCreation:
    """Verify ChatService is correctly constructed."""

    def test_creation_with_mock_provider(self) -> None:
        """A valid provider should be accepted."""
        provider = _make_mock_provider()
        service = ChatService(provider=provider, model="test-model")
        assert service is not None

    def test_creation_rejects_non_provider(self) -> None:
        """Passing an object that does not implement BaseLLMProvider
        should raise TypeError."""
        with pytest.raises(TypeError):
            ChatService(provider="not-a-provider", model="x")  # type: ignore[arg-type]

    def test_default_system_prompt_used_when_empty(self) -> None:
        """If no system_prompt is given, the default should be used."""
        service = _make_service()
        assert service._system_prompt == get_default_prompt()

    def test_custom_system_prompt_override(self) -> None:
        """A custom system_prompt should override the default."""
        custom = "You are a test bot."
        service = _make_service(system_prompt=custom)
        assert service._system_prompt == custom

    def test_model_property(self) -> None:
        """The model property should return the configured model."""
        service = _make_service(model="custom-model")
        assert service.model == "custom-model"

    def test_provider_property(self) -> None:
        """The provider property should return the injected provider."""
        provider = _make_mock_provider()
        service = ChatService(provider=provider, model="x")
        assert service.provider is provider


class TestSendMessage:
    """Core message-sending flow."""

    def test_successful_chat_returns_response(self) -> None:
        """send_message should return the full assistant response."""
        provider = _make_mock_provider(chunks=["Hello ", "world!"])
        service = _make_service(provider)
        conversation = Conversation()
        result = service.send_message(conversation, "Hi")
        assert result == "Hello world!"

    def test_user_message_added_once(self) -> None:
        """The user message should appear exactly once in the
        conversation."""
        provider = _make_mock_provider()
        service = _make_service(provider)
        conversation = Conversation()
        service.send_message(conversation, "Hello")
        user_msgs = [m for m in conversation.messages if m.role == ChatRole.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Hello"

    def test_assistant_message_added_once(self) -> None:
        """The assistant message should appear exactly once."""
        provider = _make_mock_provider(chunks=["Response"])
        service = _make_service(provider)
        conversation = Conversation()
        service.send_message(conversation, "Hi")
        assistant_msgs = [
            m for m in conversation.messages if m.role == ChatRole.ASSISTANT
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "Response"

    def test_provider_receives_correct_chat_request(self) -> None:
        """The ChatRequest passed to the provider should contain the
        correct values."""
        provider = _make_mock_provider()
        service = _make_service(
            provider,
            model="test-model",
            system_prompt="Be nice",
            temperature=0.5,
            max_tokens=100,
        )
        conversation = Conversation()
        conversation.add_user_message("Pre-existing message")
        service.send_message(conversation, "New message")

        call_args = provider.chat.call_args
        assert call_args is not None
        request: ChatRequest = call_args[0][0]

        assert request.model == "test-model"
        assert request.system_prompt == "Be nice"
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert len(request.messages) == 2

    def test_system_prompt_remains_separate(self) -> None:
        """The system_prompt field should be separate from
        messages."""
        provider = _make_mock_provider()
        service = _make_service(provider, system_prompt="System prompt")
        conversation = Conversation()
        service.send_message(conversation, "Hello")

        call_args = provider.chat.call_args
        assert call_args is not None
        request: ChatRequest = call_args[0][0]

        for msg in request.messages:
            assert msg.role != ChatRole.SYSTEM
        assert request.system_prompt == "System prompt"

    def test_message_order_preserved(self) -> None:
        """Multiple messages should maintain chronological order."""
        provider = _make_mock_provider(chunks=["Response"])
        service = _make_service(provider)
        conversation = Conversation()

        service.send_message(conversation, "First")
        service.send_message(conversation, "Second")
        service.send_message(conversation, "Third")

        contents = [m.content for m in conversation.messages]
        assert contents == [
            "First",
            "Response",
            "Second",
            "Response",
            "Third",
            "Response",
        ]

    def test_empty_input_raises_configuration_error(self) -> None:
        """Sending an empty message should raise ConfigurationError."""
        service = _make_service()
        conversation = Conversation()
        with pytest.raises(ConfigurationError):
            service.send_message(conversation, "")

    def test_whitespace_only_input_raises_configuration_error(self) -> None:
        """Sending whitespace-only input should raise
        ConfigurationError."""
        service = _make_service()
        conversation = Conversation()
        with pytest.raises(ConfigurationError):
            service.send_message(conversation, "   ")

    def test_no_user_message_added_when_input_invalid(self) -> None:
        """If validation fails, no user message should be added."""
        service = _make_service()
        conversation = Conversation()
        with pytest.raises(ConfigurationError):
            service.send_message(conversation, "")
        assert len(conversation.messages) == 0


class TestStreamMessage:
    """Streaming message flow."""

    def test_stream_yields_chunks(self) -> None:
        """stream_message should yield each non-empty chunk."""
        provider = _make_mock_provider(chunks=["A", "B", "C"])
        service = _make_service(provider)
        conversation = Conversation()
        result = list(service.stream_message(conversation, "Hi"))
        assert result == ["A", "B", "C"]

    def test_stream_ignores_empty_chunks(self) -> None:
        """Empty or whitespace-only chunks should be filtered out."""
        provider = _make_mock_provider(chunks=["", "  ", "Hello", "", " world", "  "])
        service = _make_service(provider)
        conversation = Conversation()
        result = list(service.stream_message(conversation, "Hi"))
        assert result == ["Hello", " world"]

    def test_stream_adds_assistant_message(self) -> None:
        """After streaming completes, the assistant message should be
        added to the conversation."""
        provider = _make_mock_provider(chunks=["Hello", " world"])
        service = _make_service(provider)
        conversation = Conversation()

        for _ in service.stream_message(conversation, "Hi"):
            pass

        assistant_msgs = [
            m for m in conversation.messages if m.role == ChatRole.ASSISTANT
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "Hello world"

    def test_stream_adds_user_message(self) -> None:
        """Before streaming, the user message should be added."""
        provider = _make_mock_provider(chunks=["Hi there"])
        service = _make_service(provider)
        conversation = Conversation()

        for _ in service.stream_message(conversation, "Hello"):
            pass

        user_msgs = [m for m in conversation.messages if m.role == ChatRole.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Hello"


class TestErrorPropagation:
    """Exception handling."""

    def test_provider_connection_error_propagated(self) -> None:
        """ProviderConnectionError from provider should propagate."""
        provider = _make_mock_provider(
            raise_exc=ProviderConnectionError(
                "mock",
                safe_message="Cannot connect",
            )
        )
        service = _make_service(provider)
        conversation = Conversation()
        with pytest.raises(ProviderConnectionError):
            service.send_message(conversation, "Hi")

    def test_provider_response_error_propagated(self) -> None:
        """ProviderResponseError from provider should propagate."""
        provider = _make_mock_provider(
            raise_exc=ProviderResponseError(
                "mock",
                safe_message="Bad response",
            )
        )
        service = _make_service(provider)
        conversation = Conversation()
        with pytest.raises(ProviderResponseError):
            service.send_message(conversation, "Hi")

    def test_chatbot_error_propagated(self) -> None:
        """Generic ChatbotError from provider should propagate."""
        provider = _make_mock_provider(
            raise_exc=ChatbotError(
                "Something went wrong",
                safe_message="Try again",
            )
        )
        service = _make_service(provider)
        conversation = Conversation()
        with pytest.raises(ChatbotError):
            service.send_message(conversation, "Hi")

    def test_unexpected_exception_wrapped(self) -> None:
        """Unexpected exceptions should be wrapped in
        ProviderResponseError."""
        provider = _make_mock_provider(raise_exc=RuntimeError("Unexpected"))
        service = _make_service(provider)
        conversation = Conversation()
        with pytest.raises(ProviderResponseError):
            service.send_message(conversation, "Hi")

    def test_error_does_not_add_assistant_message(self) -> None:
        """If provider raises, no assistant message should be added."""
        provider = _make_mock_provider(
            raise_exc=ProviderConnectionError(
                "mock",
                safe_message="Cannot connect",
            )
        )
        service = _make_service(provider)
        conversation = Conversation()
        with pytest.raises(ProviderConnectionError):
            service.send_message(conversation, "Hi")
        assistant_msgs = [
            m for m in conversation.messages if m.role == ChatRole.ASSISTANT
        ]
        assert len(assistant_msgs) == 0


class TestConversationHistory:
    """Conversation history preservation."""

    def test_conversation_preserved_across_calls(self) -> None:
        """Messages from earlier turns should remain in the
        conversation."""
        provider = _make_mock_provider()
        service = _make_service(provider)
        conversation = Conversation()
        conversation.add_user_message("Pre-existing")

        service.send_message(conversation, "New")

        contents = [m.content for m in conversation.messages]
        assert "Pre-existing" in contents
        assert "New" in contents

    def test_multiple_turns_maintain_exchange_structure(self) -> None:
        """After multiple turns, the conversation should follow a
        strict user/assistant pattern."""
        provider = _make_mock_provider(chunks=["Reply"])
        service = _make_service(provider)
        conversation = Conversation()

        for i in range(3):
            service.send_message(conversation, f"Message {i}")

        roles = [m.role for m in conversation.messages]
        expected = [
            ChatRole.USER,
            ChatRole.ASSISTANT,
            ChatRole.USER,
            ChatRole.ASSISTANT,
            ChatRole.USER,
            ChatRole.ASSISTANT,
        ]
        assert roles == expected


class TestNoSDKDependencies:
    """Verify ChatService has no SDK coupling."""

    def test_no_streamlit_import(self) -> None:
        """ChatService must not import Streamlit."""
        import src.services.chat_service as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "streamlit" not in content.lower()

    def test_no_ollama_import(self) -> None:
        """ChatService must not import ollama."""
        import src.services.chat_service as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "ollama" not in content.lower()

    def test_no_groq_import(self) -> None:
        """ChatService must not import groq."""
        import src.services.chat_service as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "groq" not in content.lower()


class TestNoMutableDefaults:
    """Verify no mutable default arguments."""

    def test_conversation_argument_is_not_default(self) -> None:
        """Conversation should be explicitly passed, not a default."""
        import inspect

        sig = inspect.signature(ChatService.send_message)
        params = list(sig.parameters.keys())
        assert "conversation" in params
        assert sig.parameters["conversation"].default is inspect.Parameter.empty

    def test_stream_method_requires_conversation(self) -> None:
        """stream_message should require conversation, not use a
        default."""
        import inspect

        sig = inspect.signature(ChatService.stream_message)
        params = list(sig.parameters.keys())
        assert "conversation" in params
        assert sig.parameters["conversation"].default is inspect.Parameter.empty

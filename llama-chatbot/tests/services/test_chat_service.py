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
from src.rag.rag_service import RAGResult
from src.retrieval.models import RetrievedChunk, RetrievalResult
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


# ======================================================================
# RAG Integration Tests
# ======================================================================

class TestRAGServiceCreation:
    """Verify ChatService correctly accepts RAGService."""

    def test_rag_service_defaults_to_none(self) -> None:
        """When rag_service is not provided, it should be None."""
        service = _make_service()
        assert service.rag_service is None

    def test_rag_service_property(self) -> None:
        """The rag_service property should return the injected instance."""
        provider = _make_mock_provider()
        rag_service = MagicMock()
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        assert service.rag_service is rag_service

    def test_rag_service_is_optional(self) -> None:
        """Creating ChatService without rag_service should work."""
        service = _make_service()
        assert service is not None
        assert service.rag_service is None


class TestRAGSendMessage:
    """RAG integration in send_message."""

    def test_rag_service_called_when_provided(self) -> None:
        """When RAGService is provided, it should be called."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = MagicMock()
        rag_service.query.return_value.query = "test query"
        rag_service.query.return_value.context = "Relevant document context."
        rag_service.query.return_value.retrieval_result = MagicMock()

        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        service.send_message(conversation, "test query")

        rag_service.query.assert_called_once_with("test query")

    def test_rag_context_prepended_to_user_message(self) -> None:
        """When RAGService returns context, it should be prepended to the
        user message."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = MagicMock()
        rag_service.query.return_value.query = "test query"
        rag_service.query.return_value.context = "Document context."
        rag_service.query.return_value.retrieval_result = MagicMock()

        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        service.send_message(conversation, "test query")

        user_msg = conversation.messages[0]
        assert "Relevant context:" in user_msg.content
        assert "Document context." in user_msg.content
        assert "Question:" in user_msg.content
        assert "test query" in user_msg.content

    def test_rag_not_used_when_not_provided(self) -> None:
        """Without RAGService, the user message should remain unchanged."""
        provider = _make_mock_provider(chunks=["Answer"])
        service = _make_service(provider)
        conversation = Conversation()
        service.send_message(conversation, "Hello")

        user_msg = conversation.messages[0]
        assert user_msg.content == "Hello"

    def test_rag_failure_falls_back_quietly(self) -> None:
        """When RAG query fails, the original message should be used."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = MagicMock()
        rag_service.query.side_effect = ChatbotError("RAG failed")

        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        service.send_message(conversation, "test query")


# ======================================================================
# Citation Tests
# ======================================================================

def _make_rag_chunk(
    filename: str = "report.pdf",
    text: str = "Relevant content.",
    page_number: int | None = None,
    chunk_index: int = 0,
    document_id: str = "doc-1",
    score: float = 0.95,
) -> RetrievedChunk:
    """Build a real ``RetrievedChunk`` for citation tests."""
    return RetrievedChunk(
        chunk_id=f"chunk-{chunk_index}",
        document_id=document_id,
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
        page_number=page_number,
    )


def _make_rag_result_with_chunks(
    chunks: list[RetrievedChunk] | None,
) -> RAGResult:
    """Build a real ``RAGResult`` with the given chunks."""
    retrieval_result = RetrievalResult(
        query="test query",
        chunks=chunks or [],
        total_results=len(chunks or []),
    )
    return RAGResult(
        query="test query",
        retrieval_result=retrieval_result,
        context="Relevant document context.",
    )


def _make_rag_service(rag_result: RAGResult) -> MagicMock:
    """Build a mocked RAGService returning *rag_result*."""
    rag_service = MagicMock()
    rag_service.query.return_value = rag_result
    return rag_service


class TestCitationsAppended:
    """Citations should be appended when retrieval succeeds."""

    def test_send_message_appends_citations(self) -> None:
        """send_message should append a Sources section to the response."""
        provider = _make_mock_provider(chunks=["The answer."])
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks(
                [_make_rag_chunk(filename="report.pdf", page_number=3)]
            )
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result == "The answer.\n\nSources:\n- report.pdf (Page 3)"
        assert conversation.messages[-1].content == result

    def test_stream_message_appends_citations_to_conversation(self) -> None:
        """stream_message should store the cited text in the conversation."""
        provider = _make_mock_provider(chunks=["Streamed ", "answer."])
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks(
                [_make_rag_chunk(filename="notes.txt", page_number=None)]
            )
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()

        yielded = list(service.stream_message(conversation, "test query"))

        # Streaming chunks themselves are unchanged.
        assert yielded == ["Streamed ", "answer."]
        # The stored assistant message includes citations.
        assert conversation.messages[-1].content == (
            "Streamed answer.\n\nSources:\n- notes.txt"
        )

    def test_citations_deduplicated(self) -> None:
        """Duplicate filename+page citations appear only once."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks(
                [
                    _make_rag_chunk(
                        filename="report.pdf", page_number=2, chunk_index=0
                    ),
                    _make_rag_chunk(
                        filename="report.pdf", page_number=2, chunk_index=1
                    ),
                    _make_rag_chunk(
                        filename="report.pdf", page_number=3, chunk_index=2
                    ),
                ]
            )
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result.count("- report.pdf (Page 2)") == 1
        assert result.count("- report.pdf (Page 3)") == 1

    def test_last_rag_result_preserves_metadata(self) -> None:
        """last_rag_result should expose full retrieval metadata."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_chunk = _make_rag_chunk(
            filename="report.pdf",
            page_number=3,
            chunk_index=2,
            document_id="internal-doc",
            score=0.987,
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        service.send_message(conversation, "test query")

        last = service.last_rag_result
        assert last is not None
        assert last.retrieval_result.chunks[0].document_id == "internal-doc"
        assert last.retrieval_result.chunks[0].chunk_index == 2
        assert last.retrieval_result.chunks[0].score == 0.987
        assert last.retrieval_result.chunks[0].filename == "report.pdf"
        assert last.retrieval_result.chunks[0].page_number == 3


class TestCitationsOmitted:
    """Citations should be omitted when RAG is disabled / fails / empty."""

    def test_no_citations_when_rag_disabled(self) -> None:
        """Without RAG, the response should have no Sources section."""
        provider = _make_mock_provider(chunks=["Answer"])
        service = _make_service(provider)
        conversation = Conversation()
        result = service.send_message(conversation, "Hello")

        assert result == "Answer"
        assert "Sources:" not in result
        assert service.last_rag_result is None

    def test_no_citations_when_rag_fails(self) -> None:
        """When RAG query fails, no Sources section should appear."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = MagicMock()
        rag_service.query.side_effect = ChatbotError("RAG failed")

        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result == "Answer"
        assert "Sources:" not in result
        assert service.last_rag_result is None

    def test_no_citations_when_no_chunks(self) -> None:
        """When retrieval returns no chunks, no Sources section."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = _make_rag_service(_make_rag_result_with_chunks([]))
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result == "Answer"
        assert "Sources:" not in result

    def test_no_citations_when_context_empty(self) -> None:
        """When RAG context is empty, the response should stay unchanged."""
        provider = _make_mock_provider(chunks=["Answer"])
        # A MagicMock RAG result with empty context (retrieval_result is
        # a MagicMock so the citation builder must safely ignore it).
        rag_service = MagicMock()
        rag_service.query.return_value.query = "test query"
        rag_service.query.return_value.context = ""
        rag_service.query.return_value.retrieval_result = MagicMock()

        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result == "Answer"
        assert "Sources:" not in result


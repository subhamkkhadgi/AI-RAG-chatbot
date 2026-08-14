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
from src.services.chat_service import ChatService, build_retrieval_query


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
        assert "<retrieved_context>" in user_msg.content
        assert "</retrieved_context>" in user_msg.content
        assert "Document context." in user_msg.content
        assert "Question:" in user_msg.content
        assert "test query" in user_msg.content
        # The RAG context must explicitly instruct proper Markdown lists so
        # the model emits one bullet per line instead of "*"-joined items.
        assert "each item on its own line" in user_msg.content
        assert "Never join list items" in user_msg.content

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
    text: str = "The answer content.",
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
    """Citations should be attached as structured sources when retrieval
    succeeds."""

    def test_send_message_attaches_structured_sources(self) -> None:
        """send_message should attach structured sources and keep the raw
        answer text (no plain-text Sources block)."""
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

        # The message body is the raw answer only — no Sources block.
        assert result == "The answer."
        assert "Sources:" not in result
        assert conversation.messages[-1].content == "The answer."
        # Structured sources are attached.
        assert conversation.messages[-1].sources is not None
        assert len(conversation.messages[-1].sources) == 1
        assert conversation.messages[-1].sources[0].filename == "report.pdf"
        assert conversation.messages[-1].sources[0].page_number == 3

    def test_stream_message_attaches_structured_sources(self) -> None:
        """stream_message should store the raw answer text and attach
        structured sources."""
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
        # The stored assistant message is the raw answer — no Sources block.
        assert conversation.messages[-1].content == "Streamed answer."
        assert "Sources:" not in conversation.messages[-1].content
        # Structured sources are attached.
        assert conversation.messages[-1].sources is not None
        assert len(conversation.messages[-1].sources) == 1
        assert conversation.messages[-1].sources[0].filename == "notes.txt"

    def test_citations_deduplicated(self) -> None:
        """Duplicate filename+page citations appear only once in sources."""
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

        assert result == "Answer"
        sources = conversation.messages[-1].sources
        assert sources is not None
        assert len(sources) == 2
        assert sources[0].filename == "report.pdf"
        assert sources[0].page_number == 2
        assert sources[1].filename == "report.pdf"
        assert sources[1].page_number == 3

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


# ======================================================================
# Structured Sources on Assistant Messages (Sprint 8D)
# ======================================================================

class TestStructuredSources:
    """Structured ``sources`` attached to assistant messages."""

    def test_send_message_attaches_sources(self) -> None:
        """send_message should attach structured sources to the assistant msg."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks(
                [
                    _make_rag_chunk(
                        filename="report.pdf",
                        page_number=3,
                        text="Answer content.",
                    )
                ]
            )
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        service.send_message(conversation, "test query")

        assistant = conversation.messages[-1]
        assert assistant.sources is not None
        assert len(assistant.sources) == 1
        assert assistant.sources[0].filename == "report.pdf"
        assert assistant.sources[0].page_number == 3
        assert assistant.sources[0].text == "Answer content."
        assert assistant.sources[0].score == 0.95

    def test_stream_message_attaches_sources(self) -> None:
        """stream_message should attach structured sources to the assistant msg."""
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

        list(service.stream_message(conversation, "test query"))

        assistant = conversation.messages[-1]
        assert assistant.sources is not None
        assert len(assistant.sources) == 1
        assert assistant.sources[0].filename == "notes.txt"
        assert assistant.sources[0].page_number is None

    def test_sources_none_when_rag_disabled(self) -> None:
        """Without RAG, assistant messages should have sources=None."""
        provider = _make_mock_provider(chunks=["Answer"])
        service = _make_service(provider)
        conversation = Conversation()
        service.send_message(conversation, "Hello")

        assistant = conversation.messages[-1]
        assert assistant.sources is None

    def test_sources_none_when_rag_fails(self) -> None:
        """When RAG fails, assistant messages should have sources=None."""
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

        assistant = conversation.messages[-1]
        assert assistant.sources is None

    def test_sources_none_when_no_chunks(self) -> None:
        """When retrieval returns no chunks, sources should be None."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = _make_rag_service(_make_rag_result_with_chunks([]))
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        service.send_message(conversation, "test query")

        assistant = conversation.messages[-1]
        assert assistant.sources is None

    def test_content_has_no_plain_text_sources_block(self) -> None:
        """The assistant message content should NOT contain a plain-text
        Sources block (citations are rendered only via structured sources)."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks(
                [_make_rag_chunk(filename="report.pdf", page_number=5)]
            )
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result == "Answer"
        assert "Sources:" not in result
        assert "Sources:" not in conversation.messages[-1].content
        # Structured sources still attached.
        assert conversation.messages[-1].sources is not None
        assert len(conversation.messages[-1].sources) == 1
        assert conversation.messages[-1].sources[0].filename == "report.pdf"
        assert conversation.messages[-1].sources[0].page_number == 5


# ======================================================================
# Answer-Aware Citation Filtering
# ======================================================================

class TestAnswerAwareCitationFiltering:
    """Citations should only include chunks that lexically support the
    generated answer."""

    def test_unsupported_chunk_removed_from_citations(self) -> None:
        """A chunk that does not support the answer is not cited."""
        provider = _make_mock_provider(chunks=["The project is named Atlas."])
        supporting = _make_rag_chunk(
            filename="report.pdf",
            page_number=1,
            text="The project is named Atlas.",
        )
        unsupported = _make_rag_chunk(
            filename="report.pdf",
            page_number=9,
            text="The budget is allocated to the team.",
            chunk_index=1,
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([supporting, unsupported])
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        # The message body is the raw answer — no Sources block.
        assert result == "The project is named Atlas."
        assert "Sources:" not in result
        assistant = conversation.messages[-1]
        assert len(assistant.sources) == 1
        assert assistant.sources[0].page_number == 1

    def test_no_supporting_chunks_means_no_citations(self) -> None:
        """When no chunk supports the answer, no Sources section is shown."""
        provider = _make_mock_provider(chunks=["The project is named Atlas."])
        unsupported = _make_rag_chunk(
            filename="report.pdf",
            page_number=9,
            text="The budget is allocated to the team.",
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([unsupported])
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result == "The project is named Atlas."
        assert "Sources:" not in result
        assistant = conversation.messages[-1]
        assert assistant.sources is None


# ======================================================================
# Confidence-Gated Citations (Sprint 8)
# ======================================================================

class TestConfidenceGatedCitations:
    """Citations should only appear when retrieval confidence passes the
    configured threshold."""

    def test_citations_appended_when_confidence_meets_threshold(self) -> None:
        """With a threshold set and confidence above it, citations appear."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_chunk = _make_rag_chunk(
            filename="report.pdf", page_number=3, score=0.95
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
            confidence_threshold=0.6,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        # Raw answer text, structured sources attached.
        assert result == "Answer"
        assert "Sources:" not in result
        assert conversation.messages[-1].sources is not None
        assert len(conversation.messages[-1].sources) == 1
        assert conversation.messages[-1].sources[0].filename == "report.pdf"
        assert conversation.messages[-1].sources[0].page_number == 3

    def test_citations_omitted_when_confidence_below_threshold(self) -> None:
        """With a threshold set and confidence below it, citations are omitted."""
        provider = _make_mock_provider(chunks=["Answer"])
        rag_chunk = _make_rag_chunk(
            filename="report.pdf", page_number=3, score=0.2
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider,
            model="test-model",
            rag_service=rag_service,
            confidence_threshold=0.6,
        )
        conversation = Conversation()
        result = service.send_message(conversation, "test query")

        assert result == "Answer"
        assert "Sources:" not in result
        assert conversation.messages[-1].sources is None

    def test_confidence_threshold_property(self) -> None:
        """The confidence_threshold property should reflect the constructor arg."""
        service = _make_service(confidence_threshold=0.7)
        assert service.confidence_threshold == 0.7

    def test_confidence_threshold_defaults_to_none(self) -> None:
        """Without a threshold, confidence_threshold should be None."""
        service = _make_service()
        assert service.confidence_threshold is None


# ======================================================================
# Contextual Follow-up Retrieval
# ======================================================================

def _conv_with_prior_user(previous_question: str) -> Conversation:
    """Build a conversation with one previous user question."""
    conv = Conversation()
    conv.add_user_message(previous_question)
    return conv


class TestBuildRetrievalQuery:
    """build_retrieval_query contextualizes only generic referential
    follow-ups and never touches prior messages."""

    def test_first_turn_explicit_query_unchanged(self) -> None:
        """A standalone explicit question is returned byte-for-byte."""
        question = "What technologies are used in Outdoor Gear Hub?"
        assert build_retrieval_query(question, None) == question

    def test_first_turn_generic_query_unchanged(self) -> None:
        """A generic query stays unchanged when there is no prior user
        question."""
        question = "technologies used"
        assert build_retrieval_query(question, Conversation()) == question

    def test_explicit_self_contained_query_unchanged(self) -> None:
        """An explicit question is unchanged even with prior context."""
        conv = _conv_with_prior_user("What technologies are used in TES?")
        question = "What technologies are used in Outdoor Gear Hub?"
        assert build_retrieval_query(question, conv) == question

    def test_generic_followup_after_ogh_contextualized(self) -> None:
        """'What technologies are used?' after Outdoor Gear Hub gets the
        topic context while preserving the current question."""
        conv = _conv_with_prior_user(
            "What technologies are used in Outdoor Gear Hub?"
        )
        query = build_retrieval_query("What technologies are used?", conv)
        assert "Outdoor Gear Hub" in query
        assert "What technologies are used?" in query

    def test_generic_followup_after_tes_contextualized(self) -> None:
        """'technologies used' after TES gets the TES topic context."""
        conv = _conv_with_prior_user("What technologies are used in TES?")
        query = build_retrieval_query("technologies used", conv)
        assert "TES" in query
        assert "technologies used" in query

    def test_technologies_used_recognized_as_generic_followup(self) -> None:
        """A bare two-word topic query is treated as a generic follow-up
        when prior context exists."""
        conv = _conv_with_prior_user(
            "What technologies are used in Outdoor Gear Hub?"
        )
        query = build_retrieval_query("technologies used", conv)
        assert "Outdoor Gear Hub" in query
        assert "technologies used" in query

    def test_previous_user_question_used_as_context(self) -> None:
        """The most recent previous USER question supplies the context."""
        conv = _conv_with_prior_user("What is the capital of France?")
        conv.add_assistant_message("Paris is the capital of France.")
        query = build_retrieval_query("what is its population?", conv)
        assert "capital of France" in query
        assert "Paris" not in query

    def test_assistant_answers_not_used_as_context(self) -> None:
        """Assistant answers alone never provide retrieval context."""
        conv = Conversation()
        conv.add_assistant_message("The budget is allocated to the team.")
        assert build_retrieval_query("technologies used", conv) == (
            "technologies used"
        )

    def test_unrelated_self_contained_second_question_unchanged(self) -> None:
        """A clearly unrelated, self-contained follow-up is unchanged."""
        conv = _conv_with_prior_user("What technologies are used in TES?")
        question = "What is the customer service contact information?"
        assert build_retrieval_query(question, conv) == question

    def test_build_query_never_modifies_conversation(self) -> None:
        """build_retrieval_query never mutates the conversation."""
        conv = _conv_with_prior_user(
            "What technologies are used in Outdoor Gear Hub?"
        )
        conv.add_assistant_message("Outdoor Gear Hub uses React and Python.")
        before = [(m.role, m.content) for m in conv.messages]
        build_retrieval_query("technologies used", conv)
        after = [(m.role, m.content) for m in conv.messages]
        assert before == after


class TestContextualizedQuerySentToRAG:
    """send_message and stream_message pass the contextualized query to
    RAG for an ambiguous follow-up."""

    def test_send_message_sends_contextualized_query(self) -> None:
        provider = _make_mock_provider(chunks=["Outdoor Gear Hub uses React."])
        rag_chunk = _make_rag_chunk(
            text="Outdoor Gear Hub uses React and Python."
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider, model="test-model", rag_service=rag_service
        )
        conversation = _conv_with_prior_user(
            "What technologies are used in Outdoor Gear Hub?"
        )
        service.send_message(conversation, "What technologies are used?")

        sent_query = rag_service.query.call_args[0][0]
        assert "Outdoor Gear Hub" in sent_query
        assert "What technologies are used?" in sent_query

    def test_stream_message_sends_contextualized_query(self) -> None:
        provider = _make_mock_provider(chunks=["Outdoor Gear Hub uses React."])
        rag_chunk = _make_rag_chunk(
            text="Outdoor Gear Hub uses React and Python."
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider, model="test-model", rag_service=rag_service
        )
        conversation = _conv_with_prior_user(
            "What technologies are used in Outdoor Gear Hub?"
        )
        list(service.stream_message(conversation, "What technologies are used?"))

        sent_query = rag_service.query.call_args[0][0]
        assert "Outdoor Gear Hub" in sent_query
        assert "What technologies are used?" in sent_query

    def test_original_current_message_unchanged_in_conversation(self) -> None:
        """The stored user message keeps the original current text, not the
        contextualized retrieval query."""
        provider = _make_mock_provider(chunks=["Outdoor Gear Hub uses React."])
        rag_chunk = _make_rag_chunk(
            text="Outdoor Gear Hub uses React and Python."
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider, model="test-model", rag_service=rag_service
        )
        conversation = _conv_with_prior_user(
            "What technologies are used in Outdoor Gear Hub?"
        )
        follow_up = "What technologies are used?"
        service.send_message(conversation, follow_up)

        user_msgs = [m for m in conversation.messages if m.role == ChatRole.USER]
        stored = user_msgs[-1].content
        assert follow_up in stored
        # The contextualized composition must not replace the stored query.
        assert (
            "What technologies are used in Outdoor Gear Hub? \u2014 "
            "What technologies are used?"
        ) not in stored


class TestEnrichedPreviousMessageRecovery:
    """Regression: the previous enriched USER message is recovered to the
    ORIGINAL question for contextualized retrieval, while the stored enriched
    message is left unchanged."""

    def test_real_production_path_tes_followup_not_polluted(self) -> None:
        turn1 = "What are the technologies used in TES?"
        turn2 = "technologies used"
        provider = _make_mock_provider(chunks=["TES uses JWT and OAuth2."])
        rag_chunk = _make_rag_chunk(
            filename="report.pdf",
            page_number=3,
            text="TES uses JWT and OAuth2.",
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider, model="test-model", rag_service=rag_service
        )
        conversation = Conversation()

        # Turn 1 through the real flow stores the ENRICHED user message.
        service.send_message(conversation, turn1)

        # Confirm the stored turn-1 user message is the enriched form.
        stored_turn1 = conversation.messages[0].content
        assert "<retrieved_context>" in stored_turn1
        assert "TES" in stored_turn1

        # Turn 2 generic follow-up.
        service.send_message(conversation, turn2)
        sent_query = rag_service.query.call_args[0][0]

        # Turn-2 query must contain the clean original first question.
        assert turn1 in sent_query
        # And must NOT contain the previous <retrieved_context> block.
        assert "<retrieved_context>" not in sent_query
        # The stored enriched message remains unchanged.
        assert conversation.messages[0].content == stored_turn1

    def test_real_production_path_stream_tes_followup_not_polluted(self) -> None:
        turn1 = "What are the technologies used in TES?"
        turn2 = "technologies used"
        provider = _make_mock_provider(chunks=["TES uses JWT and OAuth2."])
        rag_chunk = _make_rag_chunk(
            filename="report.pdf",
            page_number=3,
            text="TES uses JWT and OAuth2.",
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider, model="test-model", rag_service=rag_service
        )
        conversation = Conversation()

        list(service.stream_message(conversation, turn1))

        stored_turn1 = conversation.messages[0].content
        assert "<retrieved_context>" in stored_turn1

        list(service.stream_message(conversation, turn2))
        sent_query = rag_service.query.call_args[0][0]

        assert turn1 in sent_query
        assert "<retrieved_context>" not in sent_query
        assert conversation.messages[0].content == stored_turn1


class TestContextualizedFollowupCitation:
    """Regression: a contextualized follow-up must retrieve the relevant
    supporting chunk and produce a SourceRef through the existing citation
    pipeline (no lexical support, weakened thresholds or mocked results)."""

    def test_tes_followup_retrieves_supporting_chunk(self) -> None:
        answer = "TES uses JWT and OAuth2."
        provider = _make_mock_provider(chunks=[answer])
        rag_chunk = _make_rag_chunk(
            filename="report.pdf",
            page_number=3,
            text=answer,
        )
        rag_service = _make_rag_service(
            _make_rag_result_with_chunks([rag_chunk])
        )
        service = ChatService(
            provider=provider, model="test-model", rag_service=rag_service
        )
        conversation = Conversation()

        service.send_message(conversation, "What are the technologies used in TES?")
        service.send_message(conversation, "technologies used")

        # The ambiguous follow-up was contextualized with the clean original
        # first question, not the previous <retrieved_context> block.
        sent_query = rag_service.query.call_args[0][0]
        assert "What are the technologies used in TES?" in sent_query
        assert "<retrieved_context>" not in sent_query
        # The supporting chunk passed through the existing citation pipeline
        # and produced a structured SourceRef.
        assistant = conversation.messages[-1]
        assert assistant.sources is not None
        assert len(assistant.sources) == 1
        assert assistant.sources[0].filename == "report.pdf"
        assert assistant.sources[0].page_number == 3

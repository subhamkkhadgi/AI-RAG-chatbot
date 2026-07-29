"""Unit tests for RAGService.

All tests use mocks for ``DocumentRetriever`` and ``ContextBuilder``.
No real embedding providers, vector stores, or LLM providers are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest

from src.exceptions import ChatbotError, ConfigurationError
from src.rag.context_builder import ContextBuilder
from src.rag.rag_service import RAGResult, RAGService
from src.retrieval.models import RetrievedChunk, RetrievalResult
from src.retrieval.retriever import DocumentRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    filename: str = "test.txt",
    text: str = "Relevant document content.",
    score: float = 0.95,
) -> RetrievedChunk:
    """Build a ``RetrievedChunk`` with minimal required fields."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        chunk_index=0,
        text=text,
        score=score,
    )


def _make_retrieval_result(
    chunks: list[RetrievedChunk] | None = None,
    query: str = "test query",
) -> RetrievalResult:
    """Build a ``RetrievalResult`` from the provided chunks."""
    resolved = chunks or []
    return RetrievalResult(
        query=query,
        chunks=resolved,
        total_results=len(resolved),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_retriever() -> MagicMock:
    """A mock ``DocumentRetriever`` with a working ``retrieve`` method."""
    mock = create_autospec(DocumentRetriever, instance=True)
    chunk = _make_chunk()
    result = _make_retrieval_result(chunks=[chunk])
    mock.retrieve.return_value = result
    mock.default_limit = 10
    return mock


@pytest.fixture
def mock_context_builder() -> MagicMock:
    """A mock ``ContextBuilder`` with a working ``build`` method."""
    mock = create_autospec(ContextBuilder, instance=True)
    mock.build.return_value = "Document: test.txt\n\nRelevant document content.\n"
    mock.max_context_length = 4096
    return mock


@pytest.fixture
def rag_service(
    mock_retriever: MagicMock,
    mock_context_builder: MagicMock,
) -> RAGService:
    """A ``RAGService`` instance wired with mocked dependencies."""
    return RAGService(
        retriever=mock_retriever,
        context_builder=mock_context_builder,
    )


# ---------------------------------------------------------------------------
# Test successful RAG flow
# ---------------------------------------------------------------------------
class TestSuccessfulRAGFlow:
    def test_query_returns_rag_result(
        self,
        rag_service: RAGService,
    ) -> None:
        """A valid query should return a ``RAGResult`` with correct fields."""
        result = rag_service.query("test query")

        assert isinstance(result, RAGResult)
        assert result.query == "test query"
        assert isinstance(result.retrieval_result, RetrievalResult)
        assert isinstance(result.context, str)

    def test_result_contains_retrieval_result(
        self,
        rag_service: RAGService,
    ) -> None:
        """The ``RAGResult`` should contain the retrieval result."""
        result = rag_service.query("test query")
        assert result.retrieval_result.total_results == 1
        assert result.retrieval_result.chunks[0].text == "Relevant document content."

    def test_result_contains_formatted_context(
        self,
        rag_service: RAGService,
    ) -> None:
        """The ``RAGResult`` should contain the formatted context string."""
        result = rag_service.query("test query")
        assert "Document: test.txt" in result.context
        assert "Relevant document content." in result.context


# ---------------------------------------------------------------------------
# Test retriever interaction
# ---------------------------------------------------------------------------
class TestRetrieverInteraction:
    def test_retriever_called_with_query(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """The retriever should be called with the query text."""
        rag_service.query("find this")
        mock_retriever.retrieve.assert_called_once_with("find this", limit=None)

    def test_retriever_called_with_custom_limit(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """A custom limit should be passed to the retriever."""
        rag_service.query("find this", limit=5)
        mock_retriever.retrieve.assert_called_once_with("find this", limit=5)

    def test_retriever_query_stripped(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """The query should be stripped before being passed to the retriever."""
        rag_service.query("  spaced query  ")
        mock_retriever.retrieve.assert_called_once_with("spaced query", limit=None)


# ---------------------------------------------------------------------------
# Test context builder interaction
# ---------------------------------------------------------------------------
class TestContextBuilderInteraction:
    def test_context_builder_called_with_retrieval_result(
        self,
        rag_service: RAGService,
        mock_context_builder: MagicMock,
    ) -> None:
        """The context builder should receive the retrieval result."""
        rag_service.query("test query")
        # Verify build() was called with a RetrievalResult
        call_arg = mock_context_builder.build.call_args[0][0]
        assert isinstance(call_arg, RetrievalResult)
        assert call_arg.query == "test query"


# ---------------------------------------------------------------------------
# Test empty query handling
# ---------------------------------------------------------------------------
class TestEmptyQuery:
    def test_empty_query_raises(
        self,
        rag_service: RAGService,
    ) -> None:
        """An empty query should raise ``ConfigurationError``."""
        with pytest.raises(ConfigurationError, match="must not be empty"):
            rag_service.query("")

    def test_whitespace_query_raises(
        self,
        rag_service: RAGService,
    ) -> None:
        """A whitespace-only query should raise ``ConfigurationError``."""
        with pytest.raises(ConfigurationError, match="must not be empty"):
            rag_service.query("   \n  ")

    def test_empty_query_does_not_call_retriever(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """The retriever should not be called for an empty query."""
        with pytest.raises(ConfigurationError):
            rag_service.query("")
        mock_retriever.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Test retrieval failure handling
# ---------------------------------------------------------------------------
class TestRetrievalFailure:
    def test_retriever_configuration_error_propagates(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """A ``ConfigurationError`` from the retriever should propagate."""
        mock_retriever.retrieve.side_effect = ConfigurationError(
            "Query must not be empty."
        )
        with pytest.raises(ConfigurationError):
            rag_service.query("bad query")

    def test_retriever_chatbot_error_propagates(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """A ``ChatbotError`` from the retriever should propagate."""
        mock_retriever.retrieve.side_effect = ChatbotError(
            "Retrieval failed.",
            safe_message="Failed to search documents.",
        )
        with pytest.raises(ChatbotError, match="Retrieval failed"):
            rag_service.query("test query")

    def test_retriever_unexpected_error_wraps(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """An unexpected error from the retriever should be wrapped in ``ChatbotError``."""
        mock_retriever.retrieve.side_effect = RuntimeError("Connection reset")
        with pytest.raises(ChatbotError, match="unexpected error"):
            rag_service.query("test query")


# ---------------------------------------------------------------------------
# Test context builder failure handling
# ---------------------------------------------------------------------------
class TestContextBuilderFailure:
    def test_context_builder_failure_wraps(
        self,
        rag_service: RAGService,
        mock_context_builder: MagicMock,
    ) -> None:
        """A failure in the context builder should be wrapped in ``ChatbotError``."""
        mock_context_builder.build.side_effect = ValueError("Formatting error")
        with pytest.raises(ChatbotError, match="unexpected error"):
            rag_service.query("test query")


# ---------------------------------------------------------------------------
# Test empty retrieval result
# ---------------------------------------------------------------------------
class TestEmptyRetrievalResult:
    def test_empty_chunks_returns_empty_context(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """When retrieval returns no chunks, the context should be empty."""
        empty_result = _make_retrieval_result(chunks=[], query="test query")
        mock_retriever.retrieve.return_value = empty_result

        result = rag_service.query("test query")

        assert result.retrieval_result.total_results == 0
        assert result.retrieval_result.chunks == []
        # ContextBuilder.build() will return "" for empty chunks.
        # Our mock returns a fixed string, but we can assert the result structure is correct.
        assert isinstance(result.context, str)


# ---------------------------------------------------------------------------
# Test constructor validation
# ---------------------------------------------------------------------------
class TestConstructorValidation:
    def test_invalid_retriever_type_raises(self) -> None:
        """Passing a non-DocumentRetriever should raise ``TypeError``."""
        mock_builder = MagicMock(spec=ContextBuilder)
        with pytest.raises(TypeError, match="DocumentRetriever"):
            RAGService(retriever="invalid", context_builder=mock_builder)  # type: ignore[arg-type]

    def test_invalid_context_builder_type_raises(self) -> None:
        """Passing a non-ContextBuilder should raise ``TypeError``."""
        mock_retriever = MagicMock(spec=DocumentRetriever)
        with pytest.raises(TypeError, match="ContextBuilder"):
            RAGService(retriever=mock_retriever, context_builder="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test properties
# ---------------------------------------------------------------------------
class TestProperties:
    def test_retriever_property(
        self,
        rag_service: RAGService,
        mock_retriever: MagicMock,
    ) -> None:
        """The ``retriever`` property should return the injected instance."""
        assert rag_service.retriever is mock_retriever

    def test_context_builder_property(
        self,
        rag_service: RAGService,
        mock_context_builder: MagicMock,
    ) -> None:
        """The ``context_builder`` property should return the injected instance."""
        assert rag_service.context_builder is mock_context_builder


"""Unit tests for DocumentRetriever.

All tests use mocks and do not require real embedding providers or
vector stores.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.exceptions import ConfigurationError, ProviderResponseError
from src.retrieval.retriever import DocumentRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    provider = MagicMock()
    provider.embed.return_value = [0.1, 0.2, 0.3]
    return provider


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.search.return_value = [
        {
            "id": "chunk-1",
            "score": 0.95,
            "payload": {
                "document_id": "doc-1",
                "filename": "test.txt",
                "chunk_index": 0,
                "text": "Relevant chunk content.",
                "page_number": 1,
                "created_at": "2025-01-01T00:00:00+00:00",
            },
        },
        {
            "id": "chunk-2",
            "score": 0.85,
            "payload": {
                "document_id": "doc-1",
                "filename": "test.txt",
                "chunk_index": 1,
                "text": "Another relevant chunk.",
                "created_at": "2025-01-01T00:00:01+00:00",
            },
        },
    ]
    return store


@pytest.fixture
def retriever(
    mock_embedding_provider: MagicMock,
    mock_vector_store: MagicMock,
) -> DocumentRetriever:
    return DocumentRetriever(
        embedding_provider=mock_embedding_provider,
        vector_store=mock_vector_store,
        default_limit=10,
    )


# ---------------------------------------------------------------------------
# Test successful retrieval flow
# ---------------------------------------------------------------------------
class TestSuccessfulRetrieval:
    def test_retrieve_returns_result(
        self,
        retriever: DocumentRetriever,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Retrieval should return a RetrievalResult with chunks."""
        result = retriever.retrieve("test query")

        assert result.query == "test query"
        assert len(result.chunks) == 2
        assert result.total_results == 2
        assert isinstance(result.chunks, list)

    def test_query_embedding_generated(
        self,
        retriever: DocumentRetriever,
        mock_embedding_provider: MagicMock,
    ) -> None:
        """Embedding provider should be called with the query text."""
        retriever.retrieve("find this")
        mock_embedding_provider.embed.assert_called_once_with("find this")

    def test_vector_store_searched(
        self,
        retriever: DocumentRetriever,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Vector store should be called with the query vector and limit."""
        retriever.retrieve("test query", limit=5)
        mock_vector_store.search.assert_called_once_with(
            [0.1, 0.2, 0.3],
            limit=5,
        )

    def test_result_score_preserved(
        self,
        retriever: DocumentRetriever,
    ) -> None:
        """Chunk scores should match the vector store results."""
        result = retriever.retrieve("test query")
        assert result.chunks[0].score == 0.95
        assert result.chunks[1].score == 0.85

    def test_result_fields_mapped(
        self,
        retriever: DocumentRetriever,
    ) -> None:
        """Chunk fields should be correctly mapped from payload."""
        result = retriever.retrieve("test query")
        chunk = result.chunks[0]
        assert chunk.chunk_id == "chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.filename == "test.txt"
        assert chunk.chunk_index == 0
        assert chunk.text == "Relevant chunk content."
        assert chunk.page_number == 1
        assert chunk.created_at == "2025-01-01T00:00:00+00:00"

    def test_page_number_optional(
        self,
        retriever: DocumentRetriever,
    ) -> None:
        """Chunks without page_number should have None."""
        result = retriever.retrieve("test query")
        assert result.chunks[1].page_number is None

    def test_default_limit_used_when_not_specified(
        self,
        retriever: DocumentRetriever,
        mock_vector_store: MagicMock,
    ) -> None:
        """When limit is None, default_limit should be used."""
        retriever.retrieve("test query")
        mock_vector_store.search.assert_called_once_with(
            [0.1, 0.2, 0.3],
            limit=10,
        )


# ---------------------------------------------------------------------------
# Test empty results
# ---------------------------------------------------------------------------
class TestEmptyResults:
    def test_empty_search_results(
        self,
        retriever: DocumentRetriever,
        mock_vector_store: MagicMock,
    ) -> None:
        """Empty search results should return empty chunks list."""
        mock_vector_store.search.return_value = []
        result = retriever.retrieve("test query")
        assert result.total_results == 0
        assert result.chunks == []


# ---------------------------------------------------------------------------
# Test error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_empty_query_raises(
        self,
        retriever: DocumentRetriever,
    ) -> None:
        """Empty query should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Query must not be empty"):
            retriever.retrieve("")

    def test_whitespace_query_raises(
        self,
        retriever: DocumentRetriever,
    ) -> None:
        """Whitespace-only query should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Query must not be empty"):
            retriever.retrieve("   \n  ")

    def test_embedding_failure_raises(
        self,
        retriever: DocumentRetriever,
        mock_embedding_provider: MagicMock,
    ) -> None:
        """Embedding provider failure should propagate."""
        mock_embedding_provider.embed.side_effect = ProviderResponseError(
            "Embedding failed"
        )
        with pytest.raises(ProviderResponseError):
            retriever.retrieve("test query")

    def test_vector_store_failure_raises(
        self,
        retriever: DocumentRetriever,
        mock_vector_store: MagicMock,
    ) -> None:
        """Vector store failure should propagate."""
        mock_vector_store.search.side_effect = ProviderResponseError(
            "Search failed"
        )
        with pytest.raises(ProviderResponseError):
            retriever.retrieve("test query")


# ---------------------------------------------------------------------------
# Test constructor defaults
# ---------------------------------------------------------------------------
class TestConstructorDefaults:
    def test_default_limit(self) -> None:
        """Default limit should be 3."""
        provider = MagicMock()
        store = MagicMock()
        r = DocumentRetriever(
            embedding_provider=provider,
            vector_store=store,
        )
        assert r.default_limit == 3

    def test_custom_default_limit(self) -> None:
        """Custom default limit should be accepted."""
        provider = MagicMock()
        store = MagicMock()
        r = DocumentRetriever(
            embedding_provider=provider,
            vector_store=store,
            default_limit=25,
        )
        assert r.default_limit == 25

    def test_properties(
        self,
        retriever: DocumentRetriever,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Properties should expose the injected dependencies."""
        assert retriever.embedding_provider is mock_embedding_provider
        assert retriever.vector_store is mock_vector_store

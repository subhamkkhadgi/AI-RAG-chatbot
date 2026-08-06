"""Unit tests for DocumentRetriever.

All tests use mocks and do not require real embedding providers or
vector stores.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.exceptions import ConfigurationError, ProviderResponseError
from src.retrieval.models import DocumentScope
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
            filter_dict=None,
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
            filter_dict=None,
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


# ======================================================================
# Retrieval Precision Filters (Sprint 9A)
# ======================================================================

def _raw_result(
    doc_id: str,
    filename: str,
    chunk_index: int,
    score: float,
    text: str = "Relevant content.",
) -> dict:
    """Build a raw vector-store result dict for precision-filter tests."""
    return {
        "id": f"point-{doc_id}-{chunk_index}",
        "score": score,
        "payload": {
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": chunk_index,
            "text": text,
            "page_number": None,
            "created_at": "2025-01-01T00:00:00+00:00",
        },
    }


class TestScoreThreshold:
    """Similarity-threshold filtering of retrieved chunks."""

    def test_high_relevance_chunks_returned(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Chunks above the threshold are kept."""
        mock_vector_store.search.return_value = [
            _raw_result("doc-a", "a.pdf", 0, 0.9),
            _raw_result("doc-a", "a.pdf", 1, 0.8),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.5,
            max_documents=1,
        )
        result = retriever.retrieve("query")
        assert result.total_results == 2
        assert [c.score for c in result.chunks] == [0.9, 0.8]

    def test_low_relevance_chunks_filtered(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Chunks below the threshold are removed."""
        mock_vector_store.search.return_value = [
            _raw_result("doc-a", "a.pdf", 0, 0.9),
            _raw_result("doc-a", "a.pdf", 1, 0.2),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.5,
            max_documents=1,
        )
        result = retriever.retrieve("query")
        assert result.total_results == 1
        assert result.chunks[0].score == 0.9

    def test_threshold_boundary_inclusive(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """A chunk exactly at the threshold is kept (>=)."""
        mock_vector_store.search.return_value = [
            _raw_result("doc-a", "a.pdf", 0, 0.5),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.5,
            max_documents=1,
        )
        result = retriever.retrieve("query")
        assert result.total_results == 1

    def test_all_below_threshold_returns_empty(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """When no chunk passes the threshold, result is empty."""
        mock_vector_store.search.return_value = [
            _raw_result("doc-a", "a.pdf", 0, 0.2),
            _raw_result("doc-b", "b.pdf", 0, 0.3),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.9,
            max_documents=1,
        )
        result = retriever.retrieve("query")
        assert result.total_results == 0
        assert result.chunks == []


class TestDocumentCap:
    """Document-level filtering to prevent cross-document false positives."""

    def test_multiple_documents_only_relevant_returned(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Only chunks from the highest-relevance document are returned."""
        mock_vector_store.search.return_value = [
            _raw_result("cv", "Subham_khadgi_CV.pdf", 0, 0.9),
            _raw_result("proposal", "Web_Proposal.pdf", 0, 0.6),
            _raw_result("proposal", "Web_Proposal.pdf", 1, 0.55),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.0,
            max_documents=1,
        )
        result = retriever.retrieve("Who is Subham?")
        assert result.total_results == 1
        assert result.chunks[0].filename == "Subham_khadgi_CV.pdf"

    def test_unrelated_documents_excluded(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """The highest-scoring document wins; unrelated docs are excluded."""
        mock_vector_store.search.return_value = [
            _raw_result("cv", "Subham_khadgi_CV.pdf", 0, 0.7),
            _raw_result("proposal", "Web_Proposal.pdf", 0, 0.95),
            _raw_result("proposal", "Web_Proposal.pdf", 1, 0.9),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.0,
            max_documents=1,
        )
        result = retriever.retrieve("Web application")
        assert result.total_results == 2
        filenames = {c.filename for c in result.chunks}
        assert filenames == {"Web_Proposal.pdf"}

    def test_document_cap_keeps_top_n_documents(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """With max_documents=2, the top two documents are kept."""
        mock_vector_store.search.return_value = [
            _raw_result("doc-a", "a.pdf", 0, 0.9),
            _raw_result("doc-b", "b.pdf", 0, 0.8),
            _raw_result("doc-c", "c.pdf", 0, 0.7),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.0,
            max_documents=2,
        )
        result = retriever.retrieve("query")
        assert result.total_results == 2
        filenames = {c.filename for c in result.chunks}
        assert filenames == {"a.pdf", "b.pdf"}


class TestPrecisionProperties:
    """Expose the configured precision settings."""

    def test_min_score_property(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            min_score=0.6,
            max_documents=1,
        )
        assert retriever.min_score == 0.6
        assert retriever.max_documents == 1

    def test_defaults_from_settings(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Without explicit values, settings defaults are used."""
        from src.config import get_settings

        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )
        settings = get_settings()
        assert retriever.min_score == settings.retrieval_min_score
        assert retriever.max_documents == settings.retrieval_max_documents


# ======================================================================
# Document Scope (Sprint 9C)
# ======================================================================
class TestDocumentScope:
    """Scoped retrieval restricted to specific documents."""

    def test_scope_passes_filter_dict(
        self,
        retriever: DocumentRetriever,
        mock_vector_store: MagicMock,
    ) -> None:
        """A single-document scope should be passed as a filter dict."""
        scope = DocumentScope(document_ids=["doc-1"])
        retriever.retrieve("test query", scope=scope)
        mock_vector_store.search.assert_called_once_with(
            [0.1, 0.2, 0.3],
            limit=10,
            filter_dict={"document_id": "doc-1"},
        )

    def test_none_scope_no_filter(self, retriever: DocumentRetriever) -> None:
        """No scope means unrestricted search (filter_dict=None)."""
        req = [0.1, 0.2, 0.3]
        retriever.retrieve("test query")
        retriever._vector_store.search.assert_called_once_with(
            req, limit=10, filter_dict=None
        )

    def test_scope_guard_filters_outside_documents(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Chunks from unselected documents are dropped post-search."""
        mock_vector_store.search.return_value = [
            _raw_result("doc-1", "a.pdf", 0, 0.9),
            _raw_result("doc-2", "b.pdf", 0, 0.8),
        ]
        retriever = DocumentRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            default_limit=10,
            min_score=0.0,
            max_documents=0,
        )
        scope = DocumentScope(document_ids=["doc-1"])
        result = retriever.retrieve("query", scope=scope)
        assert result.total_results == 1
        assert result.chunks[0].document_id == "doc-1"

    def test_empty_scope_unrestricted(
        self,
        retriever: DocumentRetriever,
        mock_vector_store: MagicMock,
    ) -> None:
        """An empty scope behaves like no scope (no filter, no guard)."""
        scope = DocumentScope()
        result = retriever.retrieve("test query", scope=scope)
        assert result.total_results == 2
        mock_vector_store.search.assert_called_once_with(
            [0.1, 0.2, 0.3],
            limit=10,
            filter_dict=None,
        )

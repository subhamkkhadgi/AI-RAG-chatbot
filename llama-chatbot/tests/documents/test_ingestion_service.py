"""Unit tests for DocumentIngestionService.

All tests use mocks and do not require real embedding providers or
vector stores.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.documents.ingestion_service import DocumentIngestionService
from src.exceptions import ConfigurationError, ProviderResponseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    provider = MagicMock()
    provider.embed_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    return provider


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.create_collection.return_value = None
    store.upsert.return_value = None
    return store


@pytest.fixture
def service(
    mock_embedding_provider: MagicMock,
    mock_vector_store: MagicMock,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        embedding_provider=mock_embedding_provider,
        vector_store=mock_vector_store,
        chunk_size=512,
        overlap=64,
        collection_name="test_collection",
    )


# ---------------------------------------------------------------------------
# Test successful ingestion flow
# ---------------------------------------------------------------------------
class TestSuccessfulIngestion:
    def test_ingest_txt_file(
        self,
        service: DocumentIngestionService,
        mock_embedding_provider: MagicMock,
        mock_vector_store: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Ingesting a .txt file should return a document_id and call providers."""
        file_path = tmp_path / "test_doc.txt"
        file_path.write_text("Hello, world!\n\nThis is a test document.", encoding="utf-8")

        doc_id = service.ingest_file(str(file_path))

        assert doc_id, "Expected a non-empty document_id"

        mock_embedding_provider.embed_batch.assert_called_once()
        call_args = mock_embedding_provider.embed_batch.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) > 0
        assert all(isinstance(t, str) for t in call_args)

        mock_vector_store.upsert.assert_called_once()
        upsert_args = mock_vector_store.upsert.call_args[0]
        vectors, payloads = upsert_args
        assert len(vectors) == len(payloads)

    def test_ingest_pdf_file(
        self,
        service: DocumentIngestionService,
        tmp_path: Path,
    ) -> None:
        """Ingesting a .pdf file that is not a real PDF should raise."""
        file_path = tmp_path / "test_doc.pdf"
        file_path.write_text("Not a real PDF.", encoding="utf-8")

        with pytest.raises(Exception):
            service.ingest_file(str(file_path))

    def test_metadata_payload(
        self,
        service: DocumentIngestionService,
        mock_vector_store: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Payloads should contain all required metadata fields."""
        file_path = tmp_path / "meta_test.txt"
        file_path.write_text(
            "Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            encoding="utf-8",
        )

        service.ingest_file(str(file_path))

        mock_vector_store.upsert.assert_called_once()
        _, payloads = mock_vector_store.upsert.call_args[0]

        for payload in payloads:
            assert "document_id" in payload
            assert "filename" in payload
            assert "chunk_index" in payload
            assert "chunk_count" in payload
            assert "text" in payload
            assert "created_at" in payload
            assert payload["filename"] == "meta_test.txt"
            assert isinstance(payload["chunk_count"], int)
            assert payload["chunk_count"] == len(payloads)


# ---------------------------------------------------------------------------
# Test error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_file_not_found_raises(self, service: DocumentIngestionService) -> None:
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            service.ingest_file("/nonexistent/path/file.txt")

    def test_unsupported_extension_raises(
        self,
        service: DocumentIngestionService,
        tmp_path: Path,
    ) -> None:
        """Unsupported file extension should raise ConfigurationError."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("some content", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="Unsupported file type"):
            service.ingest_file(str(file_path))

    def test_extraction_failure_for_invalid_pdf(
        self,
        service: DocumentIngestionService,
        tmp_path: Path,
    ) -> None:
        """A file with .pdf extension but not a real PDF should raise."""
        file_path = tmp_path / "fake.pdf"
        file_path.write_text("Not a PDF", encoding="utf-8")

        with pytest.raises(ConfigurationError):
            service.ingest_file(str(file_path))

    @patch("src.documents.extractors.extract_document")
    def test_embedding_failure_raises(
        self,
        mock_extract: MagicMock,
        service: DocumentIngestionService,
        tmp_path: Path,
    ) -> None:
        """Embedding failure should propagate."""
        mock_extract.return_value = "Some document text."
        file_path = tmp_path / "test.txt"
        file_path.write_text("dummy", encoding="utf-8")

        service._embedding_provider.embed_batch.side_effect = ProviderResponseError(
            "Embedding failed"
        )

        with pytest.raises(ProviderResponseError):
            service.ingest_file(str(file_path))

    @patch("src.documents.extractors.extract_document")
    def test_vector_store_failure_raises(
        self,
        mock_extract: MagicMock,
        service: DocumentIngestionService,
        mock_vector_store: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Vector store failure should propagate."""
        mock_extract.return_value = "Some document text."
        mock_vector_store.upsert.side_effect = ProviderResponseError("Store failed")

        file_path = tmp_path / "test.txt"
        file_path.write_text("dummy", encoding="utf-8")

        with pytest.raises(ProviderResponseError):
            service.ingest_file(str(file_path))


# ---------------------------------------------------------------------------
# Test constructor defaults
# ---------------------------------------------------------------------------
class TestConstructorDefaults:
    def test_default_parameters(self) -> None:
        """Default parameters should be set correctly."""
        provider = MagicMock()
        store = MagicMock()
        svc = DocumentIngestionService(
            embedding_provider=provider,
            vector_store=store,
        )
        assert svc.chunk_size == 1024
        assert svc.overlap == 128
        assert svc.collection_name == "documents"
        assert svc.embedding_provider is provider
        assert svc.vector_store is store

    def test_custom_parameters(self) -> None:
        """Custom parameters should be accepted."""
        provider = MagicMock()
        store = MagicMock()
        svc = DocumentIngestionService(
            embedding_provider=provider,
            vector_store=store,
            chunk_size=256,
            overlap=32,
            collection_name="custom_collection",
        )
        assert svc.chunk_size == 256
        assert svc.overlap == 32
        assert svc.collection_name == "custom_collection"

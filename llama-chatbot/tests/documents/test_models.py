"""Unit tests for Document and DocumentChunk models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.documents.models import Document, DocumentChunk


class TestDocument:
    def test_create_document_with_valid_fields(self) -> None:
        """Document should be created with the given fields."""
        doc = Document(filename="report.pdf", content="Hello world")
        assert doc.filename == "report.pdf"
        assert doc.content == "Hello world"
        assert len(doc.document_id) == 32

    def test_document_is_frozen(self) -> None:
        """Document should be immutable after creation."""
        doc = Document(filename="test.txt", content="content")
        with pytest.raises(ValidationError):
            doc.filename = "new.txt"

    def test_empty_filename_raises(self) -> None:
        """Empty filename should raise ValidationError."""
        with pytest.raises(ValidationError):
            Document(filename="", content="content")

    def test_whitespace_filename_raises(self) -> None:
        """Whitespace-only filename should raise ValidationError."""
        with pytest.raises(ValidationError):
            Document(filename="   ", content="content")


class TestDocumentChunk:
    def test_create_chunk_with_valid_fields(self) -> None:
        """DocumentChunk should be created with the given fields."""
        chunk = DocumentChunk(
            document_id="abc123",
            filename="report.pdf",
            chunk_index=0,
            text="Some chunked text",
        )
        assert chunk.document_id == "abc123"
        assert chunk.filename == "report.pdf"
        assert chunk.chunk_index == 0
        assert chunk.text == "Some chunked text"
        assert chunk.page_number is None
        assert isinstance(chunk.created_at, datetime)

    def test_chunk_with_page_number(self) -> None:
        """Page number should be stored when provided."""
        chunk = DocumentChunk(
            document_id="abc",
            filename="report.pdf",
            chunk_index=1,
            text="Page 2 content",
            page_number=2,
        )
        assert chunk.page_number == 2

    def test_chunk_is_frozen(self) -> None:
        """DocumentChunk should be immutable after creation."""
        chunk = DocumentChunk(
            document_id="abc",
            filename="f.txt",
            chunk_index=0,
            text="hello",
        )
        with pytest.raises(ValidationError):
            chunk.text = "modified"

    def test_empty_text_raises(self) -> None:
        """Empty chunk text should raise ValidationError."""
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="abc",
                filename="f.txt",
                chunk_index=0,
                text="",
            )

    def test_whitespace_text_raises(self) -> None:
        """Whitespace-only text should raise ValidationError."""
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="abc",
                filename="f.txt",
                chunk_index=0,
                text="   \n  ",
            )

    def test_empty_document_id_raises(self) -> None:
        """Empty document_id should raise ValidationError."""
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="",
                filename="f.txt",
                chunk_index=0,
                text="content",
            )

    def test_negative_chunk_index_raises(self) -> None:
        """Negative chunk_index should raise ValidationError."""
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="abc",
                filename="f.txt",
                chunk_index=-1,
                text="content",
            )

    def test_invalid_page_number_zero_raises(self) -> None:
        """Page number 0 should raise ValidationError (must be >= 1)."""
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="abc",
                filename="f.txt",
                chunk_index=0,
                text="content",
                page_number=0,
            )


class TestDocumentChunkTimestamp:
    def test_created_at_is_utc(self) -> None:
        """created_at should have UTC timezone info."""
        chunk = DocumentChunk(
            document_id="abc",
            filename="f.txt",
            chunk_index=0,
            text="hello",
        )
        assert chunk.created_at.tzinfo is not None
        assert chunk.created_at.tzinfo is UTC

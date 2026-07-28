"""Unit tests for DocumentChunker."""

from __future__ import annotations

import pytest

from src.documents.chunking import DocumentChunker
from src.documents.models import Document


class TestDocumentChunkerInit:
    def test_default_parameters(self) -> None:
        """Default chunk_size and overlap should be set."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1024
        assert chunker.overlap == 128

    def test_custom_parameters(self) -> None:
        """Custom chunk_size and overlap should be accepted."""
        chunker = DocumentChunker(chunk_size=512, overlap=64)
        assert chunker.chunk_size == 512
        assert chunker.overlap == 64

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="overlap must be a non-negative"):
            DocumentChunker(chunk_size=100, overlap=-1)

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be a positive"):
            DocumentChunker(chunk_size=0, overlap=0)

    def test_overlap_gte_chunk_size_raises(self) -> None:
        """overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than"):
            DocumentChunker(chunk_size=100, overlap=100)


class TestDocumentChunkerChunking:
    def test_single_chunk_small_document(self) -> None:
        """A small document should produce a single chunk."""
        doc = Document(filename="test.txt", content="Hello, world!")

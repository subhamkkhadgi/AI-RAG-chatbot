"""Unit tests for document extractors."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.documents.extractors import (
    PDFExtractor,
    TXTExtractor,
    extract_document,
)
from src.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# TXT extractor tests
# ---------------------------------------------------------------------------
class TestTXTExtractor:
    def test_extract_txt_file(self, tmp_path: Path) -> None:
        """TXTExtractor should extract text from a .txt file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, world!", encoding="utf-8")
        extractor = TXTExtractor()
        result = extractor.extract(str(file_path))
        assert result == "Hello, world!"

    def test_extract_txt_file_strips_whitespace(self, tmp_path: Path) -> None:
        """Extracted text should be stripped of leading/trailing whitespace."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("  \n  Hello  \n  ", encoding="utf-8")
        extractor = TXTExtractor()
        result = extractor.extract(str(file_path))
        assert result == "Hello"

    def test_extract_non_existent_file_raises(self) -> None:
        """Missing file should raise FileNotFoundError."""
        extractor = TXTExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/file.txt")

    def test_extract_empty_file_raises(self, tmp_path: Path) -> None:
        """Empty file should raise ConfigurationError."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("   \n  ", encoding="utf-8")
        extractor = TXTExtractor()
        with pytest.raises(ConfigurationError, match="empty"):
            extractor.extract(str(file_path))

    def test_supported_extensions(self) -> None:
        """TXTExtractor should only support .txt."""
        extractor = TXTExtractor()
        assert extractor.supported_extensions == frozenset({".txt"})


# ---------------------------------------------------------------------------
# PDF extractor tests
# ---------------------------------------------------------------------------
class TestPDFExtractor:
    def test_extract_non_existent_file_raises(self) -> None:
        """Missing file should raise FileNotFoundError."""
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/file.pdf")

    def test_supported_extensions(self) -> None:
        """PDFExtractor should only support .pdf."""
        extractor = PDFExtractor()
        assert extractor.supported_extensions == frozenset({".pdf"})


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------
class TestExtractDocument:
    def test_unsupported_file_type_raises(self) -> None:
        """Unsupported file extension should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unsupported file type"):
            extract_document("file.xyz")

    def test_supported_extensions_listed_in_error(self) -> None:
        """Error message should list supported extensions."""
        with pytest.raises(ConfigurationError) as exc_info:
            extract_document("file.xyz")
        assert ".pdf" in str(exc_info.value)
        assert ".txt" in str(exc_info.value)

    def test_txt_extraction_via_factory(self, tmp_path: Path) -> None:
        """extract_document should route .txt files to TXTExtractor."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Factory test", encoding="utf-8")
        result = extract_document(str(file_path))
        assert result == "Factory test"

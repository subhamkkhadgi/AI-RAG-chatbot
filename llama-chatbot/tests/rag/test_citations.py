"""Unit tests for the source citation formatter.

All tests construct ``RetrievedChunk`` models directly — no mocks of
external providers are needed.
"""

from __future__ import annotations

import pytest

from src.rag.citations import build_citations_section
from src.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(
    filename: str = "test.txt",
    text: str = "Some content.",
    page_number: int | None = None,
    chunk_index: int = 0,
    document_id: str = "doc-1",
    score: float = 0.95,
) -> RetrievedChunk:
    """Build a ``RetrievedChunk`` with minimal required fields."""
    return RetrievedChunk(
        chunk_id=f"chunk-{chunk_index}",
        document_id=document_id,
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
        page_number=page_number,
    )


# ---------------------------------------------------------------------------
# Basic formatting
# ---------------------------------------------------------------------------
class TestCitationFormatting:
    def test_single_citation_without_page(self) -> None:
        """A single chunk without a page number yields a plain bullet."""
        chunks = [_make_chunk(filename="notes.txt", page_number=None)]
        output = build_citations_section(chunks)

        assert output == "\n\nSources:\n- notes.txt"

    def test_single_citation_with_page(self) -> None:
        """A single chunk with a page number includes the page."""
        chunks = [_make_chunk(filename="report.pdf", page_number=5)]
        output = build_citations_section(chunks)

        assert output == "\n\nSources:\n- report.pdf (Page 5)"

    def test_multiple_sources_in_order(self) -> None:
        """Multiple distinct sources preserve first-seen order."""
        chunks = [
            _make_chunk(filename="a.pdf", page_number=2, chunk_index=0),
            _make_chunk(filename="b.txt", page_number=None, chunk_index=1),
        ]
        output = build_citations_section(chunks)

        assert output == "\n\nSources:\n- a.pdf (Page 2)\n- b.txt"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class TestCitationDeduplication:
    def test_duplicate_filename_and_page_deduplicated(self) -> None:
        """Chunks from the same document and page appear only once."""
        chunks = [
            _make_chunk(filename="report.pdf", page_number=2, chunk_index=0),
            _make_chunk(filename="report.pdf", page_number=2, chunk_index=1),
            _make_chunk(filename="report.pdf", page_number=2, chunk_index=2),
        ]
        output = build_citations_section(chunks)

        assert output.count("- report.pdf (Page 2)") == 1

    def test_same_filename_different_pages_both_kept(self) -> None:
        """Different pages of the same document are distinct citations."""
        chunks = [
            _make_chunk(filename="report.pdf", page_number=2, chunk_index=0),
            _make_chunk(filename="report.pdf", page_number=3, chunk_index=1),
        ]
        output = build_citations_section(chunks)

        assert "- report.pdf (Page 2)" in output
        assert "- report.pdf (Page 3)" in output

    def test_duplicate_without_page_deduplicated(self) -> None:
        """Multiple no-page chunks from the same file appear once."""
        chunks = [
            _make_chunk(filename="notes.txt", page_number=None, chunk_index=0),
            _make_chunk(filename="notes.txt", page_number=None, chunk_index=1),
        ]
        output = build_citations_section(chunks)

        assert output == "\n\nSources:\n- notes.txt"


# ---------------------------------------------------------------------------
# Page number handling
# ---------------------------------------------------------------------------
class TestPageNumberHandling:
    def test_page_zero_rendered(self) -> None:
        """Even a page number of 0 should be rendered literally."""
        chunks = [_make_chunk(filename="odd.pdf", page_number=0)]
        output = build_citations_section(chunks)

        assert "- odd.pdf (Page 0)" in output

    def test_mixed_page_and_no_page_are_distinct(self) -> None:
        """Same file with and without a page are separate citations."""
        chunks = [
            _make_chunk(filename="notes.txt", page_number=None, chunk_index=0),
            _make_chunk(filename="notes.txt", page_number=1, chunk_index=1),
        ]
        output = build_citations_section(chunks)

        assert "- notes.txt" in output
        assert "- notes.txt (Page 1)" in output


# ---------------------------------------------------------------------------
# Empty / invalid inputs
# ---------------------------------------------------------------------------
class TestEmptyInputs:
    def test_none_returns_empty(self) -> None:
        """``None`` should return an empty string."""
        assert build_citations_section(None) == ""

    def test_empty_list_returns_empty(self) -> None:
        """An empty list should return an empty string."""
        assert build_citations_section([]) == ""

    def test_no_empty_sources_header(self) -> None:
        """No chunks should never produce a bare ``Sources:`` header."""
        assert "Sources:" not in build_citations_section([])
        assert "Sources:" not in build_citations_section(None)

    def test_missing_filename_skipped(self) -> None:
        """Chunks without a usable filename should be skipped."""
        # Simulate a chunk-like object missing filename.
        class _ChunkLike:
            filename = ""

        output = build_citations_section([_ChunkLike()])  # type: ignore[list-item]
        assert output == ""


# ---------------------------------------------------------------------------
# Sanity: no internal metadata leaked
# ---------------------------------------------------------------------------
class TestNoInternalLeak:
    def test_internal_fields_not_emitted(self) -> None:
        """document_id / chunk_index / score must never appear in output."""
        chunks = [
            _make_chunk(
                filename="report.pdf",
                page_number=3,
                chunk_index=4,
                document_id="internal-doc-id",
                score=0.987,
            )
        ]
        output = build_citations_section(chunks)

        assert "internal-doc-id" not in output
        assert "chunk_index" not in output
        assert "0.987" not in output
        assert "score" not in output


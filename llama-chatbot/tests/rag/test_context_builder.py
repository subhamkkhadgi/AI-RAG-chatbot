"""Unit tests for ContextBuilder.

All tests construct ``RetrievalResult`` and ``RetrievedChunk`` directly
— no mocks of external providers are needed.
"""

from __future__ import annotations

import pytest

from src.rag.context_builder import ContextBuilder
from src.retrieval.models import RetrievedChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(
    filename: str = "test.txt",
    text: str = "Some relevant content.",
    page_number: int | None = None,
    chunk_index: int = 0,
    score: float = 0.95,
) -> RetrievedChunk:
    """Build a ``RetrievedChunk`` with minimal required fields."""
    return RetrievedChunk(
        chunk_id=f"chunk-{chunk_index}",
        document_id="doc-1",
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
        page_number=page_number,
    )


def _make_result(
    chunks: list[RetrievedChunk] | None = None, query: str = "test query"
) -> RetrievalResult:
    """Build a ``RetrievalResult`` from the provided chunks."""
    resolved = chunks or []
    return RetrievalResult(
        query=query,
        chunks=resolved,
        total_results=len(resolved),
    )


# ---------------------------------------------------------------------------
# Test formatting
# ---------------------------------------------------------------------------
class TestFormatting:
    def test_single_chunk_formatting(self) -> None:
        """A single chunk should produce the expected output structure."""
        chunk = _make_chunk()
        result = _make_result([chunk])
        builder = ContextBuilder(max_context_length=1000)

        output = builder.build(result)

        expected = (
            "Document: test.txt\n"
            "\n"
            "Some relevant content.\n"
        )
        assert output == expected

    def test_multiple_documents(self) -> None:
        """Multiple chunks from different documents should be separated."""
        chunk_1 = _make_chunk(filename="a.pdf", text="Content A.", chunk_index=0)
        chunk_2 = _make_chunk(filename="b.txt", text="Content B.", chunk_index=1)
        result = _make_result([chunk_1, chunk_2])
        builder = ContextBuilder(max_context_length=1000)

        output = builder.build(result)

        assert "Document: a.pdf" in output
        assert "Document: b.txt" in output
        assert "Content A." in output
        assert "Content B." in output
        assert "----------------------------------------" in output

    def test_page_number_inclusion(self) -> None:
        """When a chunk has a page number, it should be rendered."""
        chunk = _make_chunk(filename="handbook.pdf", text="Page 3 text.", page_number=3)
        result = _make_result([chunk])
        builder = ContextBuilder(max_context_length=1000)

        output = builder.build(result)

        assert "Document: handbook.pdf" in output
        assert "Page: 3" in output
        assert "Page 3 text." in output

    def test_page_number_omitted_when_none(self) -> None:
        """When a chunk has no page number, the Page line should be absent."""
        chunk = _make_chunk(filename="notes.txt", text="No page.", page_number=None)
        result = _make_result([chunk])
        builder = ContextBuilder(max_context_length=1000)

        output = builder.build(result)

        assert "Document: notes.txt" in output
        assert "Page:" not in output
        assert "No page." in output

    def test_ordering_preserved(self) -> None:
        """Chunks should appear in the same order as in the result."""
        chunk_1 = _make_chunk(
            filename="first.txt", text="First chunk.", chunk_index=0, score=0.9
        )
        chunk_2 = _make_chunk(
            filename="second.txt", text="Second chunk.", chunk_index=1, score=0.8
        )
        chunk_3 = _make_chunk(
            filename="third.txt", text="Third chunk.", chunk_index=2, score=0.7
        )
        result = _make_result([chunk_1, chunk_2, chunk_3])
        builder = ContextBuilder(max_context_length=5000)

        output = builder.build(result)

        pos_first = output.index("First chunk.")
        pos_second = output.index("Second chunk.")
        pos_third = output.index("Third chunk.")
        assert pos_first < pos_second < pos_third


# ---------------------------------------------------------------------------
# Test empty result
# ---------------------------------------------------------------------------
class TestEmptyResult:
    def test_empty_retrieval_result(self) -> None:
        """An empty retrieval result should produce an empty string."""
        result = _make_result([])
        builder = ContextBuilder()
        assert builder.build(result) == ""

    def test_no_chunks_returns_empty_string(self) -> None:
        """Explicitly empty chunks list returns empty string."""
        result = _make_result([])
        builder = ContextBuilder(max_context_length=100)
        assert builder.build(result) == ""


# ---------------------------------------------------------------------------
# Test maximum context length
# ---------------------------------------------------------------------------
class TestMaxContextLength:
    def test_max_context_length_enforced(self) -> None:
        """Output should not exceed the configured maximum."""
        chunk = _make_chunk(
            filename="long.txt",
            text="A" * 500,
        )
        result = _make_result([chunk])
        builder = ContextBuilder(max_context_length=50)

        output = builder.build(result)

        assert len(output) <= 50

    def test_truncation_safe_boundary(self) -> None:
        """When total exceeds limit, the last chunk text should be trimmed."""
        chunk_1 = _make_chunk(
            filename="short.txt",
            text="Short.",
            chunk_index=0,
        )
        chunk_2 = _make_chunk(
            filename="medium.txt",
            text="B" * 200,
            chunk_index=1,
        )
        result = _make_result([chunk_1, chunk_2])
        # The first chunk takes ~40 chars.  The second chunk will be
        # truncated so total fits within the limit.
        builder = ContextBuilder(max_context_length=80)

        output = builder.build(result)

        assert len(output) <= 80
        assert "Short." in output
        # The second chunk's text should be truncated (not the full "B"*200).
        assert "B" * 200 not in output

    def test_very_small_limit_returns_empty(self) -> None:
        """If even the header cannot fit, empty string (or partial) returned."""
        chunk = _make_chunk(
            filename="tiny.txt",
            text="Hi",
        )
        result = _make_result([chunk])
        builder = ContextBuilder(max_context_length=5)

        output = builder.build(result)

        # The header "Document: tiny.txt" is > 5 chars, so nothing fits.
        assert output == "" or len(output) <= 5

    def test_multiple_chunks_truncated(self) -> None:
        """Multiple chunks should be truncated when over limit."""
        chunk_1 = _make_chunk(
            filename="doc1.txt",
            text="Hello world.",
            chunk_index=0,
        )
        chunk_2 = _make_chunk(
            filename="doc2.txt",
            text="Some longer text content here.",
            chunk_index=1,
        )
        chunk_3 = _make_chunk(
            filename="doc3.txt",
            text="Even more text that should not appear.",
            chunk_index=2,
        )
        result = _make_result([chunk_1, chunk_2, chunk_3])
        # Limit is large enough for the first two chunks but not the third.
        builder = ContextBuilder(max_context_length=120)

        output = builder.build(result)

        assert len(output) <= 120
        assert "Hello world." in output
        assert "doc3.txt" not in output


# ---------------------------------------------------------------------------
# Test default configuration
# ---------------------------------------------------------------------------
class TestDefaults:
    def test_default_max_context_length(self) -> None:
        """The default max_context_length should be 4096."""
        builder = ContextBuilder()
        assert builder.max_context_length == 4096

    def test_custom_max_context_length(self) -> None:
        """A custom max_context_length should be accepted."""
        builder = ContextBuilder(max_context_length=2048)
        assert builder.max_context_length == 2048

    def test_repr(self) -> None:
        """repr should include the configuration."""
        builder = ContextBuilder(max_context_length=1024)
        r = repr(builder)
        assert "ContextBuilder" in r
        assert "1024" in r


# ---------------------------------------------------------------------------
# Test constructor validation
# ---------------------------------------------------------------------------
class TestConstructorValidation:
    def test_non_positive_length_raises(self) -> None:
        """A non-positive max_context_length should raise ValueError."""
        with pytest.raises(ValueError, match="max_context_length must be positive"):
            ContextBuilder(max_context_length=0)

    def test_negative_length_raises(self) -> None:
        """A negative max_context_length should raise ValueError."""
        with pytest.raises(ValueError, match="max_context_length must be positive"):
            ContextBuilder(max_context_length=-1)


"""Unit tests for answer-aware citation support filtering.

All tests construct ``RetrievedChunk`` models directly — no mocks of
external providers or LLM calls are needed.
"""

from __future__ import annotations

import pytest

from src.rag.evidence import filter_supporting_chunks
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
# Supporting chunk retained
# ---------------------------------------------------------------------------
class TestSupportingChunkRetained:
    def test_retains_chunk_whose_text_overlaps_answer(self) -> None:
        """A chunk whose text appears in the answer is retained."""
        chunk = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text="The project is named Atlas.",
        )
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [chunk],
        )
        assert supporting == [chunk]

    def test_retains_order_of_supporting_chunks(self) -> None:
        """Supporting chunks preserve their original retrieval order."""
        first = _make_chunk(
            filename="a.pdf", page_number=1, text="Atlas is the project name."
        )
        second = _make_chunk(
            filename="b.pdf", page_number=2, text="The project is named Atlas."
        )
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [first, second],
        )
        assert supporting == [first, second]

    def test_case_insensitive_matching(self) -> None:
        """Token matching is case-insensitive."""
        chunk = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text="The project is named ATLAS.",
        )
        supporting = filter_supporting_chunks(
            "The project is named atlas.",
            [chunk],
        )
        assert supporting == [chunk]


# ---------------------------------------------------------------------------
# Unsupported chunk removed
# ---------------------------------------------------------------------------
class TestUnsupportedChunkRemoved:
    def test_removes_chunk_with_no_overlap(self) -> None:
        """A chunk sharing no tokens with the answer is removed."""
        supporting_chunk = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text="The project is named Atlas.",
        )
        unsupported_chunk = _make_chunk(
            filename="report.pdf",
            page_number=9,
            text="The budget is allocated to the team.",
        )
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [supporting_chunk, unsupported_chunk],
        )
        assert supporting == [supporting_chunk]

    def test_removes_all_unsupported_chunks(self) -> None:
        """When no chunk supports the answer, the result is empty."""
        chunks = [
            _make_chunk(
                filename="a.pdf", page_number=1, text="Budget details are here."
            ),
            _make_chunk(
                filename="b.pdf", page_number=2, text="The team is growing."
            ),
        ]
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            chunks,
        )
        assert supporting == []

    def test_removes_chunk_below_overlap_threshold(self) -> None:
        """A chunk with minimal overlap below the threshold is removed."""
        chunk = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text="Budget budget budget budget budget detailed.",
        )
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [chunk],
        )
        assert supporting == []


# ---------------------------------------------------------------------------
# No supporting chunks -> no citations
# ---------------------------------------------------------------------------
class TestNoSupportingCitations:
    def test_empty_answer_returns_no_chunks(self) -> None:
        """With an empty answer, no chunk can be shown as supporting."""
        chunks = [
            _make_chunk(
                filename="a.pdf", page_number=1, text="The project is named Atlas."
            )
        ]
        supporting = filter_supporting_chunks("", chunks)
        assert supporting == []

    def test_whitespace_answer_returns_no_chunks(self) -> None:
        """A whitespace-only answer yields no supporting chunks."""
        chunks = [
            _make_chunk(
                filename="a.pdf", page_number=1, text="The project is named Atlas."
            )
        ]
        supporting = filter_supporting_chunks("   ", chunks)
        assert supporting == []

    def test_none_chunks_returns_empty(self) -> None:
        """``None`` chunks produce an empty result."""
        assert filter_supporting_chunks("answer", None) == []

    def test_empty_chunks_returns_empty(self) -> None:
        """An empty chunk list produces an empty result."""
        assert filter_supporting_chunks("answer", []) == []

    def test_non_list_chunks_returns_empty(self) -> None:
        """Non-list/tuple input (e.g. MagicMock) is tolerated."""
        assert filter_supporting_chunks("answer", object()) == []

    def test_no_answer_tokens_returns_empty(self) -> None:
        """An answer with no alphanumeric tokens yields no supports."""
        chunks = [
            _make_chunk(
                filename="a.pdf", page_number=1, text="The project is named Atlas."
            )
        ]
        supporting = filter_supporting_chunks("!!!", chunks)
        assert supporting == []


# ---------------------------------------------------------------------------
# Answer coverage (long supporting chunk retained)
# ---------------------------------------------------------------------------
class TestAnswerCoverage:
    """Scoring measures how much of the answer appears in the chunk
    (answer coverage), so a long page-level chunk containing the answer's
    key tokens is retained even though most of its tokens are unrelated."""

    def test_long_supporting_chunk_retained(self) -> None:
        """A verbose page-1 chunk containing the answer's key tokens is
        retained (answer coverage, not chunk coverage)."""
        chunk = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text=(
                "Web Application Project Proposal — title page. Prepared for "
                "ACME Corp by the consulting team, dated 2024. This document "
                "outlines the plan, objectives, scope, and deliverables. The "
                "project is named Atlas and the executive summary follows."
            ),
        )
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [chunk],
        )
        assert supporting == [chunk]

    def test_unsupported_page_still_removed(self) -> None:
        """A long chunk that shares none of the answer's key tokens is
        still removed."""
        supported = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text="The project is named Atlas. This is the title page.",
        )
        unsupported = _make_chunk(
            filename="report.pdf",
            page_number=9,
            text=(
                "The budget is allocated to the team and the timeline is "
                "reviewed monthly by the steering committee."
            ),
            chunk_index=1,
        )
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [supported, unsupported],
        )
        assert supporting == [supported]

    def test_answer_mostly_covered_by_chunk(self) -> None:
        """A chunk containing most of the answer's tokens is retained."""
        chunk = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text="Budget details mention the project named Atlas in the summary.",
        )
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [chunk],
        )
        assert supporting == [chunk]


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------
class TestThreshold:
    def test_custom_min_overlap(self) -> None:
        """A stricter threshold filters out weak matches."""
        chunk = _make_chunk(
            filename="report.pdf",
            page_number=1,
            text="Atlas project Atlas project Atlas project details.",
        )
        # Default threshold (0.3) would retain this; a high threshold removes it.
        supporting = filter_supporting_chunks(
            "The project is named Atlas.",
            [chunk],
            min_overlap=0.9,
        )
        assert supporting == []

    def test_invalid_min_overlap_raises(self) -> None:
        """min_overlap outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            filter_supporting_chunks("answer", [], min_overlap=1.5)
        with pytest.raises(ValueError):
            filter_supporting_chunks("answer", [], min_overlap=-0.1)

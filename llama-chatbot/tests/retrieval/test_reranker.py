"""Unit tests for cross-encoder reranking + page-diverse selection.

All tests are unit-level and require no real vector store, embedding
provider, or model download.  The heavy ``sentence_transformers``
dependency is never required because the MiniLM model is loaded lazily
and is faked/mocked in these tests.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import ChatbotError, ConfigurationError
from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import (
    DEFAULT_TOP_N,
    MiniLMReranker,
    select_page_diverse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _chunk(
    chunk_id: str,
    filename: str,
    page_number: int | None,
    text: str = "Some chunk text.",
    score: float = 0.9,
) -> RetrievedChunk:
    """Build a minimal ``RetrievedChunk`` with the given page metadata."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{filename}",
        filename=filename,
        chunk_index=0,
        text=text,
        score=score,
        page_number=page_number,
        created_at=None,
    )


def _same_file_pages(
    filename: str, pages: list[int | None]
) -> list[RetrievedChunk]:
    """Build chunks from *pages* under the same *filename*."""
    return [
        _chunk(f"{filename}-c{i}", filename, p, score=0.9 - i * 0.1)
        for i, p in enumerate(pages)
    ]


# ---------------------------------------------------------------------------
# select_page_diverse
# ---------------------------------------------------------------------------
class TestPageDiverseSelection:
    def test_empty_input_returns_empty(self) -> None:
        assert select_page_diverse([]) == []

    def test_diversity_avoids_single_page_domination(self) -> None:
        """When >= top_n distinct pages exist, pick one chunk per page."""
        a = _same_file_pages("a.pdf", [1, 1, 1])  # 3 chunks on page 1
        b = _chunk("b1", "a.pdf", 2, score=0.85)
        c = _chunk("c1", "a.pdf", 3, score=0.6)
        candidates = a + [b, c]

        selected = select_page_diverse(candidates, top_n=3)

        # One chunk from each distinct page, in reranked (input) order.
        assert [ch.page_number for ch in selected] == [1, 2, 3]
        assert selected[0].chunk_id == "a.pdf-c0"  # strongest preserved

    def test_preserves_strongest_candidate(self) -> None:
        """The first (highest-relevance) chunk is always kept."""
        chunks = _same_file_pages("doc.pdf", [1, 2, 3, 4])
        selected = select_page_diverse(chunks, top_n=3)
        assert selected[0].chunk_id == chunks[0].chunk_id

    def test_fewer_than_top_n_returns_all(self) -> None:
        """When fewer candidates than top_n, all are returned in order."""
        chunks = _same_file_pages("doc.pdf", [1, 2])
        selected = select_page_diverse(chunks, top_n=3)
        assert [ch.chunk_id for ch in selected] == [
            c.chunk_id for c in chunks
        ]

    def test_missing_page_metadata_falls_back_to_top_n(self) -> None:
        """Chunks with no page metadata collapse safely to plain top-n."""
        chunks = [
            _chunk("n1", "notes.txt", None, score=0.9),
            _chunk("n2", "notes.txt", None, score=0.8),
            _chunk("n3", "notes.txt", None, score=0.7),
            _chunk("n4", "notes.txt", None, score=0.6),
        ]
        selected = select_page_diverse(chunks, top_n=3)
        assert [ch.chunk_id for ch in selected] == ["n1", "n2", "n3"]

    def test_mixed_page_metadata_uses_available_pages(self) -> None:
        """Pages with metadata are diversified; missing-page fallback fills."""
        a1 = _chunk("a1", "doc.pdf", 1, score=0.95)
        a2 = _chunk("a2", "doc.pdf", 1, score=0.9)
        b = _chunk("b1", "other.pdf", None, score=0.7)
        c = _chunk("c1", "doc.pdf", 2, score=0.5)
        candidates = [a1, a2, b, c]

        selected = select_page_diverse(candidates, top_n=3)
        # a1 (page 1), then prefer new page -> other file (None), then page 2.
        assert [ch.chunk_id for ch in selected] == ["a1", "b1", "c1"]

    def test_deterministic(self) -> None:
        """Same input always yields the same selection."""
        chunks = _same_file_pages("doc.pdf", [1, 1, 2, 2, 3])
        first = select_page_diverse(chunks, top_n=3)
        for _ in range(5):
            assert select_page_diverse(chunks, top_n=3) == first

    def test_default_top_n_is_three(self) -> None:
        assert DEFAULT_TOP_N == 3
        chunks = _same_file_pages("doc.pdf", [1, 2, 3, 4])
        assert len(select_page_diverse(chunks)) == 3

    def test_invalid_top_n_raises(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            select_page_diverse([_chunk("x", "a.pdf", 1)], top_n=0)

    def test_chunks_not_mutated(self) -> None:
        """Selection returns references without modifying input chunks."""
        chunks = _same_file_pages("doc.pdf", [1, 1, 2])
        original_ids = [c.chunk_id for c in chunks]
        select_page_diverse(chunks, top_n=2)
        assert [c.chunk_id for c in chunks] == original_ids


# ---------------------------------------------------------------------------
# MiniLMReranker
# ---------------------------------------------------------------------------
class TestMiniLMReranker:
    def test_constructor_rejects_empty_model(self) -> None:
        with pytest.raises(ConfigurationError, match="model name"):
            MiniLMReranker(model_name="   ")

    def test_model_name_property(self) -> None:
        r = MiniLMReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_empty_candidates_returns_empty_without_loading_model(self) -> None:
        """Empty candidates short-circuit before any model load."""
        r = MiniLMReranker()
        with patch.object(r, "_load_model", side_effect=AssertionError("loaded")):
            assert r.rank("query", []) == []

    def test_rank_orders_by_score_descending(self) -> None:
        r = MiniLMReranker()
        candidates = [
            _chunk("c1", "a.pdf", 1, text="first"),
            _chunk("c2", "a.pdf", 2, text="second"),
            _chunk("c3", "a.pdf", 3, text="third"),
        ]
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.2, 0.9, 0.5]  # per candidate
        with patch.object(r, "_load_model", return_value=fake_model):
            ranked = r.rank("query", candidates)

        assert [c.chunk_id for c in ranked] == ["c2", "c3", "c1"]
        fake_model.predict.assert_called_once_with(
            [
                ("query", "first"),
                ("query", "second"),
                ("query", "third"),
            ]
        )

    def test_rank_ties_are_stable(self) -> None:
        """Equal scores keep the original order (deterministic)."""
        r = MiniLMReranker()
        candidates = [
            _chunk("c1", "a.pdf", 1, text="one"),
            _chunk("c2", "a.pdf", 2, text="two"),
        ]
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.5, 0.5]
        with patch.object(r, "_load_model", return_value=fake_model):
            ranked = r.rank("query", candidates)
        assert [c.chunk_id for c in ranked] == ["c1", "c2"]

    def test_load_failure_raises_chatbot_error(self) -> None:
        """Language/model loading failures raise a clear ChatbotError."""
        r = MiniLMReranker()
        with patch.object(
            r, "_load_model", side_effect=ChatbotError("not available")
        ):
            with pytest.raises(ChatbotError, match="not available"):
                r.rank("query", [_chunk("c1", "a.pdf", 1)])

    def test_missing_library_produces_chatbot_error(self) -> None:
        """A missing sentence-transformers package is surfaced clearly."""
        r = MiniLMReranker()
        # Simulate the optional dependency being absent/importable-as-none.
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ChatbotError, match="sentence-transformers"):
                r._load_model()

    def test_model_cached_after_load(self) -> None:
        """A cached model is returned without re-loading/importing."""
        r = MiniLMReranker()
        fake_model = MagicMock()
        r._model = fake_model  # simulate an already-loaded model
        # Even with the optional library unavailable, the cached model is
        # returned (the caching guard short-circuits before any import).
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            assert r._load_model() is fake_model


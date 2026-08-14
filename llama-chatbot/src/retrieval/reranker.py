"""Provider-neutral cross-encoder reranking + page-diverse selection.

This module wraps a MiniLM cross-encoder (via ``sentence-transformers``)
as an optional post-retrieval stage. It operates **only** on in-memory
``RetrievedChunk`` objects returned by the retriever — it never touches
Qdrant, never writes, and never modifies stored data.

Pipeline (validated in the experimental project):
    ``Qdrant Top-N → MiniLM reranker → greedy page-diverse Top-M``

The heavy ``sentence-transformers`` / PyTorch stack is imported lazily,
so enabling the feature flag only loads the model when a ``MiniLMReranker``
is actually used. Providers and callers depend on the abstract
``ChunkReranker`` interface, keeping this layer provider-neutral.

The page-diverse selection (:func:`select_page_diverse`) is pure and
deterministic: it keeps the strongest reranked candidate first, then
greedily prefers a distinct ``(filename, page_number)`` for each remaining
slot, and only falls back to the next-best chunk once every distinct page
in the pool is already represented. Missing page metadata is handled by
collapsing a file's chunks to the ``(file, None)`` key, which naturally
falls back to plain top-``top_n`` selection.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Final

from src.exceptions import ChatbotError, ConfigurationError
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

#: Default size of the final, page-diverse context set.
DEFAULT_TOP_N: Final[int] = 3


class ChunkReranker(ABC):
    """Abstract interface for re-ranking retrieved document chunks.

    Implementations re-order a candidate list of ``RetrievedChunk``
    objects by descending relevance to *query* and return the re-ranked
    list. They must be deterministic for identical inputs and must not
    mutate the input chunks.
    """

    @abstractmethod
    def rank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Re-rank *candidates* by descending relevance to *query*.

        Parameters
        ----------
        query:
            The user's retrieval query.
        candidates:
            The retrieved candidate chunks (the Qdrant Top-N pool).

        Returns
        -------
        list[RetrievedChunk]
            The same chunk objects (unchanged) ordered by descending
            relevance.  Returns ``[]`` when *candidates* is empty.
        """


class MiniLMReranker(ChunkReranker):
    """Re-rank chunks with a Hugging Face MiniLM cross-encoder.

    The cross-encoder model is loaded lazily on the first :meth:`rank`
    call so enabling the feature does not pay the model load cost until
    a query is actually reranked.  If the model or the
    ``sentence_transformers`` library is unavailable, a clear
    ``ChatbotError`` is raised (the higher-level RAG layer already
    falls back to a non-RAG answer on such errors).

    Parameters
    ----------
    model_name:
        Hugging Face cross-encoder identifier
        (default ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        stripped = model_name.strip()
        if not stripped:
            raise ConfigurationError("Reranker model name must not be empty.")
        self._model_name: str = stripped
        #: Lazily-initialised ``CrossEncoder`` instance (module-level import).
        self._model: object | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Score *candidates* with the cross-encoder and rank them desc.

        Ties keep their original order (stable sort), making the result
        deterministic for identical inputs.
        """
        if not candidates:
            return []

        model = self._load_model()

        pairs = [(query, chunk.text) for chunk in candidates]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
        ordered = [chunk for chunk, _ in ranked]
        logger.info(
            "Reranked %d candidate chunk(s) with model=%r",
            len(ordered),
            self._model_name,
        )
        return ordered

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        """Return the configured cross-encoder identifier."""
        return self._model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> object:
        """Load and cache the cross-encoder (lazy).

        Imports ``sentence_transformers`` here so the dependency is only
        required when the reranker feature is actually used.  Raises a
        ``ChatbotError`` with a clear message if loading fails or if the
        dependency is absent.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # noqa: BLE001 - dependency absence
            raise ChatbotError(
                "The reranker library (sentence-transformers) is not available.",
                safe_message="Document reranking is unavailable. Falling back to standard search.",
            ) from exc

        try:
            model = CrossEncoder(self._model_name)
        except Exception as exc:  # noqa: BLE001 - model download/load failure
            raise ChatbotError(
                f"Failed to load reranker model {self._model_name!r}: {exc}",
                safe_message="Document reranking is unavailable. Falling back to standard search.",
            ) from exc

        self._model = model
        return model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self._model_name!r})"


def select_page_diverse(
    chunks: list[RetrievedChunk], top_n: int = DEFAULT_TOP_N
) -> list[RetrievedChunk]:
    """Select up to *top_n* chunks with greedy page diversity.

    Expects *chunks* already ordered by descending relevance (e.g. the
    output of :meth:`ChunkReranker.rank`).  Selection is deterministic:

    1. Keep the strongest (first) reranked candidate.
    2. Prefer a chunk from a new ``(filename, page_number)`` for each
       remaining slot.
    3. Only fall back to any next-best chunk once every distinct page in
       the pool is already represented -- so a strong relevant page is
       never pushed out unnecessarily.

    Page metadata fallback: when ``page_number`` is ``None``, a file's
    chunks collapse to the ``(file, None)`` key.  Consequently, when page
    metadata is unavailable the function safely degrades to plain
    top-``top_n`` selection.

    Parameters
    ----------
    chunks:
        Candidate chunks ordered by descending relevance.
    top_n:
        Maximum number of chunks to select (default ``3``).

    Returns
    -------
    list[RetrievedChunk]
        The selected chunks in their original (relevance) order.  Returns
        ``[]`` when *chunks* is empty, and all chunks when there are
        ``<= top_n`` of them.
    """
    if not chunks:
        return []

    if top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n}")

    selected: list[RetrievedChunk] = []
    seen: set[tuple[str, int | None]] = set()
    all_pages = {(c.filename, c.page_number) for c in chunks}

    for chunk in chunks:
        if len(selected) >= top_n:
            break
        key = (chunk.filename, chunk.page_number)
        if key not in seen or seen == all_pages:
            selected.append(chunk)
            seen.add(key)

    return selected


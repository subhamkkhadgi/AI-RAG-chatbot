"""RAG context builder.

Formats retrieved document chunks into a clean, deterministic context
string suitable for consumption by an LLM.

The ``ContextBuilder`` depends **only** on ``RetrievalResult`` and
``RetrievedChunk`` — it does **not** import any concrete providers,
embedding classes, or vector-store classes.
"""

from __future__ import annotations

import logging
from typing import Final

from src.retrieval.models import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)

#: Separator between document chunks in the formatted output.
_SEPARATOR: Final[str] = "\n\n----------------------------------------\n\n"

#: Template for the header line that shows the source filename.
_HEADER_TEMPLATE: Final[str] = "Document: {filename}"

#: Template for the optional page-number line.
_PAGE_TEMPLATE: Final[str] = "\nPage: {page_number}"

#: Default maximum context length in characters.
_DEFAULT_MAX_CONTEXT_LENGTH: Final[int] = 4096


class ContextBuilder:
    """Build a formatted context string from a ``RetrievalResult``.

    Parameters
    ----------
    max_context_length:
        Maximum number of characters allowed in the rendered context
        string.  If the formatted output exceeds this limit it will be
        safely truncated (default ``4096``).
    """

    def __init__(self, max_context_length: int = _DEFAULT_MAX_CONTEXT_LENGTH) -> None:
        if max_context_length <= 0:
            raise ValueError(
                f"max_context_length must be positive, got {max_context_length}"
            )
        self._max_context_length = max_context_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, result: RetrievalResult) -> str:
        """Format retrieved chunks into a single context string.

        Duplicate chunks (identical text content) are filtered out —
        only the first occurrence of each unique text is kept. This
        prevents redundant context from being passed to the LLM while
        preserving metadata and scoring behavior.

        Parameters
        ----------
        result:
            The retrieval result whose *chunks* should be formatted.

        Returns
        -------
        str
            A clean, formatted context string ready for LLM consumption.
            Returns an empty string when *result* contains no chunks.
        """
        if not result.chunks:
            return ""

        # Filter duplicate chunks by text content (keep first occurrence)
        seen_texts: set[str] = set()
        deduped_chunks: list = []
        for chunk in result.chunks:
            if chunk.text not in seen_texts:
                seen_texts.add(chunk.text)
                deduped_chunks.append(chunk)

        parts: list[str] = []
        running_length = 0

        for i, chunk in enumerate(deduped_chunks):
            formatted = self._format_chunk(chunk, is_first=(i == 0))

            candidate_length = running_length + len(formatted)

            # If this chunk (including separator) would exceed the limit,
            # truncate it and stop.
            if candidate_length > self._max_context_length:
                remaining = self._max_context_length - running_length
                if remaining > 0:
                    truncated = self._truncate_chunk(chunk, remaining, is_first=(i == 0))
                    if truncated:
                        parts.append(truncated)
                break

            parts.append(formatted)
            running_length = candidate_length

        return "".join(parts)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def max_context_length(self) -> int:
        """Return the configured maximum context length."""
        return self._max_context_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_chunk(chunk: RetrievedChunk, *, is_first: bool) -> str:
        """Format a single chunk into its text representation.

        Parameters
        ----------
        chunk:
            The chunk to format.
        is_first:
            Whether this is the first chunk in the result (no leading
            separator).

        Returns
        -------
        str
            The formatted chunk string including metadata header and
            separator.
        """
        header = _HEADER_TEMPLATE.format(filename=chunk.filename)

        if chunk.page_number is not None:
            header += _PAGE_TEMPLATE.format(page_number=chunk.page_number)

        # Only the first chunk omits the leading separator.
        prefix = "" if is_first else _SEPARATOR

        return f"{prefix}{header}\n\n{chunk.text}\n"

    @staticmethod
    def _truncate_chunk(
        chunk: RetrievedChunk, max_length: int, *, is_first: bool
    ) -> str | None:
        """Render a chunk truncated to fit within *max_length* characters.

        Parameters
        ----------
        chunk:
            The chunk to format and truncate.
        max_length:
            Maximum allowed length for the rendered output.
        is_first:
            Whether this is the first chunk (no leading separator).

        Returns
        -------
        str | None
            The truncated chunk string, or ``None`` if even the header +
            separator cannot fit.
        """
        header = _HEADER_TEMPLATE.format(filename=chunk.filename)

        if chunk.page_number is not None:
            header += _PAGE_TEMPLATE.format(page_number=chunk.page_number)

        prefix = "" if is_first else _SEPARATOR

        # Compute the size of the fixed parts (prefix + header + "\n\n").
        fixed = f"{prefix}{header}\n\n"
        fixed_len = len(fixed)

        # Account for the trailing newline after the text.
        trailing_newline_len = 1  # "\n"

        # Remaining space for the actual chunk text.
        text_budget = max_length - fixed_len - trailing_newline_len

        if text_budget <= 0:
            # Even the header + newlines cannot fit.
            return None

        truncated_text = chunk.text[:text_budget]
        return f"{fixed}{truncated_text}\n"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_context_length={self._max_context_length})"
        )


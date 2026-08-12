"""Provider-neutral source citation formatting.

This module converts retrieved ``RetrievedChunk`` metadata into a
user-facing ``Sources:`` block that is appended to the assistant
response **after** LLM generation.  The LLM is never asked to produce
citations — the sources are derived entirely from retrieved document
metadata.

User-facing output shows only:
    - filename
    - page number (when available)

Internal implementation details (``document_id``, ``chunk_index``,
Qdrant point IDs, similarity scores) are intentionally **not** emitted.
"""

from __future__ import annotations

import logging
from typing import Final

from src.models.chat import SourceRef
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

#: Header line for the citations block.
_CITATIONS_HEADER: Final[str] = "\n\nSources:"

#: Bullet prefix for each source entry.
_BULLET: Final[str] = "- "


def build_citations_section(
    chunks: list[RetrievedChunk] | None,
) -> str:
    """Build a formatted ``Sources:`` section from retrieved chunks.

    Citations are deduplicated by ``(filename, page_number)`` while
    preserving first-seen order.  If multiple chunks come from the same
    document and page, only a single entry is emitted.

    Parameters
    ----------
    chunks:
        The ordered list of retrieved chunks (most relevant first), or
        ``None``.

    Returns
    -------
    str
        A formatted citations block, or an empty string when there are
        no usable chunks.  Never returns a bare ``"Sources:"`` header.
    """
    refs = build_source_refs(chunks)
    if not refs:
        return ""

    lines: list[str] = []
    for ref in refs:
        if ref.page_number is not None:
            lines.append(f"{_BULLET}{ref.filename} (Page {ref.page_number})")
        else:
            lines.append(f"{_BULLET}{ref.filename}")

    return f"{_CITATIONS_HEADER}\n" + "\n".join(lines)


def build_source_refs(
    chunks: list[RetrievedChunk] | None,
) -> list[SourceRef]:
    """Build a structured, deduplicated list of ``SourceRef`` objects.

    Citations are deduplicated by ``(filename, page_number)`` while
    preserving first-seen order.  The returned refs retain the retrieved
    chunk text (for "Relevant excerpt" in the UI) and the similarity
    score (internal only — never displayed).

    Parameters
    ----------
    chunks:
        The ordered list of retrieved chunks (most relevant first), or
        ``None``.

    Returns
    -------
    list[SourceRef]
        A list of structured source references.  Empty when there are no
        usable chunks.
    """
    if chunks is None:
        return []
    if not isinstance(chunks, (list, tuple)):
        return []
    if not chunks:
        return []

    refs: list[SourceRef] = []
    seen: set[tuple[str, int | None]] = set()

    for chunk in chunks:
        filename = getattr(chunk, "filename", "") or ""
        if not filename:
            logger.debug(
                "RAG_DIAG[citations] source_ref skipped | reason=empty_filename "
                "filename=%r page_number=%r created=False",
                filename,
                getattr(chunk, "page_number", None),
            )
            continue

        page_number = getattr(chunk, "page_number", None)

        key = (filename, page_number)
        if key in seen:
            logger.debug(
                "RAG_DIAG[citations] source_ref skipped | reason=duplicate "
                "filename=%r page_number=%r created=False",
                filename,
                page_number,
            )
            continue
        seen.add(key)

        refs.append(
            SourceRef(
                filename=filename,
                page_number=page_number,
                text=getattr(chunk, "text", None),
                score=getattr(chunk, "score", None),
            )
        )
        logger.debug(
            "RAG_DIAG[citations] source_ref created | filename=%r page_number=%r created=True",
            filename,
            page_number,
        )

    return refs


__all__ = ["build_citations_section", "build_source_refs"]


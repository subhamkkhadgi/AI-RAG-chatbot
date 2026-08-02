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

from typing import Final

from src.retrieval.models import RetrievedChunk

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
    if chunks is None:
        return ""
    if not isinstance(chunks, (list, tuple)):
        return ""
    if not chunks:
        return ""

    lines: list[str] = []
    seen: set[tuple[str, int | None]] = set()

    for chunk in chunks:
        filename = getattr(chunk, "filename", "") or ""
        if not filename:
            continue

        page_number = getattr(chunk, "page_number", None)

        key = (filename, page_number)
        if key in seen:
            continue
        seen.add(key)

        if page_number is not None:
            lines.append(f"{_BULLET}{filename} (Page {page_number})")
        else:
            lines.append(f"{_BULLET}{filename}")

    if not lines:
        return ""

    return f"{_CITATIONS_HEADER}\n" + "\n".join(lines)


__all__ = ["build_citations_section"]


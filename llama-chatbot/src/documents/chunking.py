"""Configurable text chunking for document processing.

The chunker splits plain text into overlapping chunks while attempting to
preserve paragraph boundaries.  It is independent of embedding generation
and vector storage.
"""

from __future__ import annotations

import logging
import re

from src.documents.models import Document, DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal chunking logic
# ---------------------------------------------------------------------------
def _paragraph_preserving_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int | None]]:
    """Split *text* into overlapping chunks, preserving paragraph boundaries.

    Paragraphs are delimited by one or more blank lines (``\\n\\n+``).
    Each paragraph that fits entirely within *chunk_size* is kept whole.
    Paragraphs longer than *chunk_size* are split by sentence boundaries,
    then by character boundary as a last resort.

    Parameters
    ----------
    text:
        The full plain-text document content.
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters to overlap between consecutive chunks.

    Returns
    -------
    list[tuple[str, int | None]]
        A list of ``(chunk_text, page_number)`` tuples.
    """
    # Split on form-feed characters (inserted by PDFExtractor per page)
    pages = text.split("\f")
    all_chunks: list[tuple[str, int | None]] = []

    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        # Split page text into paragraphs (one or more blank lines)
        paragraphs = re.split(r"\n\s*\n", page_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        page_chunks = _chunk_paragraphs(paragraphs, chunk_size, overlap)
        for chunk_text in page_chunks:
            all_chunks.append((chunk_text, page_idx))

    return all_chunks


def _chunk_paragraphs(
    paragraphs: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Group paragraphs into chunks respecting *chunk_size* and *overlap*."""
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # If adding this paragraph would exceed the chunk size, finalise current
        # and start a new one (with overlap).
        if current_chunk and current_len + para_len + 1 > chunk_size:
            chunk_text = "\n\n".join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)

            # Carry over overlap from the end of the finished chunk
            current_chunk = _build_overlap_window(
                current_chunk, current_len, overlap
            )
            current_len = sum(len(p) + 2 for p in current_chunk)

        current_chunk.append(para)
        current_len += para_len + (2 if len(current_chunk) > 1 else 0)

    # Final chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)

    # If any single chunk exceeds chunk_size (huge paragraph), split it
    return _split_oversized_chunks(chunks, chunk_size, overlap)


def _build_overlap_window(
    finished_paragraphs: list[str],
    finished_len: int,
    overlap: int,
) -> list[str]:
    """Build the overlap window from the tail of finished paragraphs."""
    if overlap <= 0 or not finished_paragraphs:
        return []

    accumulated: list[str] = []
    acc_len = 0

    for para in reversed(finished_paragraphs):
        para_len = len(para)
        if acc_len + para_len + 1 > overlap:
            remaining = overlap - acc_len
            if remaining > 0:
                accumulated.insert(0, para[-remaining:])
            break
        accumulated.insert(0, para)
        acc_len += para_len + 2

    return accumulated


def _split_oversized_chunks(
    chunks: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split any chunk that exceeds *chunk_size* by sentence boundaries."""
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            result.append(chunk)
            continue

        # Split by sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        temp: list[str] = []
        temp_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if temp and temp_len + sent_len + 1 > chunk_size:
                result.append(" ".join(temp))
                # Overlap: carry last sentences
                overlap_sents = _build_sentence_overlap(temp, temp_len, overlap)
                temp = overlap_sents
                temp_len = sum(len(s) + 1 for s in temp)
            temp.append(sent)
            temp_len += sent_len + (1 if temp else 0)

        if temp:
            result.append(" ".join(temp))

    return result


def _build_sentence_overlap(
    sentences: list[str],
    total_len: int,
    overlap: int,
) -> list[str]:
    """Build overlap window from the tail of sentences."""
    if overlap <= 0 or not sentences:
        return []

    accumulated: list[str] = []
    acc_len = 0

    for sent in reversed(sentences):
        sent_len = len(sent)
        if acc_len + sent_len + 1 > overlap:
            remaining = overlap - acc_len
            if remaining > 0:
                accumulated.insert(0, sent[-remaining:])
            break
        accumulated.insert(0, sent)
        acc_len += sent_len + 1

    return accumulated


# ---------------------------------------------------------------------------
# Main chunking class
# ---------------------------------------------------------------------------
class DocumentChunker:
    """Splits a ``Document`` into a list of ``DocumentChunk`` objects.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.  Read from Settings by default.
    overlap:
        Number of characters to overlap between consecutive chunks.
        Read from Settings by default.

    The chunker never generates empty chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self._chunk_size: int = chunk_size if chunk_size is not None else 1024
        self._overlap: int = overlap if overlap is not None else 128

        if self._chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if self._overlap < 0:
            raise ValueError("overlap must be a non-negative integer")
        if self._overlap >= self._chunk_size:
            raise ValueError("overlap must be less than chunk_size")

    @property
    def chunk_size(self) -> int:
        """Return the configured chunk size."""
        return self._chunk_size

    @property
    def overlap(self) -> int:
        """Return the configured overlap."""
        return self._overlap

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """Split *document* into chunks.

        Parameters
        ----------
        document:
            The document to split.

        Returns
        -------
        list[DocumentChunk]
            A list of chunks, in order, with metadata populated.

        Raises
        ------
        ValueError
            If the document content is empty.
        """
        content = document.content.strip()
        if not content:
            raise ValueError("Cannot chunk an empty document")

        raw_chunks = _paragraph_preserving_chunks(
            content,
            self._chunk_size,
            self._overlap,
        )

        document_chunks: list[DocumentChunk] = []
        for idx, (text, page_number) in enumerate(raw_chunks):
            chunk = DocumentChunk(
                document_id=document.document_id,
                filename=document.filename,
                chunk_index=idx,
                text=text,
                page_number=page_number,
            )
            document_chunks.append(chunk)

        return document_chunks

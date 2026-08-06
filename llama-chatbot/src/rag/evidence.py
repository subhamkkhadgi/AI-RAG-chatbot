"""Provider-neutral answer-aware citation support filtering.

This module determines which retrieved document chunks actually support
a generated answer using **lexical** matching — no semantic embeddings
and no additional LLM call.

The approach is deliberately simple and deterministic:

1. Tokenize both the generated answer and each chunk's text into
   normalized word tokens (lowercased, non-alphanumeric stripped).
2. For each chunk, compute the fraction of the answer's unique tokens
   that appear in the chunk (answer coverage).
3. A chunk is considered *supporting* when that coverage ratio meets a
   configurable threshold.

Because this is a retrieval-loading concern that runs **after** LLM
generation, it is intentionally provider-neutral: it imports only the
``RetrievedChunk`` model and never touches concrete providers, embedding
classes, or vector stores.
"""

from __future__ import annotations

import re
from typing import Final

from src.retrieval.models import RetrievedChunk

#: Default minimum fraction of the answer's unique tokens that must appear
#: in a chunk for that chunk to be considered supporting.
_DEFAULT_MIN_OVERLAP: Final[float] = 0.3

#: Splits text into lowercase alphanumeric word tokens.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")

#: Common English stopwords removed before overlap scoring.  These carry
#: little lexical signal and would otherwise inflate the overlap ratio
#: (e.g. "the" / "is" shared between an answer and an unrelated chunk).
_STOPWORDS: Final[set[str]] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "will", "with",
}


def _tokenize(text: str) -> set[str]:
    """Return the set of normalized word tokens in *text*.

    Stopwords are removed so they do not count as supporting evidence.

    Parameters
    ----------
    text:
        The raw text to tokenize.

    Returns
    -------
    set[str]
        The set of lowercase alphanumeric tokens, excluding stopwords.
        Empty when *text* is empty or contains only stopwords /
        non-alphanumeric tokens.
    """
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _overlap_ratio(answer_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Fraction of the answer's unique tokens that appear in the chunk.

    This is *answer coverage*: how much of the generated answer is
    supported by the chunk's text.  A long chunk that contains the
    answer's key tokens scores high regardless of how many other
    unrelated tokens the chunk carries.

    Parameters
    ----------
    answer_tokens:
        Tokenized answer.
    chunk_tokens:
        Tokenized chunk text.

    Returns
    -------
    float
        ``len(chunk_tokens & answer_tokens) / len(answer_tokens)`` (0.0
        when the answer has no tokens).
    """
    if not answer_tokens:
        return 0.0
    return len(chunk_tokens & answer_tokens) / len(answer_tokens)


def filter_supporting_chunks(
    answer: str,
    chunks: list[RetrievedChunk] | None,
    *,
    min_overlap: float = _DEFAULT_MIN_OVERLAP,
) -> list[RetrievedChunk]:
    """Return only the chunks that lexically support the *answer*.

    A chunk is retained when at least ``min_overlap`` of the answer's
    unique tokens appear in the chunk's text (answer coverage).  Chunks
    that do not support the answer are removed, which in turn prevents
    them from being cited.

    Parameters
    ----------
    answer:
        The generated LLM answer text.
    chunks:
        The ordered list of retrieved chunks (most relevant first), or
        ``None``.
    min_overlap:
        Minimum answer-coverage ratio required for a chunk to be
        considered supporting.  Must be in ``[0.0, 1.0]``.  Default
        ``0.3``.

    Returns
    -------
    list[RetrievedChunk]
        The subset of *chunks* that support the answer, preserving their
        original order.  Empty when *chunks* is ``None``/empty or when no
        chunk meets the overlap threshold.
    """
    if min_overlap < 0.0 or min_overlap > 1.0:
        raise ValueError(
            f"min_overlap must be between 0.0 and 1.0, got {min_overlap}"
        )
    if chunks is None:
        return []
    if not isinstance(chunks, (list, tuple)):
        return []
    if not chunks:
        return []

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        # No answer tokens to match against — nothing can be shown as
        # supporting its content.
        return []

    supporting: list[RetrievedChunk] = []
    for chunk in chunks:
        chunk_text = getattr(chunk, "text", "") or ""
        chunk_tokens = _tokenize(chunk_text)
        if not chunk_tokens:
            continue
        if _overlap_ratio(answer_tokens, chunk_tokens) >= min_overlap:
            supporting.append(chunk)

    return supporting


__all__ = ["filter_supporting_chunks"]

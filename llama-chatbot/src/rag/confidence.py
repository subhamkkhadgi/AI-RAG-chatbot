"""Provider-neutral retrieval confidence scoring.

This module decides whether retrieved context is strong enough to be
injected into the LLM prompt (confidence-aware RAG).

It only reads ``RetrievalResult`` / ``RetrievedChunk``-shaped objects
produced by the retrieval layer — no provider SDKs, no vector stores,
no UI framework.

Confidence strategy
-------------------
The primary confidence signal is the **highest (best) chunk similarity
score** in the retrieval result.  A single strong hit is more reliable
evidence than many weak hits.

- A real retrieval result with chunks -> ``max(chunk.score)``.
- A real retrieval result with no chunks (or no usable scores) -> ``0.0``
  (treated as low-confidence / unusable context).
- An unknown / non-iterable ``chunks`` value (e.g. a mock in tests) ->
  ``None``, meaning *confidence cannot be determined*.  Callers that
  cannot determine confidence fall back to the previous behaviour so
  existing flows are preserved.
"""

from __future__ import annotations

from typing import Any

__all__ = ["compute_retrieval_confidence", "is_confident"]


def compute_retrieval_confidence(retrieval_result: Any) -> float | None:
    """Compute a confidence score for a retrieval result.

    Parameters
    ----------
    retrieval_result:
        A ``RetrievalResult`` (or a duck-typed object exposing ``chunks``).

    Returns
    -------
    float | None
        - ``max(chunk.score)`` for a real result with usable chunks.
        - ``0.0`` for a real result with no chunks (or no usable scores).
        - ``None`` when the result structure is unrecognisable (e.g. a
          mock from tests) — confidence *cannot* be determined.
    """
    chunks = getattr(retrieval_result, "chunks", None)
    if not isinstance(chunks, (list, tuple)):
        # Unknown / non-iterable chunk list — cannot judge confidence.
        return None
    if not chunks:
        return 0.0

    scores: list[float] = []
    for chunk in chunks:
        score = getattr(chunk, "score", None)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            scores.append(float(score))

    if not scores:
        return 0.0

    return float(max(scores))


def is_confident(retrieval_result: Any, threshold: float | None) -> bool:
    """Decide whether a retrieval result meets the confidence threshold.

    Parameters
    ----------
    retrieval_result:
        A ``RetrievalResult`` (or a duck-typed object exposing ``chunks``).
    threshold:
        Minimum confidence required.  ``None`` disables the gate — every
        retrieval is considered confident (existing behaviour).

    Returns
    -------
    bool
        ``True`` when the result is confident enough to use its context.
    """
    if threshold is None:
        return True

    confidence = compute_retrieval_confidence(retrieval_result)
    if confidence is None:
        # Cannot determine confidence — fall back to using context to
        # preserve existing behaviour (e.g. MagicMock results in tests).
        return True

    return confidence >= threshold


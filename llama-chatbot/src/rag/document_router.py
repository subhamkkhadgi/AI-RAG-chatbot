"""Provider-neutral query-aware document routing.

Selects a specific uploaded document when a user query clearly refers to
one, so retrieval can be scoped to that document. This is a lightweight,
deterministic router — no LLM, no external dependencies.

Routing signals (all deterministic):
- Exact / partial filename mention in the query.
- Non-trivial filename tokens appearing in the query.
- Common aliases (cv/resume, proposal, report) matched via filename tokens.

When no document is confidently identified, the router returns an empty
scope so the caller falls back to global (unrestricted) retrieval.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel

from src.retrieval.models import DocumentScope

logger = logging.getLogger(__name__)

#: Minimum confidence score required to route to a single document.
_ROUTE_THRESHOLD: float = 1.0

#: Tokens that are too generic to disambiguate a document.
_COMMON_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
        "of", "to", "in", "on", "for", "with", "about", "what", "how", "why",
        "who", "when", "where", "tell", "me", "explain", "describe", "this",
        "that", "these", "those", "page", "doc", "document", "file", "pdf",
    }
)

#: Alias rules: (tokens in the filename, query terms that activate the alias).
_ALIAS_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("cv", "resume"), ("cv", "resume")),
    (("proposal",), ("proposal", "project")),
    (("report",), ("report",)),
]


class RouteResult(BaseModel):
    """The outcome of a routing attempt.

    Attributes:
        scope: The selected ``DocumentScope`` (empty when no routing).
        confidence: The confidence score of the best match.
        routed: Whether a document was selected (scope is non-empty).
    """

    scope: DocumentScope
    confidence: float = 0.0
    routed: bool = False

    model_config = {"frozen": True}


class DocumentRouter:
    """Route a query to a specific uploaded document when confident.

    Parameters
    ----------
    documents:
        A list of document records, each containing ``filename`` and
        ``document_id`` keys (as returned by ``DocumentManager``).
    threshold:
        Minimum confidence score required to route. Defaults to
        ``_ROUTE_THRESHOLD``.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        threshold: float = _ROUTE_THRESHOLD,
    ) -> None:
        self._documents: list[dict[str, Any]] = documents or []
        self._threshold: float = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def route(self, query: str) -> RouteResult:
        """Route a query to a specific document, if confident.

        Parameters
        ----------
        query:
            The user's query text.

        Returns
        -------
        RouteResult
            A route result with the selected scope (empty when no
            document is confidently identified).
        """
        if not self._documents or not query or not query.strip():
            return RouteResult(scope=DocumentScope(), confidence=0.0, routed=False)

        normalized_query = self._normalize(query)
        if not normalized_query:
            return RouteResult(scope=DocumentScope(), confidence=0.0, routed=False)

        best_doc: dict[str, Any] | None = None
        best_score: float = 0.0

        for doc in self._documents:
            filename = str(doc.get("filename", "") or "")
            if not filename:
                continue
            score = self._score_document(normalized_query, filename)
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is not None and best_score >= self._threshold:
            scope = DocumentScope(
                document_ids=[str(best_doc.get("document_id", ""))],
                filenames=[str(best_doc.get("filename", ""))],
            )
            logger.info(
                "Routed query to '%s' (confidence=%.2f)",
                best_doc.get("filename"),
                best_score,
            )
            return RouteResult(scope=scope, confidence=best_score, routed=True)

        logger.debug("No confident document route for query (best=%.2f)", best_score)
        return RouteResult(scope=DocumentScope(), confidence=best_score, routed=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _score_document(self, normalized_query: str, filename: str) -> float:
        """Score how well a filename matches the query.

        Returns
        -------
        float
            A confidence score. Higher is more confident.
        """
        score: float = 0.0

        normalized_filename = self._normalize(filename)
        if not normalized_filename:
            return 0.0

        # 1. Full normalized filename mention (strongest signal).
        if normalized_filename in normalized_query:
            score += 3.0

        # 2. Full original filename mention.
        if filename.lower() in normalized_query:
            score += 3.0

        # 3. Significant filename tokens appearing in the query.
        tokens = self._filename_tokens(normalized_filename)
        for token in tokens:
            if token in self._COMMON_STOPWORDS:
                continue
            if token in normalized_query:
                score += 1.0

        # 4. Alias rules (cv/resume, proposal, report).
        for alias_tokens, query_terms in _ALIAS_RULES:
            if any(t in tokens for t in alias_tokens):
                if any(term in normalized_query for term in query_terms):
                    score += 1.5

        return score

    @staticmethod
    def _filename_tokens(normalized_filename: str) -> set[str]:
        """Return the set of whitespace-delimited tokens."""
        return set(normalized_filename.split())

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and replace non-alphanumeric separators with spaces."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

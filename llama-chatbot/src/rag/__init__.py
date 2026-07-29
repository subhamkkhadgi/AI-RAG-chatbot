"""RAG (Retrieval-Augmented Generation) context building and orchestration.

The ``ContextBuilder`` formats retrieved document chunks into a clean,
LLM-friendly context string.

The ``RAGService`` orchestrates the full RAG pipeline: retrieval via
``DocumentRetriever``, context formatting via ``ContextBuilder``, and
returns a ``RAGResult`` with structured and formatted results.
"""

from __future__ import annotations

from src.rag.context_builder import ContextBuilder
from src.rag.rag_service import RAGResult, RAGService

__all__ = [
    "ContextBuilder",
    "RAGResult",
    "RAGService",
]


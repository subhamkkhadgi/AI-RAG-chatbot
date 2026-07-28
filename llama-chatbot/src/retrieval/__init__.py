"""Provider-neutral retrieval layer.

Retrieval is independent from the chatbot conversation flow.
It generates query embeddings and searches a vector store,
returning relevant document chunks.
"""

from __future__ import annotations

from src.retrieval.models import RetrievedChunk, RetrievalResult
from src.retrieval.retriever import DocumentRetriever

__all__ = [
    "DocumentRetriever",
    "RetrievedChunk",
    "RetrievalResult",
]

"""Provider-neutral retrieval layer.

Retrieval is independent from the chatbot conversation flow.
It generates query embeddings and searches a vector store,
returning relevant document chunks.
"""

from __future__ import annotations

from src.retrieval.models import RetrievedChunk, RetrievalResult
from src.retrieval.reranker import ChunkReranker, MiniLMReranker, select_page_diverse
from src.retrieval.retriever import DocumentRetriever

__all__ = [
    "ChunkReranker",
    "DocumentRetriever",
    "MiniLMReranker",
    "RetrievedChunk",
    "RetrievalResult",
    "select_page_diverse",
]

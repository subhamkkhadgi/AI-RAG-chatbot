"""Pydantic models for retrieval results.

These models are provider-neutral and contain no Qdrant-specific or
Ollama-specific logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single document chunk returned by a retrieval search.

    Attributes:
        chunk_id: Unique identifier of the chunk in the vector store.
        document_id: The document this chunk belongs to.
        filename: Original filename of the source document.
        chunk_index: Zero-based index of this chunk within the document.
        text: The chunk text content.
        score: Similarity score from the vector search (higher is more relevant).
        page_number: Optional page number if extracted from a page-based format.
        created_at: Optional UTC timestamp when the chunk was created.
    """

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int = Field(ge=0)
    text: str
    score: float = Field(ge=0.0)
    page_number: int | None = None
    created_at: str | None = None

    model_config = {"frozen": True}


class RetrievalResult(BaseModel):
    """The complete result of a retrieval query.

    Attributes:
        query: The original query text that produced these results.
        chunks: Ordered list of retrieved chunks (most relevant first).
        total_results: Total number of chunks returned.
    """

    query: str
    chunks: list[RetrievedChunk]
    total_results: int

    model_config = {"frozen": True}

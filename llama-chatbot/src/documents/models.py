"""Pydantic models for document processing.

These models are provider-neutral and contain no SDK-specific logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    """A raw document with extracted plain-text content.

    Attributes:
        document_id: Unique identifier for the document.
        filename: Original filename (e.g. ``"report.pdf"``).
        content: The full plain-text content of the document.
    """

    document_id: str = Field(default_factory=lambda: uuid4().hex)
    filename: str
    content: str

    @field_validator("filename")
    @classmethod
    def _filename_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("filename must not be empty")
        return stripped

    model_config = {"frozen": True}


class DocumentChunk(BaseModel):
    """A single chunk of text from a document, ready for embedding.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        document_id: The document this chunk belongs to.
        filename: Original filename of the source document.
        chunk_index: Zero-based index of this chunk within the document.
        text: The chunk text content.
        page_number: Optional page number if extracted from a page-based format.
        created_at: UTC timestamp when this chunk was created.
    """

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    filename: str
    chunk_index: int = Field(ge=0)
    text: str
    page_number: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("text")
    @classmethod
    def _text_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("chunk text must not be empty or whitespace-only")
        return stripped

    @field_validator("document_id")
    @classmethod
    def _document_id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be empty")
        return v.strip()

    model_config = {"frozen": True}

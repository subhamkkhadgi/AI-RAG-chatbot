"""Document processing abstractions for extraction, chunking, ingestion,
and management.

The public API currently exposes:

- ``Document`` — Pydantic model for a raw document.
- ``DocumentChunk`` — Pydantic model for a text chunk.
- ``BaseDocumentExtractor`` — abstract base class for document extractors.
- ``PDFExtractor`` — extracts text from PDF files using pdfplumber.
- ``TXTExtractor`` — extracts text from plain text files using UTF-8.
- ``extract_document`` — factory function that selects the right extractor.
- ``DocumentChunker`` — splits extracted text into overlapping chunks.
- ``DocumentIngestionService`` — orchestrates the full ingestion pipeline
  from file to vector storage.
- ``DocumentManager`` — lists and deletes uploaded documents via the
  vector store.
"""

from .chunking import DocumentChunker
from .document_manager import DocumentManager
from .extractors import PDFExtractor, TXTExtractor, extract_document
from .ingestion_service import DocumentIngestionService
from .models import Document, DocumentChunk

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentIngestionService",
    "DocumentManager",
    "PDFExtractor",
    "TXTExtractor",
    "extract_document",
]

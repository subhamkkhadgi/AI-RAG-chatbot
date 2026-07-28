"""Document processing abstractions for extraction and chunking.

The public API currently exposes:

- ``Document`` — Pydantic model for a raw document.
- ``DocumentChunk`` — Pydantic model for a text chunk.
- ``BaseDocumentExtractor`` — abstract base class for document extractors.
- ``PDFExtractor`` — extracts text from PDF files using pdfplumber.
- ``TXTExtractor`` — extracts text from plain text files using UTF-8.
- ``extract_document`` — factory function that selects the right extractor.
- ``DocumentChunker`` — splits extracted text into overlapping chunks.
"""

from .chunking import DocumentChunker
from .extractors import PDFExtractor, TXTExtractor, extract_document
from .models import Document, DocumentChunk

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentChunker",
    "PDFExtractor",
    "TXTExtractor",
    "extract_document",
]

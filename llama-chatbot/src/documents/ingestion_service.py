"""Provider‑neutral document ingestion pipeline.

Orchestrates the full flow:
    document file  →  extract text  →  split into chunks
    →  generate embeddings  →  store vectors + metadata

The service depends only on abstract provider interfaces and existing
document abstractions.  Concrete providers (Ollama, Qdrant, …) are
injected at construction time.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from src.documents.chunking import DocumentChunker
from src.documents.extractors import extract_document
from src.documents.models import Document, DocumentChunk
from src.embeddings.base import BaseEmbeddingProvider
from src.exceptions import ConfigurationError, ProviderConnectionError, ProviderResponseError
from src.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CHUNK_SIZE = 1024
_DEFAULT_OVERLAP = 128
_DEFAULT_COLLECTION_NAME = "documents"


# ---------------------------------------------------------------------------
# Ingestion service
# ---------------------------------------------------------------------------
class DocumentIngestionService:
    """Orchestrate document ingestion from file to vector storage.

    Parameters
    ----------
    embedding_provider:
        An embedding provider that implements ``BaseEmbeddingProvider``.
    vector_store:
        A vector store that implements ``BaseVectorStore``.
    chunk_size:
        Maximum number of characters per chunk (default 1024).
    overlap:
        Number of characters to overlap between consecutive chunks
        (default 128).
    collection_name:
        Vector store collection name (default ``"documents"``).
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._collection_name = collection_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_file(self, file_path: str, original_filename: str | None = None) -> str:
        """Ingest a document file into the vector store.

        Parameters
        ----------
        file_path:
            Path to the document file (``.pdf`` or ``.txt``).
        original_filename:
            Original uploaded filename to store in the document metadata.
            If ``None``, falls back to the basename of *file_path*.

        Returns
        -------
        str
            The ``document_id`` assigned to the ingested document.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ConfigurationError
            If the file type is unsupported, extraction fails, or the
            document is empty.
        ProviderConnectionError
            If the embedding provider or vector store cannot be reached.
        ProviderResponseError
            If the embedding provider or vector store returns an
            unexpected response.
        """
        # 1. Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 2. Extract text via existing extractor pipeline
        logger.info("Ingesting file: %s", file_path)
        content = extract_document(file_path)

        # 3. Build Document model and validate
        # Use original_filename when provided, otherwise fall back to path.name
        filename = original_filename if original_filename is not None else path.name
        document = Document(filename=filename, content=content)

        if not document.content.strip():
            raise ConfigurationError(
                f"Document '{document.filename}' contains no extractable content."
            )

        # 4. Chunk the document
        chunker = DocumentChunker(
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )
        chunks = chunker.chunk_document(document)

        if not chunks:
            raise ConfigurationError(
                f"Document '{document.filename}' produced zero chunks."
            )

        # 5. Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        logger.info(
            "Generating embeddings for %d chunks from '%s'",
            len(texts),
            document.filename,
        )
        vectors = self._embedding_provider.embed_batch(texts)

        if len(vectors) != len(chunks):
            raise ProviderResponseError(
                f"Embedding provider returned {len(vectors)} vectors "
                f"for {len(chunks)} chunks."
            )

        # 6. Build payloads with required metadata
        chunk_count = len(chunks)
        payloads: list[dict] = []
        for chunk in chunks:
            payload: dict = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "chunk_count": chunk_count,
                "text": chunk.text,
                "created_at": chunk.created_at.isoformat(),
            }
            if chunk.page_number is not None:
                payload["page_number"] = chunk.page_number
            payloads.append(payload)

        # 7. Ensure collection exists
        if vectors:
            vector_size = len(vectors[0])
            try:
                self._vector_store.create_collection(vector_size)
            except Exception:
                # It is common for the collection to already exist; log and proceed.
                logger.debug(
                    "Collection '%s' may already exist; proceeding.",
                    self._collection_name,
                )

        # 8. Store vectors
        logger.info(
            "Storing %d vectors in collection '%s'",
            len(vectors),
            self._collection_name,
        )
        self._vector_store.upsert(vectors, payloads)

        logger.info(
            "Successfully ingested '%s' (document_id=%s, chunks=%d)",
            document.filename,
            document.document_id,
            chunk_count,
        )
        return document.document_id

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        """Return the configured embedding provider."""
        return self._embedding_provider

    @property
    def vector_store(self) -> BaseVectorStore:
        """Return the configured vector store."""
        return self._vector_store

    @property
    def chunk_size(self) -> int:
        """Return the configured chunk size."""
        return self._chunk_size

    @property
    def overlap(self) -> int:
        """Return the configured overlap."""
        return self._overlap

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""
        return self._collection_name


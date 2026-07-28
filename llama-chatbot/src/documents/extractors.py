"""Document extractors for PDF and TXT files.

Each extractor implements a common interface and returns plain text.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseDocumentExtractor(ABC):
    """Abstract interface for a document extractor.

    Subclasses must define:

    - ``supported_extensions`` — a frozenset of lowercase file extensions.
    - ``extract`` — return the plain-text content of a file.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """Return the set of file extensions this extractor handles.

        Example: ``frozenset({".pdf"})`` or ``frozenset({".txt"})``.
        """

    @abstractmethod
    def extract(self, file_path: str) -> str:
        """Extract and return plain text from *file_path*.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the document file.

        Returns
        -------
        str
            The full plain-text content of the document.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ConfigurationError
            If the file type is not supported or extraction fails.
        """


# ---------------------------------------------------------------------------
# PDF extractor
# ---------------------------------------------------------------------------
class PDFExtractor(BaseDocumentExtractor):
    """Extract text from PDF files using ``pdfplumber``."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def extract(self, file_path: str) -> str:
        """Extract text from a PDF file.

        Uses ``pdfplumber`` to iterate over pages and concatenate text.
        Each page's text is separated by a form-feed character (``\\f``)
        so downstream chunking can optionally track page boundaries.

        Raises
        ------
        ConfigurationError
            If pdfplumber is not installed or extraction fails.
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import pdfplumber  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigurationError(
                "pdfplumber is required for PDF extraction. "
                "Install it with: pip install pdfplumber"
            ) from exc

        pages: list[str] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(text.strip())
        except Exception as exc:
            logger.error("PDF extraction failed for %s: %s", file_path, exc)
            raise ConfigurationError(
                f"Failed to extract text from PDF: {file_path}"
            ) from exc

        result = "\f".join(pages)
        if not result.strip():
            raise ConfigurationError(
                f"PDF file '{file_path}' contains no extractable text."
            )

        return result


# ---------------------------------------------------------------------------
# TXT extractor
# ---------------------------------------------------------------------------
class TXTExtractor(BaseDocumentExtractor):
    """Extract text from plain text files using UTF-8 encoding."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".txt"})

    def extract(self, file_path: str) -> str:
        """Extract text from a plain text file.

        Reads the file with UTF-8 encoding.

        Raises
        ------
        ConfigurationError
            If the file is empty or cannot be decoded.
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            logger.error("UTF-8 decode failed for %s: %s", file_path, exc)
            raise ConfigurationError(
                f"File '{file_path}' is not valid UTF-8 text."
            ) from exc
        except Exception as exc:
            logger.error("Failed to read %s: %s", file_path, exc)
            raise ConfigurationError(
                f"Failed to read file: {file_path}"
            ) from exc

        stripped = text.strip()
        if not stripped:
            raise ConfigurationError(
                f"Text file '{file_path}' is empty."
            )

        return stripped


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_extractor_registry: dict[str, BaseDocumentExtractor] = {}


def _register_extractor(extractor: BaseDocumentExtractor) -> None:
    """Register an extractor instance for its supported extensions."""
    for ext in extractor.supported_extensions:
        _extractor_registry[ext.lower()] = extractor


# Register known extractors
_register_extractor(PDFExtractor())
_register_extractor(TXTExtractor())


def extract_document(file_path: str) -> str:
    """Extract plain text from a document file.

    Selects the appropriate extractor based on the file extension.

    Parameters
    ----------
    file_path:
        Path to the document file.

    Returns
    -------
    str
        The full plain-text content of the document.

    Raises
    ------
    ConfigurationError
        If the file type is unsupported or extraction fails.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    extractor = _extractor_registry.get(ext)
    if extractor is None:
        supported = ", ".join(sorted(_extractor_registry.keys()))
        raise ConfigurationError(
            f"Unsupported file type '{ext}'. Supported types: {supported}."
        )

    logger.info("Extracting text from %s using %s", file_path, type(extractor).__name__)
    return extractor.extract(file_path)

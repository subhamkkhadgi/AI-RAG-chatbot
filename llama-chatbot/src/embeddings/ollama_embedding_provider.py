"""Ollama embedding provider implementation.

This provider wraps the ``ollama`` Python SDK and implements the
``BaseEmbeddingProvider`` interface.  It is stateless: each ``embed()``
or ``embed_batch()`` call communicates with Ollama and returns vectors.

The model name is sourced from Settings via ``embedding_model`` —
never hardcoded — so any Ollama-compatible embedding model can be used
by changing configuration only.
"""

from __future__ import annotations

import logging
from typing import Any

import ollama

from src.embeddings.base import BaseEmbeddingProvider
from src.exceptions import (
    MissingCredentialsError,
    ProviderConnectionError,
    ProviderResponseError,
)

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Provider that communicates with a local Ollama instance for embeddings.

    Parameters
    ----------
    settings:
        A validated ``Settings`` instance containing *ollama_host* and
        *embedding_model* values.
    """

    def __init__(self, settings: object) -> None:
        self._host: str = getattr(settings, "ollama_host", "http://localhost:11434")
        self._model: str = getattr(settings, "embedding_model", "")
        self._client: ollama.Client = ollama.Client(host=self._host)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        """Return ``"ollama"`` as the stable provider identifier."""
        return "ollama"

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    def validate_configuration(self) -> None:
        """Verify that the Ollama host and embedding model are present.

        This is a local-only check — no network call is made.
        Network reachability is tested by :meth:`is_available`.
        """
        if not self._host:
            raise MissingCredentialsError(
                self.provider_name,
                safe_message="Ollama host is not configured. Set OLLAMA_HOST in your .env file.",
            )
        if not self._model:
            raise MissingCredentialsError(
                self.provider_name,
                safe_message="Embedding model is not configured. Set EMBEDDING_MODEL in your .env file.",
            )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Uses a lightweight ``list()`` call (no model loading) so it is
        safe to call frequently.  Returns ``False`` on any connection
        issue without raising.
        """
        try:
            self._client.list()
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Ollama availability check failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Parameters
        ----------
        text:
            The input text to embed.

        Returns
        -------
        list[float]
            A dense vector representation of the input text.

        Raises
        ------
        ProviderConnectionError
            If Ollama cannot be reached.
        ProviderResponseError
            If the response is unexpected or malformed.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of vectors.

        Parameters
        ----------
        texts:
            A list of input text strings to embed.

        Returns
        -------
        list[list[float]]
            A list of dense vectors, one per input text, in the same order.

        Raises
        ------
        ProviderConnectionError
            If Ollama cannot be reached.
        ProviderResponseError
            If the response is unexpected or malformed.
        """
        if not texts:
            return []

        response = self._call_embed_api(texts)
        return self._validate_and_return_embeddings(response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call_embed_api(self, texts: list[str]) -> Any:
        """Call the Ollama embed API and handle transport-level errors.

        Parameters
        ----------
        texts:
            The input texts to embed.

        Returns
        -------
        Any
            The raw response from the Ollama ``embed()`` call.

        Raises
        ------
        ProviderConnectionError
            If Ollama cannot be reached.
        ProviderResponseError
            If Ollama returns an API error or an unexpected error occurs.
        """
        try:
            return self._client.embed(
                model=self._model,
                input=texts,
            )
        except ollama.ResponseError as exc:
            logger.error(
                "Ollama embedding response error (status=%s): %s",
                exc.status_code,
                exc.error,
            )
            raise ProviderResponseError(
                self.provider_name,
                safe_message=(
                    f"Ollama returned an error (status {exc.status_code}). "
                    "Check the logs for details."
                ),
            ) from exc
        except ConnectionError as exc:
            logger.error("Ollama connection error: %s", exc)
            raise ProviderConnectionError(
                self.provider_name,
                safe_message="Cannot connect to Ollama. Make sure it is installed and running.",
            ) from exc
        except Exception as exc:
            logger.error("Unexpected Ollama error: %s", exc, exc_info=True)
            raise ProviderResponseError(
                self.provider_name,
                safe_message="An unexpected error occurred while communicating with Ollama.",
            ) from exc

    @staticmethod
    def _validate_and_return_embeddings(response: Any) -> list[list[float]]:
        """Validate the response and return extracted embeddings.

        Parameters
        ----------
        response:
            The raw response from the Ollama ``embed()`` call.

        Returns
        -------
        list[list[float]]
            Extracted embeddings.

        Raises
        ------
        ProviderResponseError
            If the response contains no embeddings or is malformed.
        """
        embeddings = OllamaEmbeddingProvider._extract_embeddings(response)
        if not embeddings:
            raise ProviderResponseError(
                "ollama",
                safe_message="Ollama returned an empty embedding response.",
            )
        return embeddings

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        """Extract embeddings from an Ollama ``embed()`` response.

        Handles **both** object-style responses (``response.embeddings``)
        and dict-style responses (``response["embeddings"]``).
        """
        # Object-style: response.embeddings
        embeddings = getattr(response, "embeddings", None)
        if embeddings is not None and isinstance(embeddings, list):
            return embeddings

        # Dict-style: response["embeddings"]
        if isinstance(response, dict):
            embeddings = response.get("embeddings")
            if isinstance(embeddings, list):
                return embeddings

        return []

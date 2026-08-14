"""Application configuration loaded from environment variables.

Uses Pydantic ``BaseSettings`` (v2) to read and validate ``.env``.

Usage
-----
    from src.config import get_settings

    settings = get_settings()
    print(settings.llm_provider)

Testing
-------
    from src.config import get_settings, clear_settings_cache

    clear_settings_cache()  # force re-read on next ``get_settings()``
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.prompts.system_prompts import get_default_prompt

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------
VALID_PROVIDERS: Final[frozenset[str]] = frozenset({"groq", "ollama"})
VALID_EMBEDDING_PROVIDERS: Final[frozenset[str]] = frozenset({"ollama"})
VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``.

    Only **shared** settings are validated here.  Provider-specific
    credentials (e.g. ``GROQ_API_KEY``) are validated at provider
    construction time so the application can start even when one provider
    is not configured.
    """

    # ── Provider selection ────────────────────────────────────────────
    llm_provider: str = "groq"

    # ── Groq (optional — validated when selected) ─────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"

    # ── Document processing ──────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # ── Embedding provider ────────────────────────────────────────────
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"

    # ── Retrieval precision ───────────────────────────────────────────
    #: Minimum cosine-similarity score for a chunk to be considered
    #: relevant.  Chunks with ``score < retrieval_min_score`` are dropped.
    retrieval_min_score: float = 0.3
    #: Maximum number of documents allowed in a single retrieval result.
    #: ``0`` means unlimited (threshold-only filtering).  A positive value
    #: keeps only the top-``max_documents`` documents ranked by their best
    #: chunk score, which prevents unrelated documents from being mixed in
    #: just because they appear inside the top-k results.
    retrieval_max_documents: int = 1

    #: Minimum retrieval confidence required to inject retrieved context
    #: into the LLM prompt (confidence-aware RAG).  Confidence is measured
    #: as the highest chunk similarity score in the retrieval result.
    #: When retrieval confidence is below this threshold, the weak/irrelevant
    #: context is NOT injected and the LLM answers from its own knowledge
    #: (with no citations).
    retrieval_confidence_threshold: float = 0.6

    # ── Cross-encoder reranking (feature-flagged, default OFF) ─────────
    #: Whether to rerank the retrieved candidate pool with a MiniLM
    #: cross-encoder and select a final, page-diverse context set.
    #: ``False`` (default) preserves the exact existing retrieval behaviour.
    #: When ``True``, the pipeline becomes
    #: ``Qdrant Top-N -> reranker -> greedy page-diverse Top-M -> context``
    #: and the retriever is relaxed to produce a genuine cross-document
    #: candidate pool (``max_documents`` unlimited, ``default_limit`` =
    #: ``reranker_candidate_limit``).
    reranker_enabled: bool = False
    #: Hugging Face cross-encoder model identifier used for reranking.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    #: Size of the candidate pool retrieved from the vector store before
    #: reranking (the Qdrant "Top-N").
    reranker_candidate_limit: int = 10
    #: Number of chunks kept after reranking + page-diverse selection
    #: (the final context set size).
    reranker_final_limit: int = 3

    # ── Qdrant (optional — validated when selected) ───────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # ── Ollama (optional — validated when selected) ───────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # ── Defaults (overridable at runtime via sidebar) ─────────────────
    #: Default system prompt.  Single source of truth lives in the
    #: provider-neutral ``src.prompts.system_prompts`` module; this field
    #: delegates to it so the app and the prompt registry never diverge.
    default_system_prompt: str = get_default_prompt()
    temperature: float = 0.7
    max_tokens: int = 2048

    # ── Networking ────────────────────────────────────────────────────
    request_timeout: int = 30

    # ── Observability ─────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Pydantic model configuration ──────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Validators ────────────────────────────────────────────────────

    @field_validator("llm_provider")
    @classmethod
    def _normalise_provider(cls, v: str) -> str:
        stripped = v.strip().lower()
        if stripped not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid LLM_PROVIDER {v!r}. Must be one of: {', '.join(sorted(VALID_PROVIDERS))}."
            )
        return stripped

    @field_validator("embedding_provider")
    @classmethod
    def _normalise_embedding_provider(cls, v: str) -> str:
        stripped = v.strip().lower()
        if stripped not in VALID_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER {v!r}. Must be one of: {', '.join(sorted(VALID_EMBEDDING_PROVIDERS))}."
            )
        return stripped

    @field_validator("groq_model", "ollama_model", "embedding_model")
    @classmethod
    def _strip_model_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("ollama_host")
    @classmethod
    def _validate_ollama_host(cls, v: str) -> str:
        stripped = v.strip().rstrip("/")
        if not stripped.startswith(("http://", "https://")):
            raise ValueError(
                f"OLLAMA_HOST must start with http:// or https://, got {v!r}"
            )
        return stripped

    @field_validator("retrieval_min_score")
    @classmethod
    def _validate_retrieval_min_score(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(
                f"RETRIEVAL_MIN_SCORE must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("retrieval_max_documents")
    @classmethod
    def _validate_retrieval_max_documents(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"RETRIEVAL_MAX_DOCUMENTS must be a non-negative integer, got {v}"
            )
        return v

    @field_validator("retrieval_confidence_threshold")
    @classmethod
    def _validate_retrieval_confidence_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(
                f"RETRIEVAL_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError(f"TEMPERATURE must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"MAX_TOKENS must be a positive integer, got {v}")
        return v

    @field_validator("request_timeout")
    @classmethod
    def _validate_request_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"REQUEST_TIMEOUT must be a positive integer (seconds), got {v}"
            )
        return v

    @field_validator("reranker_model")
    @classmethod
    def _strip_reranker_model(cls, v: str) -> str:
        return v.strip()

    @field_validator("reranker_candidate_limit")
    @classmethod
    def _validate_reranker_candidate_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"RERANKER_CANDIDATE_LIMIT must be a positive integer, got {v}"
            )
        return v

    @field_validator("reranker_final_limit")
    @classmethod
    def _validate_reranker_final_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"RERANKER_FINAL_LIMIT must be a positive integer, got {v}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in VALID_LOG_LEVELS:
            raise ValueError(
                f"LOG_LEVEL must be one of {', '.join(sorted(VALID_LOG_LEVELS))}, got {v!r}"
            )
        return upper


# ---------------------------------------------------------------------------
# Cached accessor (usable from tests)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (singleton per interpreter).

    Use ``clear_settings_cache()`` in tests to force a re-read.
    """
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings so the next ``get_settings()`` re-reads ``.env``."""
    get_settings.cache_clear()

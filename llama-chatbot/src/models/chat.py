"""Pydantic models for chat messages, conversations, and provider requests."""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class ChatRole(enum.StrEnum):
    """Role of a message participant in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class SourceRef(BaseModel):
    """A structured source reference attached to an assistant message.

    Attributes:
        filename: Original filename of the source document.
        page_number: Optional page number when available.
        text: The retrieved chunk text, used for the "Relevant excerpt"
            shown in the UI.
        score: Optional similarity score kept internally for RAG
            evaluation.  Never displayed to the user.
    """

    filename: str
    page_number: int | None = None
    text: str | None = None
    score: float | None = None

    @field_validator("filename")
    @classmethod
    def _filename_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("filename must not be empty")
        return stripped

    model_config = {"frozen": True}


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Who sent the message (system / user / assistant).
        content: The message body.  Whitespace is stripped and empty-only
            content is rejected at construction time.
        timestamp: UTC-aware datetime set automatically when the message is
            created.  Callers should not pass this manually.
        sources: Optional structured source references (assistant messages
            only).  ``None`` for user/system messages or when RAG returned
            no sources.
    """

    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sources: list[SourceRef] | None = None

    @field_validator("content")
    @classmethod
    def _content_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message content must not be empty or whitespace-only")
        return stripped

    # ── Model configuration ──────────────────────────────────────────────
    model_config = {"frozen": False, "validate_assignment": True}


class Conversation(BaseModel):
    """An ordered list of chat messages representing a full conversation.

    This model is provider-agnostic and contains no Streamlit or SDK logic.
    """

    messages: list[ChatMessage] = Field(default_factory=list)

    def add_message(self, message: ChatMessage) -> None:
        """Append a pre-built *message* to the conversation."""
        self.messages.append(message)

    def add_user_message(self, content: str) -> ChatMessage:
        """Create a user message with *content*, append it, and return it."""
        msg = ChatMessage(role=ChatRole.USER, content=content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(
        self, content: str, sources: list[SourceRef] | None = None
    ) -> ChatMessage:
        """Create an assistant message with *content*, append it, and return it.

        Parameters
        ----------
        content:
            The assistant message body.
        sources:
            Optional structured source references attached to the message
            (assistant messages with RAG citations).  Defaults to ``None``.
        """
        msg = ChatMessage(role=ChatRole.ASSISTANT, content=content, sources=sources)
        self.messages.append(msg)
        return msg

    def clear(self) -> None:
        """Remove all messages from the conversation."""
        self.messages.clear()


class ChatRequest(BaseModel):
    """A provider-neutral request object sent to an LLM provider.

    This object is created fresh for each chat turn so that later mutations
    to the *Conversation* do not affect in-flight requests.  The provider
    implementation is responsible for translating this into its SDK format.
    """

    messages: list[ChatMessage]
    model: str
    system_prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)

    @field_validator("model")
    @classmethod
    def _model_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("model name must not be empty")
        return stripped

    @field_validator("messages")
    @classmethod
    def _copy_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """Return a shallow copy so the request is not mutated externally."""
        return list(v)

    model_config = {"frozen": True}

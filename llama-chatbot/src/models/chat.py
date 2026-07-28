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


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Who sent the message (system / user / assistant).
        content: The message body.  Whitespace is stripped and empty-only
            content is rejected at construction time.
        timestamp: UTC-aware datetime set automatically when the message is
            created.  Callers should not pass this manually.
    """

    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

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

    def add_assistant_message(self, content: str) -> ChatMessage:
        """Create an assistant message with *content*, append it, and return it."""
        msg = ChatMessage(role=ChatRole.ASSISTANT, content=content)
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

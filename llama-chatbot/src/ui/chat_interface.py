"""Streamlit chat interface — message display and user input.

This module handles the visual rendering of the conversation and
captures user input.  It does **not** modify the ``Conversation``
directly — only ``ChatService`` owns message history.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.config import get_settings
from src.exceptions import (
    ChatbotError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
)
from src.models.chat import Conversation
from src.providers.factory import create_provider
from src.services.chat_service import ChatService

# Import sidebar session-state keys for consistency
from src.ui.sidebar import (
    CONVERSATION_KEY,
    MAX_TOKENS_KEY,
    MODEL_KEY,
    PROVIDER_KEY,
    SERVICE_KEY,
    SYSTEM_PROMPT_KEY,
    TEMPERATURE_KEY,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_chat_interface() -> None:
    """Render the main chat area: history, input, and streaming display.

    Must be called on every Streamlit rerun after ``init_session_state``
    and ``render_sidebar``.
    """
    _display_chat_history()

    # Retrieve the current ChatService (or None if provider was switched)
    chat_service: ChatService | None = st.session_state.get(SERVICE_KEY)

    # Retrieve conversation
    conversation: Conversation = st.session_state.get(CONVERSATION_KEY, Conversation())

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if not user_input:
        return

    # Ensure a ChatService exists; create one if needed
    if chat_service is None:
        try:
            chat_service = _build_chat_service()
            st.session_state[SERVICE_KEY] = chat_service
        except (ChatbotError, ConfigurationError) as exc:
            safe = getattr(exc, "safe_message", str(exc))
            st.error(f"Configuration error: {safe}")
            logger.error("Failed to create ChatService: %s", exc, exc_info=True)
            return

    # Sync system prompt from sidebar (user may have changed it since
    # the ChatService was created) without recreating the service.
    latest_system_prompt: str = st.session_state.get(
        SYSTEM_PROMPT_KEY,
        get_settings().default_system_prompt,
    )
    chat_service.system_prompt = latest_system_prompt

    # Send the message via stream_message
    _handle_user_message(chat_service, conversation, user_input)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _display_chat_history() -> None:
    """Render all messages from the conversation using ``st.chat_message``."""
    conversation: Conversation = st.session_state.get(CONVERSATION_KEY, Conversation())

    for msg in conversation.messages:
        with st.chat_message(msg.role.value):
            st.markdown(msg.content)


def _build_chat_service() -> ChatService:
    """Create a new ``ChatService`` from current session state and settings."""
    settings = get_settings()
    provider_name: str = st.session_state.get(PROVIDER_KEY, settings.llm_provider)
    model: str = st.session_state.get(MODEL_KEY, settings.groq_model)
    system_prompt: str = st.session_state.get(
        SYSTEM_PROMPT_KEY, settings.default_system_prompt
    )
    temperature: float = st.session_state.get(TEMPERATURE_KEY, settings.temperature)
    max_tokens: int = st.session_state.get(MAX_TOKENS_KEY, settings.max_tokens)

    provider = create_provider(provider_name, settings)

    return ChatService(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _handle_user_message(
    chat_service: ChatService,
    conversation: Conversation,
    content: str,
) -> None:
    """Process a user message: stream the response and update the display.

    The UI does **not** append messages to the conversation — that is
    ``ChatService``'s responsibility.
    """
    assistant_content = ""

    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

    try:
        # Stream chunks into the placeholder
        for chunk in chat_service.stream_message(conversation, content):
            assistant_content += chunk
            response_placeholder.markdown(assistant_content + "▌")

    except ConfigurationError as exc:
        st.error(exc.safe_message)
        logger.warning("Input validation error: %s", exc)
        return

    except ProviderConnectionError as exc:
        st.error(exc.safe_message)
        logger.error("Provider connection error: %s", exc, exc_info=True)
        return

    except ProviderResponseError as exc:
        st.error(exc.safe_message)
        logger.error("Provider response error: %s", exc, exc_info=True)
        return

    except ChatbotError as exc:
        st.error(exc.safe_message)
        logger.error("Chatbot error: %s", exc, exc_info=True)
        return

    # Remove the cursor indicator and show final response
    if assistant_content:
        response_placeholder.markdown(assistant_content)
    else:
        response_placeholder.markdown("*No response generated.*")

    # Re-render the full conversation so user message appears
    st.rerun()

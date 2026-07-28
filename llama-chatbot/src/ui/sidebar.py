"""Streamlit sidebar — configuration controls and provider status.

This module is the only place where Streamlit sidebar widgets are
created.  It reads from and writes to ``st.session_state``.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from src.config import get_settings
from src.models.chat import Conversation
from src.providers.factory import list_registered_providers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------
_SESSION_PREFIX = "llama_chatbot_"

CONVERSATION_KEY = f"{_SESSION_PREFIX}conversation"
PROVIDER_KEY = f"{_SESSION_PREFIX}provider"
MODEL_KEY = f"{_SESSION_PREFIX}model"
SYSTEM_PROMPT_KEY = f"{_SESSION_PREFIX}system_prompt"
TEMPERATURE_KEY = f"{_SESSION_PREFIX}temperature"
MAX_TOKENS_KEY = f"{_SESSION_PREFIX}max_tokens"
SERVICE_KEY = f"{_SESSION_PREFIX}chat_service"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise all session state keys with their default values.

    Must be called once at application startup before any other UI
    functions.
    """
    settings = get_settings()

    if CONVERSATION_KEY not in st.session_state:
        st.session_state[CONVERSATION_KEY] = Conversation()

    if PROVIDER_KEY not in st.session_state:
        st.session_state[PROVIDER_KEY] = settings.llm_provider

    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = _get_default_model(
            st.session_state[PROVIDER_KEY], settings
        )

    if SYSTEM_PROMPT_KEY not in st.session_state:
        st.session_state[SYSTEM_PROMPT_KEY] = settings.default_system_prompt

    if TEMPERATURE_KEY not in st.session_state:
        st.session_state[TEMPERATURE_KEY] = settings.temperature

    if MAX_TOKENS_KEY not in st.session_state:
        st.session_state[MAX_TOKENS_KEY] = settings.max_tokens

    if SERVICE_KEY not in st.session_state:
        st.session_state[SERVICE_KEY] = None


def render_sidebar() -> None:
    """Render the sidebar with configuration controls and status.

    This function is called on every Streamlit rerun.
    """
    settings = get_settings()
    registered = list_registered_providers()

    st.sidebar.title("Settings")

    # ── Provider selection ─────────────────────────────────────────
    provider_options = registered if registered else ["groq", "ollama"]
    current_provider = st.session_state.get(PROVIDER_KEY, settings.llm_provider)

    selected_provider = st.sidebar.selectbox(
        "LLM Provider",
        options=provider_options,
        index=_safe_index(provider_options, current_provider),
        key="sidebar_provider_select",
    )

    # Preserve conversation when switching provider
    if selected_provider != current_provider:
        st.session_state[PROVIDER_KEY] = selected_provider
        st.session_state[MODEL_KEY] = _get_default_model(selected_provider, settings)
        st.session_state[SERVICE_KEY] = None
        st.rerun()

    # ── Model ──────────────────────────────────────────────────────
    default_model = st.session_state.get(
        MODEL_KEY, _get_default_model(selected_provider, settings)
    )
    model = st.sidebar.text_input(
        "Model", value=default_model, key="sidebar_model_input"
    )
    st.session_state[MODEL_KEY] = model

    # ── System prompt ──────────────────────────────────────────────
    st.session_state[SYSTEM_PROMPT_KEY] = st.sidebar.text_area(
        "System Prompt",
        value=st.session_state.get(SYSTEM_PROMPT_KEY, settings.default_system_prompt),
        height=150,
        key="sidebar_system_prompt",
    )

    # ── Temperature ────────────────────────────────────────────────
    st.session_state[TEMPERATURE_KEY] = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get(TEMPERATURE_KEY, settings.temperature),
        step=0.1,
        key="sidebar_temperature",
    )

    # ── Max tokens ─────────────────────────────────────────────────
    st.session_state[MAX_TOKENS_KEY] = st.sidebar.number_input(
        "Max Tokens",
        min_value=1,
        max_value=32768,
        value=st.session_state.get(MAX_TOKENS_KEY, settings.max_tokens),
        step=1,
        key="sidebar_max_tokens",
    )

    st.sidebar.divider()

    # ── Clear conversation ─────────────────────────────────────────
    if st.sidebar.button(
        "Clear Conversation", type="secondary", use_container_width=True
    ):
        conversation: Conversation = st.session_state[CONVERSATION_KEY]
        conversation.clear()
        st.session_state[CONVERSATION_KEY] = Conversation()
        st.rerun()

    st.sidebar.divider()

    # ── Status section ─────────────────────────────────────────────
    st.sidebar.subheader("Configuration Status")
    _render_status_section(settings)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_default_model(provider: str, settings: Any) -> str:
    """Return the default model name for the given *provider*."""
    if provider == "groq":
        return settings.groq_model
    elif provider == "ollama":
        return settings.ollama_model
    return "llama3.1:8b"


def _safe_index(options: list[str], value: str) -> int:
    """Return the index of *value* in *options*, or 0 if not found."""
    try:
        return options.index(value)
    except ValueError:
        return 0


def _render_status_section(settings: Any) -> None:
    """Show a read-only overview of the current configuration."""
    registered = list_registered_providers()

    st.sidebar.markdown("**Provider**")
    st.sidebar.caption(st.session_state.get(PROVIDER_KEY, settings.llm_provider))

    st.sidebar.markdown("**Model**")
    st.sidebar.caption(st.session_state.get(MODEL_KEY, "—"))

    st.sidebar.markdown("**Registered Providers**")
    st.sidebar.caption(", ".join(registered) if registered else "None")

    st.sidebar.markdown("**Temperature**")
    st.sidebar.caption(str(st.session_state.get(TEMPERATURE_KEY, settings.temperature)))

    st.sidebar.markdown("**Max Tokens**")
    st.sidebar.caption(str(st.session_state.get(MAX_TOKENS_KEY, settings.max_tokens)))

    st.sidebar.markdown("**Conversation Length**")
    conversation: Conversation = st.session_state.get(CONVERSATION_KEY, Conversation())
    st.sidebar.caption(f"{len(conversation.messages)} messages")

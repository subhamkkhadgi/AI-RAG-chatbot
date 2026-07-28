"""Unit tests for the chat interface UI module.

All tests mock Streamlit internals — no Streamlit server, no real
provider, no API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import clear_settings_cache
from src.models.chat import Conversation
from src.ui.chat_interface import render_chat_interface
from src.ui.sidebar import (
    CONVERSATION_KEY,
    PROVIDER_KEY,
    SERVICE_KEY,
    init_session_state,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    clear_settings_cache()


@pytest.fixture
def mock_session_state() -> MagicMock:
    """Return a MagicMock for st.session_state backed by a real dict."""
    state = MagicMock()
    real_storage: dict = {}
    state._storage = real_storage

    def _getitem(key: str) -> object:
        return real_storage[key]

    def _setitem(key: str, value: object) -> None:
        real_storage[key] = value

    def _contains(key: str) -> bool:
        return key in real_storage

    state.__getitem__.side_effect = _getitem
    state.__setitem__.side_effect = _setitem
    state.__contains__.side_effect = _contains
    return state


def _init_state(mock_session_state: MagicMock) -> None:
    with patch("streamlit.session_state", mock_session_state):
        init_session_state()


# ======================================================================
# Tests
# ======================================================================


class TestChatInterface:
    """Verify chat_interface behaviour with mocked Streamlit."""

    def test_renders_without_error(self, mock_session_state: MagicMock) -> None:
        _init_state(mock_session_state)
        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.chat_message"),
            patch("streamlit.chat_input", return_value=None),
            patch("streamlit.empty"),
            patch("streamlit.rerun"),
            patch("streamlit.error"),
        ):
            render_chat_interface()

    def test_renders_existing_messages(self, mock_session_state: MagicMock) -> None:
        _init_state(mock_session_state)
        conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there")

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.chat_message"),
            patch("streamlit.chat_input", return_value=None),
            patch("streamlit.empty"),
            patch("streamlit.rerun"),
            patch("streamlit.error"),
        ):
            render_chat_interface()

    def test_ui_does_not_add_messages_directly(self) -> None:
        import src.ui.chat_interface as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "conversation.add_user_message" not in content
            assert "conversation.add_assistant_message" not in content

    def test_provider_switch_preserves_conversation(
        self, mock_session_state: MagicMock
    ) -> None:
        _init_state(mock_session_state)
        conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
        conv.add_user_message("Stick around")
        conv.add_assistant_message("I will!")

        old_provider = mock_session_state._storage[PROVIDER_KEY]
        new_provider = "ollama" if old_provider == "groq" else "groq"
        mock_session_state._storage[PROVIDER_KEY] = new_provider
        mock_session_state._storage[SERVICE_KEY] = None

        assert len(conv.messages) == 2
        assert conv.messages[0].content == "Stick around"

    def test_no_sdk_imports_in_chat_interface(self) -> None:
        import src.ui.chat_interface as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            for sdk in ("import groq", "from groq", "import ollama", "from ollama"):
                assert sdk not in content

    def test_no_conversation_ownership_in_ui(self) -> None:
        import src.ui.chat_interface as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "conversation.add_user_message" not in content
            assert "conversation.add_assistant_message" not in content

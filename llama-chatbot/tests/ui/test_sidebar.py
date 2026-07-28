"""Unit tests for the sidebar UI module.

All tests mock Streamlit internals — no Streamlit server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import clear_settings_cache
from src.models.chat import Conversation
from src.ui.sidebar import (
    CONVERSATION_KEY,
    MAX_TOKENS_KEY,
    MODEL_KEY,
    PROVIDER_KEY,
    SERVICE_KEY,
    SYSTEM_PROMPT_KEY,
    TEMPERATURE_KEY,
    init_session_state,
    render_sidebar,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Ensure a fresh Settings instance for each test."""
    clear_settings_cache()


@pytest.fixture(autouse=True)
def _deterministic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override selected env vars so Settings returns deterministic values
    regardless of the developer's local ``.env`` file.

    Environment variables take precedence over ``.env`` in Pydantic
    Settings v2, so this fixture isolates tests from local configuration.
    """
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("MAX_TOKENS", "2048")
    monkeypatch.setenv("TEMPERATURE", "0.7")
    # Clear the cached Settings instance so the next get_settings()
    # call re-reads environment (now using our patched values).
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


# ======================================================================
# Tests: init_session_state
# ======================================================================


class TestInitSessionState:
    """Verify session state keys are created with default values."""

    def test_creates_conversation(self, mock_session_state: MagicMock) -> None:
        """init_session_state should create a Conversation instance."""
        with patch("streamlit.session_state", mock_session_state):
            init_session_state()

        assert CONVERSATION_KEY in mock_session_state._storage
        conv = mock_session_state._storage[CONVERSATION_KEY]
        assert isinstance(conv, Conversation)

    def test_sets_provider_from_settings(self, mock_session_state: MagicMock) -> None:
        """The provider should be initialised from Settings."""
        with patch("streamlit.session_state", mock_session_state):
            init_session_state()

        assert mock_session_state._storage.get(PROVIDER_KEY) == "groq"

    def test_sets_model_from_provider_default(
        self, mock_session_state: MagicMock
    ) -> None:
        """Model should be set based on the provider default."""
        with patch("streamlit.session_state", mock_session_state):
            init_session_state()

        assert MODEL_KEY in mock_session_state._storage

    def test_sets_system_prompt(self, mock_session_state: MagicMock) -> None:
        """System prompt should come from default_system_prompt."""
        with patch("streamlit.session_state", mock_session_state):
            init_session_state()

        prompt = mock_session_state._storage.get(SYSTEM_PROMPT_KEY, "")
        assert len(prompt) > 0

    def test_sets_temperature(self, mock_session_state: MagicMock) -> None:
        """Temperature should be initialised from settings."""
        with patch("streamlit.session_state", mock_session_state):
            init_session_state()

        assert mock_session_state._storage.get(TEMPERATURE_KEY) == 0.7

    def test_sets_max_tokens(self, mock_session_state: MagicMock) -> None:
        """Max tokens should be initialised from settings."""
        with patch("streamlit.session_state", mock_session_state):
            init_session_state()

        assert mock_session_state._storage.get(MAX_TOKENS_KEY) == 2048

    def test_sets_service_to_none(self, mock_session_state: MagicMock) -> None:
        """ChatService should start as None."""
        with patch("streamlit.session_state", mock_session_state):
            init_session_state()

        assert mock_session_state._storage.get(SERVICE_KEY) is None

    def test_does_not_overwrite_existing_keys(
        self, mock_session_state: MagicMock
    ) -> None:
        """Existing session state keys should not be overwritten."""
        with patch("streamlit.session_state", mock_session_state):
            mock_session_state["my_key"] = "preserved"
            init_session_state()

        assert mock_session_state._storage["my_key"] == "preserved"


# ======================================================================
# Tests: render_sidebar
# ======================================================================


class TestRenderSidebar:
    """Verify render_sidebar runs without errors (mocked Streamlit)."""

    def _patch_sidebar_widgets(self) -> dict:
        """Return a dict of patches for all sidebar widgets to avoid
        Streamlit API validation errors."""
        return {
            "streamlit.sidebar.title": MagicMock(),
            "streamlit.sidebar.selectbox": MagicMock(return_value="groq"),
            "streamlit.sidebar.text_input": MagicMock(return_value="llama3.1:8b"),
            "streamlit.sidebar.text_area": MagicMock(
                return_value="You are a helpful assistant."
            ),
            "streamlit.sidebar.slider": MagicMock(return_value=0.7),
            "streamlit.sidebar.number_input": MagicMock(return_value=2048),
            "streamlit.sidebar.divider": MagicMock(),
            "streamlit.sidebar.button": MagicMock(return_value=False),
            "streamlit.sidebar.subheader": MagicMock(),
            "streamlit.sidebar.markdown": MagicMock(),
            "streamlit.sidebar.caption": MagicMock(),
            "streamlit.rerun": MagicMock(),
        }

    def test_render_with_default_state(self, mock_session_state: MagicMock) -> None:
        """render_sidebar should not raise with freshly initialised state."""
        patches = self._patch_sidebar_widgets()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch(
                "src.ui.sidebar.list_registered_providers",
                return_value=["groq", "ollama"],
            ),
        ):
            for target, mock_val in patches.items():
                patcher = patch(target, mock_val)
                patcher.start()

            try:
                init_session_state()
                render_sidebar()
            finally:
                for patcher in patches:
                    patch(target, mock_val).stop()

        # Basic smoke test — no exception means widgets were reached
        assert True

    def test_render_without_registered_providers(
        self, mock_session_state: MagicMock
    ) -> None:
        """render_sidebar should work even when no providers are registered."""
        patches = self._patch_sidebar_widgets()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("src.ui.sidebar.list_registered_providers", return_value=[]),
        ):
            for target, mock_val in patches.items():
                patcher = patch(target, mock_val)
                patcher.start()
            try:
                init_session_state()
                render_sidebar()
            finally:
                for patcher in patches:
                    patch(target, mock_val).stop()

        assert True

    def test_clear_button_creates_new_conversation(
        self, mock_session_state: MagicMock
    ) -> None:
        """Pressing 'Clear Conversation' should reset the conversation."""
        patches = self._patch_sidebar_widgets()
        # Simulate button press (return True)
        patches["streamlit.sidebar.button"] = MagicMock(return_value=True)

        with (
            patch("streamlit.session_state", mock_session_state),
            patch(
                "src.ui.sidebar.list_registered_providers",
                return_value=["groq", "ollama"],
            ),
        ):
            for target, mock_val in patches.items():
                patcher = patch(target, mock_val)
                patcher.start()
            try:
                init_session_state()
                # Add messages before clear
                conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
                conv.add_user_message("Hello")
                conv.add_assistant_message("Hi")
                assert len(conv.messages) == 2

                render_sidebar()
            finally:
                for patcher in patches:
                    patch(target, mock_val).stop()

            # Should be a fresh conversation now
            new_conv = mock_session_state._storage[CONVERSATION_KEY]
            assert len(new_conv.messages) == 0

    def test_provider_switch_clears_service(
        self, mock_session_state: MagicMock
    ) -> None:
        """Switching provider should set SERVICE_KEY to None."""
        patches = self._patch_sidebar_widgets()
        # Simulate selecting "ollama" (different from current "groq")
        patches["streamlit.sidebar.selectbox"] = MagicMock(return_value="ollama")

        with (
            patch("streamlit.session_state", mock_session_state),
            patch(
                "src.ui.sidebar.list_registered_providers",
                return_value=["groq", "ollama"],
            ),
        ):
            for target, mock_val in patches.items():
                patcher = patch(target, mock_val)
                patcher.start()
            try:
                init_session_state()
                mock_session_state._storage[PROVIDER_KEY] = "groq"
                mock_session_state._storage[SERVICE_KEY] = MagicMock()

                render_sidebar()
            finally:
                for patcher in patches:
                    patch(target, mock_val).stop()

            # Provider should have been switched and service cleared
            assert mock_session_state._storage[PROVIDER_KEY] == "ollama"
            assert mock_session_state._storage[SERVICE_KEY] is None

    def test_no_sdk_imports_in_sidebar(self) -> None:
        """sidebar.py must not import groq or ollama SDKs."""
        import src.ui.sidebar as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            for sdk in ("import groq", "from groq", "import ollama", "from ollama"):
                assert sdk not in content

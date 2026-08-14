"""Unit tests for the chat interface UI module.

All tests mock Streamlit internals — no Streamlit server, no real
provider, no API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import clear_settings_cache, get_settings
from src.models.chat import Conversation, SourceRef
from src.providers.base import BaseLLMProvider
from src.ui.chat_interface import render_chat_interface, _build_chat_service
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

    def _get(key: str, default: object = None) -> object:
        return real_storage.get(key, default)

    state.__getitem__.side_effect = _getitem
    state.__setitem__.side_effect = _setitem
    state.__contains__.side_effect = _contains
    state.get.side_effect = _get
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


# ======================================================================
# Source Rendering Tests (Sprint 8D)
# ======================================================================

class TestSourceRendering:
    """Verify the expandable Sources cards render correctly."""

    def test_sources_expander_rendered(self, mock_session_state: MagicMock) -> None:
        """Assistant messages with sources should render an expander."""
        _init_state(mock_session_state)
        conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
        conv.add_assistant_message(
            "Answer text.",
            sources=[
                SourceRef(
                    filename="report.pdf",
                    page_number=3,
                    text="Relevant excerpt.",
                )
            ],
        )

        expander_mock = MagicMock()
        expander_ctx = MagicMock()
        expander_mock.return_value = expander_ctx
        markdown_mock = MagicMock()
        caption_mock = MagicMock()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.chat_message"),
            patch("streamlit.chat_input", return_value=None),
            patch("streamlit.empty"),
            patch("streamlit.rerun"),
            patch("streamlit.error"),
            patch("streamlit.expander", expander_mock),
            patch("streamlit.markdown", markdown_mock),
            patch("streamlit.caption", caption_mock),
        ):
            render_chat_interface()

        # The label is passed to st.expander(...), not to __enter__().
        expander_mock.assert_called_once()
        label = expander_mock.call_args[0][0]
        assert "📚" in label
        assert "1 Source" in label
        assert "Sources" not in label

        # The context manager should be entered.
        expander_ctx.__enter__.assert_called_once()

        # The markdown should include the filename.
        md_args = [c.args[0] for c in markdown_mock.call_args_list]
        assert any("report.pdf" in m for m in md_args)

        # The caption should include the page number.
        cap_args = [c.args[0] for c in caption_mock.call_args_list]
        assert any("Page 3" in m for m in cap_args)

    def test_no_sources_no_expander(self, mock_session_state: MagicMock) -> None:
        """Assistant messages without sources should not render an expander."""
        _init_state(mock_session_state)
        conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
        conv.add_assistant_message("Plain answer.")

        expander_mock = MagicMock()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.chat_message"),
            patch("streamlit.chat_input", return_value=None),
            patch("streamlit.empty"),
            patch("streamlit.rerun"),
            patch("streamlit.error"),
            patch("streamlit.expander", return_value=expander_mock),
        ):
            render_chat_interface()

        expander_mock.assert_not_called()

    def test_multiple_sources_shown(self, mock_session_state: MagicMock) -> None:
        """Multiple sources should each be rendered with dividers."""
        _init_state(mock_session_state)
        conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
        conv.add_assistant_message(
            "Answer.",
            sources=[
                SourceRef(filename="a.pdf", page_number=1, text="A."),
                SourceRef(filename="b.pdf", page_number=2, text="B."),
            ],
        )

        expander_mock = MagicMock()
        markdown_mock = MagicMock()
        divider_mock = MagicMock()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.chat_message"),
            patch("streamlit.chat_input", return_value=None),
            patch("streamlit.empty"),
            patch("streamlit.rerun"),
            patch("streamlit.error"),
            patch("streamlit.expander", return_value=expander_mock),
            patch("streamlit.markdown", markdown_mock),
            patch("streamlit.divider", divider_mock),
        ):
            render_chat_interface()

        md_args = [c.args[0] for c in markdown_mock.call_args_list]
        assert any("a.pdf" in m for m in md_args)
        assert any("b.pdf" in m for m in md_args)

        # Divider between two sources.
        assert divider_mock.call_count >= 1

    def test_multiple_sources_plural_label(
        self, mock_session_state: MagicMock
    ) -> None:
        """The Sources label uses the plural form for more than one source."""
        _init_state(mock_session_state)
        conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
        conv.add_assistant_message(
            "Answer.",
            sources=[
                SourceRef(filename="a.pdf", page_number=1, text="A."),
                SourceRef(filename="b.pdf", page_number=2, text="B."),
            ],
        )

        expander_mock = MagicMock()
        expander_ctx = MagicMock()
        expander_mock.return_value = expander_ctx
        markdown_mock = MagicMock()
        caption_mock = MagicMock()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.chat_message"),
            patch("streamlit.chat_input", return_value=None),
            patch("streamlit.empty"),
            patch("streamlit.rerun"),
            patch("streamlit.error"),
            patch("streamlit.expander", expander_mock),
            patch("streamlit.markdown", markdown_mock),
            patch("streamlit.caption", caption_mock),
        ):
            render_chat_interface()

        expander_mock.assert_called_once()
        label = expander_mock.call_args[0][0]
        assert "2 Sources" in label

    def test_internal_metadata_not_rendered(
        self, mock_session_state: MagicMock
    ) -> None:
        """document_id / chunk_index / score must not appear in the UI."""
        _init_state(mock_session_state)
        conv: Conversation = mock_session_state._storage[CONVERSATION_KEY]
        conv.add_assistant_message(
            "Answer.",
            sources=[
                SourceRef(
                    filename="report.pdf",
                    page_number=3,
                    text="Excerpt.",
                    score=0.987,
                )
            ],
        )

        markdown_mock = MagicMock()
        caption_mock = MagicMock()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.chat_message"),
            patch("streamlit.chat_input", return_value=None),
            patch("streamlit.empty"),
            patch("streamlit.rerun"),
            patch("streamlit.error"),
            patch("streamlit.expander"),
            patch("streamlit.markdown", markdown_mock),
            patch("streamlit.caption", caption_mock),
        ):
            render_chat_interface()

        md_args = [c.args[0] for c in markdown_mock.call_args_list]
        cap_args = [c.args[0] for c in caption_mock.call_args_list]
        all_text = " ".join(md_args + cap_args)

        assert "0.987" not in all_text
        assert "score" not in all_text
        assert "document_id" not in all_text
        assert "chunk_index" not in all_text


# ======================================================================
# Confidence Threshold Wiring (Sprint 9B.1)
# ======================================================================

class TestConfidenceThresholdWiring:
    """Verify the configured retrieval confidence threshold is passed into
    ChatService through the application construction path."""

    def test_build_chat_service_passes_configured_threshold(
        self, mock_session_state: MagicMock
    ) -> None:
        """The constructed ChatService should receive the configured
        retrieval_confidence_threshold from settings."""
        _init_state(mock_session_state)
        settings = get_settings()
        expected = settings.retrieval_confidence_threshold

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("src.ui.chat_interface.create_provider") as provider_mock,
            patch("src.ui.chat_interface.create_embedding_provider"),
            patch("src.ui.chat_interface.create_vector_store"),
        ):
            provider_mock.return_value = MagicMock(spec=BaseLLMProvider)
            service = _build_chat_service()

        assert service is not None
        assert service.confidence_threshold == expected

    def test_confidence_threshold_is_not_none_when_configured(
        self, mock_session_state: MagicMock
    ) -> None:
        """When retrieval_confidence_threshold is configured, the
        ChatService should not have a None threshold (i.e. gating active)."""
        _init_state(mock_session_state)
        settings = get_settings()
        if settings.retrieval_confidence_threshold is None:
            pytest.skip("retrieval_confidence_threshold not configured")

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("src.ui.chat_interface.create_provider") as provider_mock,
            patch("src.ui.chat_interface.create_embedding_provider"),
            patch("src.ui.chat_interface.create_vector_store"),
        ):
            provider_mock.return_value = MagicMock(spec=BaseLLMProvider)
            service = _build_chat_service()

        assert service.confidence_threshold is not None
        assert service.confidence_threshold == settings.retrieval_confidence_threshold

"""llama-chatbot — Streamlit entry point.

This module initialises session state, renders the sidebar, and
displays the chat interface.  It is intentionally thin — all
business logic lives in ``ChatService`` and the provider layer.
"""

from __future__ import annotations

import streamlit as st

from src.config import get_settings
from src.logging_config import setup_logging
from src.ui.chat_interface import render_chat_interface
from src.ui.sidebar import init_session_state, render_sidebar

# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Llama Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Run the Streamlit application."""
    # ── Bootstrap ─────────────────────────────────────────────────────
    settings = get_settings()
    setup_logging(settings.log_level)

    # ── Session state ────────────────────────────────────────────────
    init_session_state()

    # ── UI ───────────────────────────────────────────────────────────
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()

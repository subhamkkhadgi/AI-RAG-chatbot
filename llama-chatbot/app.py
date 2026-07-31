"""llama-chatbot — Streamlit entry point.

This module initialises session state, renders the sidebar, and
displays the chat interface.  It is intentionally thin — all
business logic lives in ``ChatService`` and the provider layer.
"""

from __future__ import annotations

import logging 
import tempfile
from pathlib import Path

import streamlit as st

from src.config import get_settings
from src.documents import DocumentIngestionService, DocumentManager
from src.embeddings.factory import create_embedding_provider
from src.exceptions import ChatbotError
from src.logging_config import setup_logging
from src.ui.chat_interface import render_chat_interface
from src.ui.sidebar import init_session_state, render_sidebar
from src.vectorstores.factory import create_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------
_INGESTION_SERVICE_KEY = "llama_chatbot_ingestion_service"
_DOCUMENT_MANAGER_KEY = "llama_chatbot_document_manager"

# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Llama Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_ingestion_service() -> DocumentIngestionService | None:
    """Return the cached ``DocumentIngestionService``, creating it once."""
    service: DocumentIngestionService | None = st.session_state.get(
        _INGESTION_SERVICE_KEY
    )
    if service is not None:
        return service
    try:
        settings = get_settings()
        embedding_provider = create_embedding_provider(
            settings.embedding_provider, settings
        )
        vector_store = create_vector_store("qdrant", settings)
        service = DocumentIngestionService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            collection_name=settings.qdrant_collection,
        )
        st.session_state[_INGESTION_SERVICE_KEY] = service
        return service
    except ChatbotError as exc:
        logger.error("Failed to create DocumentIngestionService: %s", exc)
        return None


def _render_document_upload() -> None:
    """Render the document upload section in the sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("Document Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a document",
        type=["pdf", "txt"],
        key="sidebar_doc_uploader",
    )

    if uploaded_file is None:
        # Clear any ingestion tracking when no file is present
        st.session_state.pop("_llama_ingested_file", None)
        return

    # ── Rerun guard ────────────────────────────────────────────────────────
    # Prevent ingest_file() from being called repeatedly on Streamlit reruns
    # for the same uploaded file.
    file_id = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("_llama_ingested_file") == file_id:
        return

    # Save uploaded file to a temporary location
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        service = _get_ingestion_service()
        if service is None:
            st.sidebar.error(
                "Ingestion service is unavailable. Check the logs for details."
            )
            return

        with st.sidebar.status("Ingesting document...", expanded=False) as status:
            status.write(f"Processing **{uploaded_file.name}**...")
            document_id = service.ingest_file(tmp_path, original_filename=uploaded_file.name)

        st.sidebar.success(
            f"Document ingested successfully.\n\n"
            f"**File:** {uploaded_file.name}\n"
            f"**Document ID:** `{document_id}`"
        )
        logger.info(
            "Ingested '%s' (document_id=%s)", uploaded_file.name, document_id
        )
        # Record ingestion so the rerun guard blocks re-ingestion
        st.session_state["_llama_ingested_file"] = file_id
    except ChatbotError as exc:
        safe = getattr(exc, "safe_message", str(exc))
        st.sidebar.error(f"Ingestion failed: {safe}")
        logger.error(
            "Ingestion failed for '%s': %s", uploaded_file.name, exc
        )
    except Exception as exc:
        st.sidebar.error(f"Unexpected error during ingestion: {exc}")
        logger.error(
            "Unexpected ingestion error for '%s': %s",
            uploaded_file.name,
            exc,
            exc_info=True,
        )
    finally:
        # Clean up the temporary file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def _get_document_manager() -> DocumentManager | None:
    """Return the cached ``DocumentManager``, creating it once."""
    manager: DocumentManager | None = st.session_state.get(_DOCUMENT_MANAGER_KEY)
    if manager is not None:
        return manager
    try:
        settings = get_settings()
        vector_store = create_vector_store("qdrant", settings)
        manager = DocumentManager(vector_store=vector_store)
        st.session_state[_DOCUMENT_MANAGER_KEY] = manager
        return manager
    except ChatbotError as exc:
        logger.error("Failed to create DocumentManager: %s", exc)
        return None


def _render_document_management() -> None:
    """Render the document management section in the sidebar.

    Shows a table of uploaded documents with a delete button per row.
    """
    manager = _get_document_manager()
    if manager is None:
        st.sidebar.error("Document manager is unavailable.")
        return

    st.sidebar.divider()
    st.sidebar.subheader("Uploaded Documents")

    # Fetch document list
    try:
        documents = manager.list_documents()
    except ChatbotError as exc:
        safe = getattr(exc, "safe_message", str(exc))
        st.sidebar.error(f"Failed to list documents: {safe}")
        return

    if not documents:
        st.sidebar.caption("No documents uploaded yet.")
        return

    # Display each document with a delete button
    for doc in documents:
        filename = doc.get("filename", "unknown")
        doc_id = doc.get("document_id", "")
        chunk_count = doc.get("chunk_count", 0)

        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.markdown(f"**{filename}**")
            st.caption(f"ID: `{doc_id[:8]}...` | {chunk_count} chunk(s)")
        with col2:
            if st.button("🗑️", key=f"delete_{doc_id}", help="Delete this document"):
                try:
                    deleted = manager.delete_document(doc_id)
                    st.sidebar.success(
                        f"Deleted **{filename}** ({deleted} chunk(s))."
                    )
                    logger.info(
                        "Deleted document '%s' (%s) — %d chunk(s).",
                        filename,
                        doc_id,
                        deleted,
                    )
                    st.rerun()
                except ChatbotError as exc:
                    safe = getattr(exc, "safe_message", str(exc))
                    st.sidebar.error(f"Delete failed: {safe}")


def main() -> None:
    """Run the Streamlit application."""
    # -- Bootstrap ----------------------------------------------------------
    settings = get_settings()
    setup_logging(settings.log_level)

    # -- Session state ------------------------------------------------------
    init_session_state()

    # -- UI -----------------------------------------------------------------
    render_sidebar()
    _render_document_upload()
    _render_document_management()
    render_chat_interface()


if __name__ == "__main__":
    main()


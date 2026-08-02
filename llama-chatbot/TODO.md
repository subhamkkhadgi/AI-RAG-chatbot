# Sprint 8B — Source Citation Integration

## Implementation Order

- [x] Step 1: Inspection of retrieval, context-building, and chat response flow
- [x] Step 2: Create `src/rag/citations.py` — provider-neutral citation formatter
- [x] Step 3: Extend `ChatService` to capture retrieval metadata and append citations
- [x] Step 4: Add citation-related tests in `tests/rag/test_citations.py`
- [x] Step 5: Add citation-related tests in `tests/services/test_chat_service.py`
- [x] Step 6: Run `python -m pytest -v` and verify regression status

# Sprint 8D — Citation UI Enhancement

## Implementation Order

- [x] Step 1: Inspection of chat response flow and UI rendering
- [x] Step 2: Add `SourceRef` model + optional `sources` field on `ChatMessage`
- [x] Step 3: Add `build_source_refs()` structured builder in `src/rag/citations.py`
- [x] Step 4: Extend `ChatService` to attach structured `sources` to assistant messages
- [x] Step 5: Update `_display_chat_history()` to render expandable `📚 Sources` cards
- [x] Step 6: Add tests (structured sources, UI rendering, backward compat, empty handling)
- [x] Step 7: Run `python -m pytest -v` and verify regression status



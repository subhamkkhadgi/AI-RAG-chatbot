# Implementation Plan: Contextual Follow-up Retrieval

## Steps
- [ ] 1. Inspect current `_enrich_with_context` and call sites (done in prior investigation).
- [ ] 2. Add pure helper `build_retrieval_query(content, conversation)` in `chat_service.py`.
- [ ] 3. Thread `conversation` into `_enrich_with_context` from `send_message` and `stream_message`.
- [ ] 4. Use the helper to build the retrieval query; keep stored user message unchanged.
- [ ] 5. Add focused tests in `tests/services/test_chat_service.py`.
- [ ] 6. Run targeted ChatService tests.
- [ ] 7. Run the complete test suite.

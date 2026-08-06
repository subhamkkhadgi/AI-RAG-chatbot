# Answer-Aware Citation Filtering — Task Tracking

## Steps
- [x] Create `src/rag/evidence.py` with lexical supporting-chunk filtering
- [x] Modify `src/services/chat_service.py` to filter chunks before building citations
- [x] Add `tests/rag/test_evidence.py`
- [x] Update existing citation tests in `tests/services/test_chat_service.py`
- [x] Fix stopword filtering so common words don't inflate overlap
- [x] Fix scoring direction to answer coverage (chunk rejected was valid)
- [x] Add answer-coverage tests (long supporting chunk retained, unsupported page removed)
- [x] Run `python -m pytest -q` — 317 passed

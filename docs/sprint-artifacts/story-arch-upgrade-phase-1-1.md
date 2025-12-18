# Story 1.1: Implement Weaviate Vector Store Service

**Status:** Draft

---

## User Story

As a **Backend Developer**,
I want **to replace the local FAISS vector store with a Weaviate client**,
So that **the application has a persistent, scalable knowledge base that supports hybrid search**.

---

## Acceptance Criteria

**Given** a configured Weaviate instance
**When** the application starts
**Then** it should connect to Weaviate without errors

**And** the `VectorStoreService` should expose methods for:
1.  `add_documents(chunks)`
2.  `hybrid_search(query, alpha)`
3.  `delete_collection()`

**And** the schema should include fields for `text`, `header_path`, and `source`.

---

## Implementation Details

### Tasks / Subtasks

- [ ] (AC: #1) Add `weaviate-client` and `langchain-weaviate` to `requirements.txt`.
- [ ] (AC: #1) Update `app/core/config.py` to include `WEAVIATE_URL` and `WEAVIATE_API_KEY`.
- [ ] (AC: #2) Create `app/services/vector_store.py` (replacing existing FAISS logic).
- [ ] (AC: #2) Implement `__init__` to establish Weaviate connection.
- [ ] (AC: #2) Implement `ensure_schema()` to define the class properties.
- [ ] (AC: #2) Implement `add_documents()` to batch upload chunks.
- [ ] (AC: #2) Implement `hybrid_search()` using Weaviate's `hybrid` operator.
- [ ] (AC: #1) Create unit tests for the new service (mocking Weaviate).

### Technical Summary

Replace the file-based FAISS implementation with a robust Weaviate service. This involves defining a schema that supports the new metadata fields (`header_path`) required for Phase 1.

### Project Structure Notes

- **Files to modify:**
    - `requirements.txt`
    - `app/core/config.py`
    - `app/services/vector_store.py`
- **Expected test locations:** `tests/test_vector_store.py`
- **Estimated effort:** 3 story points (1-2 days)
- **Prerequisites:** None

### Key Code References

- `app/services/vector_store.py` (Existing FAISS implementation to be removed/replaced)
- `app/core/config.py`

---

## Context References

**Tech-Spec:** [tech-spec.md](../tech-spec.md) - Primary context document containing:

- Brownfield codebase analysis (if applicable)
- Framework and library details with versions
- Existing patterns to follow
- Integration points and dependencies
- Complete implementation guidance

**Architecture:** See `docs/implementation-strategy-mcl-agent.md`

---

## Dev Agent Record

### Agent Model Used

<!-- Will be populated during dev-story execution -->

### Debug Log References

<!-- Will be populated during dev-story execution -->

### Completion Notes

<!-- Will be populated during dev-story execution -->

### Files Modified

<!-- Will be populated during dev-story execution -->

### Test Results

<!-- Will be populated during dev-story execution -->

---

## Review Notes

<!-- Will be populated during code review -->

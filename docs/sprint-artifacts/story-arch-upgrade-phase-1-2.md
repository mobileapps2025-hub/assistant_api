# Story 1.2: Implement Structure-Aware Ingestion Service

**Status:** Draft

---

## User Story

As a **Backend Developer**,
I want **to ingest Markdown files using header-based splitting**,
So that **retrieved chunks retain their semantic context (e.g., which section they belong to)**.

---

## Acceptance Criteria

**Given** a Markdown file with headers (#, ##, ###)
**When** the ingestion process runs
**Then** the file should be split into chunks based on these headers

**And** each chunk must contain a `header_path` metadata field (e.g., "Title > Section > Subsection")
**And** the chunks must be uploaded to the Weaviate vector store
**And** an Admin API endpoint `POST /api/admin/ingest` should trigger this process.

---

## Implementation Details

### Tasks / Subtasks

- [ ] (AC: #1) Create `app/services/ingestion_service.py`.
- [ ] (AC: #1) Implement `load_and_split_document` using `MarkdownHeaderTextSplitter`.
- [ ] (AC: #2) Implement `ingest_all` method to iterate `app/documents/` and call `vector_store.add_documents`.
- [ ] (AC: #3) Create `app/routers/admin.py`.
- [ ] (AC: #3) Define `POST /ingest` endpoint protected by a simple secret or open for now (dev context).
- [ ] (AC: #3) Register `admin` router in `app/main.py`.
- [ ] (AC: #1) Add unit tests for splitting logic.

### Technical Summary

This story moves away from naive character-based chunking to structure-aware chunking. It also introduces an explicit ingestion trigger (API) rather than running it on app startup, solving the "slow startup" issue.

### Project Structure Notes

- **Files to modify:**
    - `app/services/ingestion_service.py` (New)
    - `app/routers/admin.py` (New)
    - `app/main.py`
- **Expected test locations:** `tests/test_ingestion.py`
- **Estimated effort:** 3 story points (1-2 days)
- **Prerequisites:** Story 1.1 (Weaviate Service)

### Key Code References

- `app/services/vector_store.py` (New Weaviate service)
- `app/documents/` (Source files)

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

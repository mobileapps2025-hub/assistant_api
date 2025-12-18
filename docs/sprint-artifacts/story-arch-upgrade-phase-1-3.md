# Story 1.3: Upgrade Chat Service with Hybrid Retrieval

**Status:** Draft

---

## User Story

As a **Product Manager**,
I want **the chat bot to use Hybrid Search and Re-ranking**,
So that **users get accurate answers for both conceptual questions and specific technical queries**.

---

## Acceptance Criteria

**Given** a user query
**When** the system retrieves documents
**Then** it should use Hybrid Search (Alpha=0.5) to combine Vector and Keyword results

**And** re-rank the top 25 results using Cohere Rerank
**And** only return results with a re-rank score > 0.7
**And** pass these high-quality chunks to the LLM for generation.

---

## Implementation Details

### Tasks / Subtasks

- [ ] (AC: #2) Add `cohere` to `requirements.txt`.
- [ ] (AC: #2) Update `app/core/config.py` with `COHERE_API_KEY`.
- [ ] (AC: #1) Modify `app/services/chat_service.py`.
- [ ] (AC: #1) Replace `vector_store.search` with `vector_store.hybrid_search`.
- [ ] (AC: #2) Implement `rerank_results` method using Cohere client.
- [ ] (AC: #3) Apply filtering logic (score > 0.7).
- [ ] (AC: #1) Update prompt construction to use the new chunk format (including header paths).

### Technical Summary

This is the final piece of Phase 1, connecting the new data pipeline to the user-facing chat. It introduces the "Re-ranking" step which is the single biggest factor in reducing hallucinations.

### Project Structure Notes

- **Files to modify:**
    - `requirements.txt`
    - `app/core/config.py`
    - `app/services/chat_service.py`
- **Expected test locations:** `tests/test_chat_service.py`
- **Estimated effort:** 2 story points (1 day)
- **Prerequisites:** Story 1.1, 1.2

### Key Code References

- `app/services/chat_service.py`
- `app/services/vector_store.py`

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

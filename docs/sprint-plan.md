# Sprint Plan: MCL AI Agent 2.0 Refactor

## Sprint Goal
Refactor the MCL AI Agent to a modular, stateless architecture with persistent vector storage and unified vision capabilities, reducing startup time to < 5 seconds.

## Schedule
- **Start Date**: 2025-11-27
- **Duration**: 1 Sprint

## Selected Stories

### Priority 1: Core Infrastructure (Epic 3 & 1)
These stories set the foundation for the new architecture.
- [ ] **Story 3.1**: Project Restructuring (Create `app/services`, `app/core`, etc.)
- [ ] **Story 3.2**: Structured Logging (Replace `print` with `logging`)
- [x] **Story 1.1**: VectorStoreService Skeleton & File Handling
- [ ] **Story 1.2**: Implement Index Building & Persistence
- [ ] **Story 1.3**: Implement Index Loading
- [ ] **Story 1.4**: Semantic Search Implementation

### Priority 2: Vision Services (Epic 2)
Consolidating the vision logic.
- [ ] **Story 2.1**: VisionService Implementation (GPT-4o Chat Completion)
- [ ] **Story 2.2**: Port Image Validation
- [ ] **Story 2.3**: Refactor Vision Endpoint

### Priority 3: Integration & Cleanup (Epic 3)
Wiring it all together.
- [ ] **Story 3.3**: Dependency Injection & Main Refactor
- [ ] **Story 3.4**: Cleanup Legacy Code (`vision_assistant.py`, old `services.py`)

## Risks
- **Dependency Issues**: Need to ensure `faiss-cpu` and `sentence-transformers` install correctly on the target environment.
- **Migration**: Existing feedback data is in SQLite; this refactor touches the code structure but should preserve the database file.

## Definition of Done
- All code is in the new directory structure.
- Application starts in < 5 seconds.
- `/api/chat` and `/api/vision/analyze-screenshot` endpoints work as expected.
- No `print()` statements in application logic.

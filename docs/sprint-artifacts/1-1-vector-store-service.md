# Story 1.1: VectorStoreService Skeleton & File Handling

**Epic**: Persistent Vector Store
**Status**: review

## Description
Create the `VectorStoreService` class in `app/services/vector_store.py`. Implement logic to define storage paths for the index and metadata. This service will be responsible for managing the FAISS index and document metadata, decoupling it from the application startup.

## Acceptance Criteria
- [ ] Class `VectorStoreService` exists in `app/services/vector_store.py`
- [ ] Class initializes with a `storage_path` parameter
- [ ] Class defines paths for `faiss_index.bin` and `chunk_metadata.json`
- [ ] Method `index_exists()` returns True if both files exist on disk, False otherwise
- [ ] Unit tests verify initialization and file existence checks

## Tasks/Subtasks
- [x] Create directory `app/services` if it doesn't exist
- [x] Create `app/services/__init__.py`
- [x] Create `app/services/vector_store.py`
- [x] Implement `VectorStoreService` class structure
- [x] Implement `__init__` method
- [x] Implement `index_exists` method
- [x] Create tests in `tests/test_vector_store.py` (or similar)

## Dev Agent Record

### Debug Log
- [x] Initial plan created
- [x] Implemented basic class structure and file handling
- [x] Verified with unit tests

### Completion Notes
- [x] Created `VectorStoreService` with basic file handling capabilities.
- [x] Added unit tests covering initialization and file existence checks.

## File List
- [x] app/services/vector_store.py
- [x] app/services/__init__.py
- [x] tests/test_vector_store.py

## Change Log
- [x] Initial creation of VectorStoreService

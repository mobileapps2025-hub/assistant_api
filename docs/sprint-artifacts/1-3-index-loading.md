# Story 1.3: Implement Index Loading

**Epic**: Persistent Vector Store
**Status**: completed

## Description
Implement the `load_index` method in `VectorStoreService` to load the FAISS index and metadata from disk. This allows the application to start up quickly using a pre-built index.

## Acceptance Criteria
- [x] `VectorStoreService` has a `load_index()` method
- [x] Method checks if index exists before loading
- [x] Method loads `faiss_index.bin` into memory
- [x] Method loads `chunk_metadata.json` into memory
- [x] Method initializes the embedding model if not already loaded
- [x] Unit tests verify that index and metadata are correctly loaded

## Tasks/Subtasks
- [x] Implement `load_index` in `app/services/vector_store.py`
- [x] Ensure `_initialize_model` is called during load
- [x] Add error handling for corrupted or missing files
- [x] Create tests in `tests/test_vector_store.py`

## Dev Agent Record

### Debug Log
- [x] Initial plan created
- [x] Implemented `load_index`
- [x] Added unit tests
- [x] Tests passed

### Completion Notes
- `load_index` successfully loads the FAISS index and metadata.
- It also initializes the embedding model so that the service is ready for queries immediately after loading.
- Unit tests cover successful loading and handling of missing index files.

## File List
- [x] app/services/vector_store.py
- [x] tests/test_vector_store.py

## Change Log
- Added `load_index` to `app/services/vector_store.py`
- Added `test_load_index` and `test_load_index_missing` to `tests/test_vector_store.py`

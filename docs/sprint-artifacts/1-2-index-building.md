# Story 1.2: Implement Index Building & Persistence

**Epic**: Persistent Vector Store
**Status**: completed

## Description
Implement the `build_index` method in `VectorStoreService` to process documents, generate embeddings, build a FAISS index, and save it to disk. This enables the application to load a pre-built index instead of rebuilding it on every startup.

## Acceptance Criteria
- [x] `requirements.txt` includes `faiss-cpu` and `sentence-transformers`
- [x] `VectorStoreService` has a `build_index(documents_path)` method
- [x] Method reads supported files (PDF, DOCX, etc.) from the given path
- [x] Method chunks text appropriately
- [x] Method generates embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`
- [x] Method builds a FAISS index from embeddings
- [x] Method saves `faiss_index.bin` and `chunk_metadata.json` to `storage_path`
- [x] Unit tests verify that files are created after `build_index` is called

## Tasks/Subtasks
- [x] Update `requirements.txt`
- [x] Refactor document reading/chunking logic (move from `app/services.py` or duplicate for now, then cleanup later)
- [x] Implement `build_index` in `app/services/vector_store.py`
- [x] Add `save_index` helper method
- [x] Create tests in `tests/test_vector_store.py`

## Dev Agent Record

### Debug Log
- [x] Initial plan created
- [x] Dependencies installed (faiss-cpu>=1.9.0, sentence-transformers>=2.7.0)
- [x] Unit tests passed

### Completion Notes
- Implemented `VectorStoreService.build_index` with support for PDF, DOCX, PPTX, and MD files.
- Used `faiss.IndexFlatIP` for cosine similarity.
- Updated `requirements.txt` to use compatible versions of `faiss-cpu` and `sentence-transformers`.

## File List
- [x] requirements.txt
- [x] app/services/vector_store.py
- [x] tests/test_vector_store.py

## Change Log
- Updated `requirements.txt`
- Modified `app/services/vector_store.py`
- Modified `tests/test_vector_store.py`


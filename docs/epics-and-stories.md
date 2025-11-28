# Epics and Stories: MCL AI Agent 2.0

## Epic 1: Persistent Vector Store
**Goal**: Decouple document indexing from application startup to improve performance and scalability.

### Story 1.1: VectorStoreService Skeleton & File Handling
- **Description**: Create the `VectorStoreService` class in `app/services/vector_store.py`. Implement logic to define storage paths for the index and metadata.
- **Acceptance Criteria**:
  - Class initializes with a storage path.
  - Can check if index files exist on disk.

### Story 1.2: Implement Index Building & Persistence
- **Description**: Implement the `build_index` method to process documents, generate embeddings, build a FAISS index, and save it to disk (`faiss_index.bin`, `chunk_metadata.json`).
- **Acceptance Criteria**:
  - Reads files from `app/documents/`.
  - Generates embeddings using `SentenceTransformer` (or OpenAI).
  - Saves FAISS index and metadata to disk.
  - Startup time is not affected by this process (it will be run manually or on demand).

### Story 1.3: Implement Index Loading
- **Description**: Implement `load_index` to load the FAISS index and metadata from disk into memory on startup.
- **Acceptance Criteria**:
  - Loads index in < 2 seconds.
  - Handles missing index files gracefully (logs warning).

### Story 1.4: Semantic Search Implementation
- **Description**: Implement the `search` method in `VectorStoreService` to query the loaded FAISS index.
- **Acceptance Criteria**:
  - Returns relevant chunks with similarity scores.
  - Matches the accuracy of the previous implementation.

## Epic 2: Unified Vision Service
**Goal**: Consolidate Vision AI logic into a single, efficient service using GPT-4o.

### Story 2.1: VisionService Implementation
- **Description**: Create `app/services/vision.py` with a `VisionService` class. Implement `analyze_image` using `client.chat.completions.create` (GPT-4o).
- **Acceptance Criteria**:
  - Accepts image (base64/url) and query.
  - Returns text response from GPT-4o.
  - Does NOT use the Assistants API.

### Story 2.2: Port Image Validation
- **Description**: Move the MCL-specific image validation logic (checking if it's an MCL screenshot) into `VisionService` or a helper.
- **Acceptance Criteria**:
  - Validates images before sending to GPT-4o (optional, based on config).

### Story 2.3: Refactor Vision Endpoint
- **Description**: Update the `/api/vision/analyze-screenshot` endpoint in `app/main.py` (or new routes file) to use `VisionService`.
- **Acceptance Criteria**:
  - Endpoint accepts file upload.
  - Returns analysis result.
  - Latency is reduced (no polling).

## Epic 3: Architecture Refactoring
**Goal**: Modernize the codebase structure and improve observability.

### Story 3.1: Project Restructuring
- **Description**: Create the new directory structure (`app/services`, `app/core`, `app/api`). Move existing files to their new locations.
- **Acceptance Criteria**:
  - New folders exist.
  - `__init__.py` files created.

### Story 3.2: Structured Logging
- **Description**: Create `app/core/logging.py` and configure Python's `logging` module. Replace `print()` statements in key files.
- **Acceptance Criteria**:
  - Logs output to console with timestamps and levels.
  - No `print()` statements in `VectorStoreService` or `VisionService`.

### Story 3.3: Dependency Injection & Main Refactor
- **Description**: Refactor `app/main.py` to initialize services (`VectorStoreService`, `VisionService`) and inject them into route handlers using `FastAPI.Depends`.
- **Acceptance Criteria**:
  - No global variables for services.
  - App starts up successfully.

### Story 3.4: Cleanup Legacy Code
- **Description**: Delete `app/vision_assistant.py` and the old `app/services.py`.
- **Acceptance Criteria**:
  - Codebase is clean.
  - All tests (if any) pass.

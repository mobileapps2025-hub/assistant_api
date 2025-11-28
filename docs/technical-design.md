# Technical Design: MCL AI Agent 2.0

## 1. Architecture Overview
The goal of this refactor is to transition the MCL AI Agent from a stateful, monolithic script-based architecture to a modular, service-oriented design with persistent storage.

### Current vs. Target Architecture

| Component | Current State | Target State |
| :--- | :--- | :--- |
| **Vector Store** | In-memory FAISS, rebuilt on every startup. | Persistent FAISS index on disk + Metadata DB. |
| **Vision AI** | Split between `ChatCompletion` and `Assistants API`. | Unified `ChatCompletion` (GPT-4o) service. |
| **State Management** | Global variables in `services.py`. | Dependency Injection (Singleton Services). |
| **Logging** | `print()` statements. | Structured `logging` module. |

## 2. Detailed Design

### 2.1. Persistent Vector Store Service
We will create a new `VectorStoreService` class responsible for managing the FAISS index and document metadata.

**Storage Strategy:**
-   **Index**: `faiss_index.bin` (The actual vector index).
-   **Metadata**: `chunk_metadata.json` (List of dicts containing text content, source, etc.).
-   **Hash Map**: `file_hashes.json` (To track file changes and avoid re-indexing unchanged files).

**Class Structure (`app/services/vector_store.py`):**
```python
class VectorStoreService:
    def __init__(self, storage_path: str):
        self.index = None
        self.chunks = []
        self.load_index()

    def load_index(self):
        """Loads index and metadata from disk if they exist."""
        # Load faiss_index.bin
        # Load chunk_metadata.json

    def build_index(self, documents_path: str):
        """Scans documents, creates chunks, generates embeddings, saves to disk."""
        # 1. Scan files
        # 2. Check hashes (skip unchanged)
        # 3. Generate embeddings (OpenAI/SentenceTransformer)
        # 4. Build FAISS index
        # 5. Save to disk

    def search(self, query: str, k: int = 5):
        """Performs semantic search."""
```

### 2.2. Unified Vision Service
We will deprecate the `MCLVisionAssistant` class (which uses the Assistants API) and consolidate logic into a `VisionService`.

**Class Structure (`app/services/vision.py`):**
```python
class VisionService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    async def analyze_image(self, image_base64: str, query: str, context: dict) -> str:
        """
        Uses GPT-4o Chat Completion with image input.
        Replaces the polling mechanism of Assistants API.
        """
        messages = [
            {"role": "system", "content": "You are the MCL App Assistant..."},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]}
        ]
        # Call OpenAI
```

### 2.3. Dependency Injection & App Structure
We will refactor `app/main.py` to use FastAPI's dependency injection system instead of importing global variables.

**New Directory Structure:**
```
app/
  services/
    __init__.py
    vector_store.py  # New
    vision.py        # New
    chat.py          # Refactored from services.py
  core/
    config.py        # Configuration & Settings
    logging.py       # Logging setup
  api/
    routes.py        # API endpoints
  main.py            # App entry point
```

### 2.4. Logging Strategy
We will implement a centralized logging configuration in `app/core/logging.py`.

```python
# app/core/logging.py
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
```

## 3. Migration Plan

1.  **Phase 1: Core Services**: Create `VectorStoreService` and `VisionService`. Implement persistence.
2.  **Phase 2: Refactor Endpoints**: Update `app/main.py` to use the new services.
3.  **Phase 3: Cleanup**: Remove old `services.py`, `vision_assistant.py`, and global variables.

## 4. Security Considerations
-   Ensure `faiss_index.bin` and metadata files are stored in a secure location (not exposed via static file serving).
-   Validate all image inputs before processing (size, type).

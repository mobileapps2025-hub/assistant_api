# Product Requirements Document (PRD)
## MCL AI Agent 2.0: Architecture Refactor

| Metadata | Details |
| :--- | :--- |
| **Project** | MCL AI Assistant |
| **Version** | 2.0 |
| **Status** | Draft |
| **Date** | 2025-11-27 |

## 1. Introduction
The MCL AI Assistant is a critical tool for providing context-aware help to users. However, the current "Brownfield" implementation suffers from architectural technical debt that limits its performance and scalability. This PRD outlines the requirements for refactoring the core services to enable persistence, faster startup, and unified vision capabilities.

## 2. Problem Statement
-   **Slow Startup**: The application re-processes and re-indexes all PDF documents into an in-memory vector store every time it starts. This causes downtime during deployments and prevents quick scaling.
-   **Inconsistent Vision AI**: Two different OpenAI APIs (Chat Completions vs. Assistants API) are used for similar vision tasks, leading to code duplication and inconsistent user experiences.
-   **Stateful Architecture**: The heavy reliance on global variables prevents the application from running on multiple workers or scaling horizontally.

## 3. Goals & Objectives
1.  **Persistence**: Decouple document indexing from application startup. Store embeddings persistently.
2.  **Performance**: Achieve a startup time of under 5 seconds, regardless of document count.
3.  **Unification**: Consolidate all Vision AI logic into a single, efficient service using the GPT-4o Chat API.
4.  **Observability**: Replace print-based debugging with structured logging.

## 4. Functional Requirements

### 4.1. Persistent Vector Store
-   **FR-01**: The system **MUST** persist document embeddings and chunks to disk (or a database) after indexing.
-   **FR-02**: On startup, the system **MUST** load the index from persistence instead of re-processing raw files.
-   **FR-03**: The system **MUST** provide an API endpoint (e.g., `POST /api/admin/reindex`) to trigger a manual re-indexing of documents.

### 4.2. Unified Vision Service
-   **FR-04**: The `MCLVisionAssistant` class (Assistants API implementation) **MUST** be deprecated.
-   **FR-05**: The `/api/vision/analyze-screenshot` endpoint **MUST** be refactored to use the same `get_vision_enabled_response` logic (Chat Completions API) used by the chat endpoint.
-   **FR-06**: Image validation logic **MUST** be applied consistently across all vision endpoints.

### 4.3. Logging & Monitoring
-   **FR-07**: All `print()` statements used for application logic **MUST** be replaced with Python's `logging` module.
-   **FR-08**: Logs **MUST** include timestamps, log levels (INFO, ERROR, DEBUG), and request IDs where applicable.

## 5. Non-Functional Requirements
-   **NFR-01 (Performance)**: Application startup time must be < 5 seconds.
-   **NFR-02 (Scalability)**: The application must be stateless (except for the loaded read-only index), allowing for multiple Uvicorn workers.
-   **NFR-03 (Reliability)**: Vision API requests should not rely on polling (which is required by the Assistants API) to reduce latency and failure points.

## 6. Technical Strategy
-   **Vector Store**: Use `FAISS` with `write_index` / `read_index` for disk persistence. Store metadata (chunks) in a companion JSON/Pickle file or SQLite.
-   **Vision**: Standardize on `gpt-4o` via `client.chat.completions.create`.
-   **Refactoring**: Move logic from `app/services.py` into dedicated classes (e.g., `VectorStoreService`, `VisionService`) to eliminate global state.

## 7. Out of Scope
-   Adding new user-facing features (UI changes).
-   Changing the underlying LLM model (sticking with GPT-4o).
-   Migrating to a cloud-native vector DB (e.g., Pinecone) is out of scope for this sprint; local persistence is sufficient.

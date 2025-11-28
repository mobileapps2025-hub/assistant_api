# Research Findings: MCL AI Agent

## Executive Summary
The MCL AI Agent is a functional FastAPI application that leverages OpenAI's GPT-4o for text and vision tasks. However, it suffers from significant architectural limitations that hinder scalability and performance. The primary issues are the in-memory, non-persistent vector store that re-indexes on every startup, and inconsistent implementation of Vision AI capabilities.

## Key Findings

### 1. RAG Performance & Scalability (Critical)
- **Issue**: The application re-processes and re-indexes all documents in `app/documents/` every time the server starts (`start_mcl_knowledge_base` in `app/services.py`).
- **Impact**: Slow startup times (linear growth with document count) and loss of index on restart.
- **Recommendation**: Implement a persistent vector store (e.g., FAISS on disk, ChromaDB, or Qdrant). Indexing should be a separate background process or CLI command, not part of the app startup.

### 2. Vision AI Inconsistency
- **Issue**: There are two separate implementations for Vision AI:
  1.  `/api/chat` uses `get_vision_enabled_response` (Chat Completions API) for multimodal chat.
  2.  `/api/vision/analyze-screenshot` uses `MCLVisionAssistant` (Assistants API) for file uploads.
- **Impact**: Duplicate code, inconsistent behavior, and higher latency for the Assistants API (due to polling).
- **Recommendation**: Consolidate on the Chat Completions API (GPT-4o) for all vision tasks. It supports streaming and is faster.

### 3. Global State Management
- **Issue**: The application relies heavily on global variables (`_mcl_document_chunks`, `_mcl_faiss_index`, `MCL_VECTOR_STORE_ID`) to store application state.
- **Impact**: The application is stateful, making it impossible to scale horizontally (run multiple workers/instances).
- **Recommendation**: Move state to external services (Redis for cache/session, Vector DB for embeddings).

### 4. Code Quality & Observability
- **Issue**: Extensive use of `print()` statements for logging.
- **Impact**: Difficult to debug in production; logs are unstructured.
- **Recommendation**: Replace `print()` with the standard Python `logging` module.

### 5. Context Management
- **Strength**: The `ContextAnalysis` in `app/context_manager.py` is a well-implemented rule-based system for detecting user context (Mobile vs. Web, OS, etc.).
- **Recommendation**: Keep this logic but integrate it more cleanly into the prompt construction pipeline.

## Proposed Improvements (Sprint Candidates)

1.  **Persistence Layer**: Implement a persistent vector store to remove startup indexing.
2.  **Unified Vision Service**: Refactor `vision_assistant.py` to use the same logic as `services.py`.
3.  **Async Indexing**: Create an API endpoint or script to trigger document indexing without restarting the server.
4.  **Logging**: Implement structured logging.

## Technical Debt
- **Custom Chunking**: `create_text_chunks` is custom and fragile. Suggest moving to `langchain.text_splitter`.
- **Global Variables**: Must be refactored for production readiness.

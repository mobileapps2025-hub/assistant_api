# mcl-ai-agent - Technical Specification

**Author:** BMad
**Date:** December 8, 2025
**Project Level:** Brownfield
**Change Type:** Architecture Upgrade (Phase 1)
**Development Context:** Phase 1 of Self-Adaptive Agent Strategy

---

## Context

### Available Documents

*   **Strategy Document**: `docs/implementation-strategy-mcl-agent.md`
*   **Research Findings**: `docs/research-findings.md`
*   **Project Summary**: `docs/project-summary.md`

### Project Stack

*   **Framework**: FastAPI 0.116.0
*   **Language**: Python 3.11+
*   **AI/ML**: LangChain 0.3.26, OpenAI 1.93.2
*   **Vector Store**: Currently FAISS (local), Target: Weaviate
*   **Orchestration**: Currently Linear, Target: LangGraph
*   **Dependencies**: Managed via `requirements.txt`

### Existing Codebase Structure

*   **Entry Point**: `app/main.py`
*   **Services**: `app/services/` (Chat, Vision, Vector Store)
*   **Core**: `app/core/` (Config, Logging)
*   **Routers**: `app/routers/`
*   **Documents**: `app/documents/` (Source PDFs/MDs)
*   **Tests**: `tests/` (Pytest)

---

## The Change

### Problem Statement

The current RAG implementation suffers from "retrieval drift" and context loss due to naive chunking of Markdown files and reliance on simple vector search. This leads to inconsistent answers. Additionally, the system lacks a persistent vector store, causing slow startups.

### Proposed Solution

Implement **Phase 1: Foundation (Data & Retrieval)** of the new agent architecture. This involves:
1.  **Structure-Aware Ingestion**: Replacing naive chunking with `MarkdownHeaderTextSplitter` to preserve header context.
2.  **Hybrid Search**: Migrating from FAISS to **Weaviate** to enable combined Vector + Keyword (BM25) search.
3.  **Re-ranking**: Integrating **Cohere Rerank** to filter irrelevant results.
4.  **Persistence**: Decoupling the vector store from app startup.

### Scope

**In Scope:**

*   New `IngestionService` with `MarkdownHeaderTextSplitter`.
*   New `WeaviateVectorStore` service replacing `VectorStoreService`.
*   Integration of `Cohere` for re-ranking.
*   New API endpoint `POST /api/admin/ingest` to trigger indexing.
*   Updates to `ChatService` to use the new retrieval pipeline.
*   Dependency updates (`weaviate-client`, `cohere`, `langchain-weaviate`).

**Out of Scope:**

*   Multimodal/Vision ingestion (Phase 2).
*   LangGraph orchestration (Phase 3).
*   DSPy optimization (Phase 4).
*   Frontend changes.

---

## Implementation Details

### Source Tree Changes

*   `requirements.txt` - MODIFY - Add `weaviate-client`, `cohere`, `langchain-weaviate`.
*   `app/core/config.py` - MODIFY - Add `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `COHERE_API_KEY`.
*   `app/services/ingestion_service.py` - CREATE - Handles Markdown parsing and Weaviate indexing.
*   `app/services/vector_store.py` - MODIFY - Rewrite to use Weaviate instead of FAISS.
*   `app/routers/admin.py` - CREATE - Endpoint for triggering ingestion.
*   `app/services/chat_service.py` - MODIFY - Update retrieval logic to use `hybrid_search` and `rerank`.

### Technical Approach

1.  **Vector Store**: Use `Weaviate` (Cloud or Local). Schema will include `text`, `header_path`, and `source` fields.
2.  **Ingestion**:
    *   Read `.md` files from `app/documents/`.
    *   Split using `MarkdownHeaderTextSplitter` (headers: `#`, `##`, `###`).
    *   Batch upload to Weaviate.
3.  **Retrieval**:
    *   `hybrid_search(query, alpha=0.5)`: Combines dense vector and sparse keyword scores.
    *   `rerank(query, results)`: Uses Cohere to re-score top 25 results, returning top 5 with score > 0.7.

### Existing Patterns to Follow

*   **Service Pattern**: Keep logic in `app/services/`.
*   **Dependency Injection**: Pass services to routers/other services (as seen in `ChatService`).
*   **Configuration**: Use `app/core/config.py` for environment variables.
*   **Async/Await**: Use async methods for all I/O operations (Weaviate/OpenAI calls).

### Integration Points

*   **Weaviate API**: External vector database.
*   **Cohere API**: External re-ranking service.
*   **OpenAI API**: Existing embedding model (`text-embedding-3-small` recommended over `all-MiniLM-L6-v2` for better performance, or keep existing if preferred). *Decision: Upgrade to OpenAI Embeddings for better compatibility with Weaviate/Hybrid.*

---

## Development Context

### Relevant Existing Code

*   `app/services/vector_store.py`: Current FAISS implementation (to be replaced).
*   `app/services/chat_service.py`: Current consumer of vector store.
*   `app/core/config.py`: Config management.

### Dependencies

**Framework/Libraries:**

*   `weaviate-client` (v4.x)
*   `cohere`
*   `langchain-weaviate`
*   `langchain-text-splitters`

**Internal Modules:**

*   `app.core.logging`

### Configuration Changes

New Environment Variables:
```
WEAVIATE_URL=...
WEAVIATE_API_KEY=...
COHERE_API_KEY=...
```

### Existing Conventions (Brownfield)

*   Use `get_logger` from `app.core.logging`.
*   Type hinting for all function arguments.
*   Pydantic models for API request/response.

### Test Framework & Standards

*   `pytest` is the runner.
*   Mock external APIs (Weaviate, Cohere) in tests.

---

## Implementation Stack

*   **Runtime**: Python 3.11+
*   **Vector DB**: Weaviate
*   **Re-ranker**: Cohere
*   **Embeddings**: OpenAI `text-embedding-3-small`

---

## Technical Details

*   **Hybrid Search Alpha**: Set to 0.5 initially (balanced).
*   **Rerank Threshold**: 0.7 (strict filtering).
*   **Chunking**: Split by headers. If a chunk is still too large (>2000 chars), apply recursive character splitting on top.

---

## Development Setup

1.  `pip install -r requirements.txt`
2.  Set up Weaviate (Docker or WCS).
3.  Get Cohere API Key (Trial key is fine).
4.  Update `.env`.
5.  Run `uvicorn app.main:app --reload`.

---

## Implementation Guide

### Setup Steps

1.  Add dependencies to `requirements.txt`.
2.  Update `app/core/config.py`.
3.  Set up Weaviate instance.

### Implementation Steps

1.  **Ingestion Service**: Implement `app/services/ingestion_service.py` with `MarkdownHeaderTextSplitter`.
2.  **Vector Store**: Implement `WeaviateVectorStore` in `app/services/vector_store.py`.
3.  **Admin Router**: Create `POST /ingest` in `app/routers/admin.py`.
4.  **Chat Service**: Update `ChatService` to call `vector_store.hybrid_search` and then `cohere.rerank`.
5.  **Main**: Register new router.

### Testing Strategy

*   **Unit**: Test `IngestionService` splitting logic (mock file reading).
*   **Integration**: Test `WeaviateVectorStore` against a real/mocked Weaviate instance.
*   **End-to-End**: Test `POST /ingest` and then `POST /chat` to verify retrieval.

### Acceptance Criteria

1.  Ingestion preserves Markdown headers in metadata.
2.  Searching for specific error codes (keyword) returns the correct document (Hybrid search working).
3.  Irrelevant documents are filtered out by Re-ranker (Score < 0.7).
4.  App startup is instant (no re-indexing).

---

## Developer Resources

### File Paths Reference

*   `app/services/ingestion_service.py`
*   `app/services/vector_store.py`
*   `app/routers/admin.py`

### Key Code Locations

*   `ChatService.process_chat_request`

### Testing Locations

*   `tests/test_ingestion.py`
*   `tests/test_vector_store.py`

### Documentation to Update

*   `README.md` (Setup instructions)
*   `docs/api-contracts-backend.md` (New admin endpoint)

---

## UX/UI Considerations

None. Backend changes only.

---

## Testing Approach

*   **Unit Tests**: 80% coverage for new services.
*   **Manual Verification**: Verify "ERR-502" query returns exact match.

---

## Deployment Strategy

### Deployment Steps

1.  Deploy Weaviate (if self-hosted) or configure WCS.
2.  Deploy App code.
3.  Run Ingestion (via API or script) once to populate DB.

### Rollback Plan

Revert to previous commit (FAISS implementation).

### Monitoring

Monitor Weaviate latency and Cohere API errors.

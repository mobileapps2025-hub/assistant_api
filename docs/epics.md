# mcl-ai-agent - Epic Breakdown

**Date:** December 8, 2025
**Project Level:** Brownfield

---

## Epic 1: Architecture Upgrade Phase 1: Foundation

**Slug:** arch-upgrade-phase-1

### Goal

Establish a robust, structure-aware retrieval pipeline with Weaviate and Cohere to eliminate retrieval drift and enable hybrid search. This foundation is critical for the subsequent multimodal and self-adaptive capabilities.

### Scope

**In Scope:**
*   Migration from FAISS to Weaviate Vector Store.
*   Implementation of `MarkdownHeaderTextSplitter` for structure-aware chunking.
*   Creation of an Admin Ingestion Endpoint (`POST /api/admin/ingest`).
*   Integration of Hybrid Search (Vector + BM25) and Cohere Re-ranking in `ChatService`.

**Out of Scope:**
*   Vision/Multimodal ingestion (Phase 2).
*   LangGraph orchestration (Phase 3).
*   Frontend UI changes.

### Success Criteria

1.  **Structure Preservation**: Ingested chunks contain metadata reflecting their Markdown header hierarchy (e.g., `{"header": "Settings > Profile"}`).
2.  **Hybrid Accuracy**: Searching for specific error codes (e.g., "ERR-502") returns the exact document, even if semantically generic.
3.  **Re-ranking Precision**: Irrelevant documents (score < 0.7) are filtered out before reaching the LLM context.
4.  **Performance**: Application startup time is instantaneous and independent of the document set size (no re-indexing on boot).

### Dependencies

*   Weaviate Instance (Cloud or Local)
*   Cohere API Key
*   OpenAI API Key

---

## Story Map - Epic 1

```
Epic: Architecture Upgrade Phase 1: Foundation (8 points)
├── Story 1.1: Implement Weaviate Vector Store Service (3 points)
│   Dependencies: None
│
├── Story 1.2: Implement Structure-Aware Ingestion Service (3 points)
│   Dependencies: Story 1.1
│
└── Story 1.3: Upgrade Chat Service with Hybrid Retrieval (2 points)
    Dependencies: Stories 1.1, 1.2
```

---

## Stories - Epic 1

### Story 1.1: Implement Weaviate Vector Store Service

As a **Backend Developer**,
I want **to replace the local FAISS vector store with a Weaviate client**,
So that **the application has a persistent, scalable knowledge base that supports hybrid search**.

**Acceptance Criteria:**

**Given** a configured Weaviate instance
**When** the application starts
**Then** it should connect to Weaviate without errors

**And** the `VectorStoreService` should support `add_documents` and `hybrid_search` methods.

**Prerequisites:** Weaviate URL/Key in .env

**Technical Notes:** Use `weaviate-client` v4.

**Estimated Effort:** 3 points (1-2 days)

### Story 1.2: Implement Structure-Aware Ingestion Service

As a **Backend Developer**,
I want **to ingest Markdown files using header-based splitting**,
So that **retrieved chunks retain their semantic context (e.g., which section they belong to)**.

**Acceptance Criteria:**

**Given** a Markdown file with headers
**When** I trigger the `POST /api/admin/ingest` endpoint
**Then** the file should be split by headers (#, ##, ###)
**And** chunks should be uploaded to Weaviate with `header_path` metadata.

**Prerequisites:** Story 1.1

**Technical Notes:** Use `MarkdownHeaderTextSplitter`.

**Estimated Effort:** 3 points (1-2 days)

### Story 1.3: Upgrade Chat Service with Hybrid Retrieval

As a **Product Manager**,
I want **the chat bot to use Hybrid Search and Re-ranking**,
So that **users get accurate answers for both conceptual questions and specific technical queries**.

**Acceptance Criteria:**

**Given** a user query
**When** the system retrieves documents
**Then** it should use Hybrid Search (Alpha=0.5)
**And** re-rank the top 25 results using Cohere
**And** only return results with a re-rank score > 0.7.

**Prerequisites:** Story 1.1, 1.2

**Technical Notes:** Use `cohere` SDK for re-ranking.

**Estimated Effort:** 2 points (1 day)

---

## Implementation Timeline - Epic 1

**Total Story Points:** 8

**Estimated Timeline:** 3-5 days

---

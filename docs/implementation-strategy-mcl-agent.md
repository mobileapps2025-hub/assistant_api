# Implementation Strategy: Self-Adaptive Multimodal AI Agent

**Date:** December 8, 2025
**Author:** John (Product Manager)
**Source:** AI Agent Development Research Report (1).md

## 1. Executive Summary

This document outlines the strategic roadmap for upgrading the current MCL AI Agent from a basic RAG implementation to a **Self-Adaptive Multimodal Agent**. The goal is to resolve current inconsistencies in retrieval, enable robust visual guidance, and establish an automated self-improvement loop.

## 2. Current State Analysis

| Component | Current Implementation | Limitations |
| :--- | :--- | :--- |
| **Ingestion** | Naive chunking (likely fixed-size) | Breaks Markdown structure; loses context of headers. |
| **Retrieval** | Dense Vector Search (`faiss` + `all-MiniLM-L6-v2`) | Fails on specific keywords (error codes, versions); no re-ranking. |
| **Vision** | Direct GPT-4o call (`ChatService`) | No visual grounding; prone to hallucinating UI elements. |
| **Orchestration** | Linear Logic (`ChatService` if/else) | Brittle; cannot "retry" or "reflect" on bad retrievals. |
| **Feedback** | None | No mechanism to learn from user corrections. |

## 3. Target Architecture

The new architecture shifts to a **Graph-Based Agent** with **Hybrid Search** and **Programmatic Optimization**.

### 3.1 Core Components
1.  **Structure-Aware Ingestion**:
    *   Use `MarkdownHeaderTextSplitter` to preserve document hierarchy.
    *   **Multimodal Indexing**: Generate text summaries for documentation images using GPT-4o-mini and index them for retrieval.
2.  **Hybrid Retrieval Layer**:
    *   **Database**: Migrate to **Weaviate** (or Chroma with BM25 support) to enable Hybrid Search (Vector + Keyword).
    *   **Re-ranking**: Integrate **Cohere Rerank** to filter irrelevant chunks before they reach the LLM.
3.  **Agent Orchestration (LangGraph)**:
    *   Model the agent as a State Machine.
    *   **Nodes**: `Retrieve`, `GradeDocuments`, `RewriteQuery`, `GenerateAnswer`.
    *   **Self-Correction**: If retrieved documents are graded as "irrelevant", the agent rewrites the query and retries.
4.  **Visual Grounding**:
    *   Implement **Set-of-Mark (SoM)** preprocessing for user screenshots to enable precise UI navigation instructions.
5.  **Self-Improvement (DSPy)**:
    *   Implement a feedback pipeline where "Negative Feedback + Correction" pairs are used to optimize prompts using **DSPy's MIPRO** optimizer.

## 4. Implementation Roadmap

### Phase 1: Foundation (Data & Retrieval)
**Goal**: Stabilize retrieval and eliminate "retrieval drift".
1.  **Dependency Update**: Add `langgraph`, `weaviate-client`, `cohere`, `dspy-ai`.
2.  **Refactor Ingestion**:
    *   Create `app/services/ingestion_service.py`.
    *   Implement `MarkdownHeaderTextSplitter`.
    *   Implement Image Summarization (GPT-4o-mini).
3.  **Upgrade Vector Store**:
    *   Replace `VectorStoreService` (FAISS) with a Weaviate-based implementation.
    *   Implement `hybrid_search` method.
4.  **Add Re-ranking**:
    *   Integrate Cohere Rerank API.

### Phase 2: Multimodal Capabilities
**Goal**: Enable the agent to "see" and "guide".
1.  **Visual Knowledge Base**: Index the image summaries generated in Phase 1.
2.  **Vision Tool**:
    *   Create `app/tools/vision.py`.
    *   Implement Set-of-Mark preprocessing (using `supervision` or similar, or a lightweight DOM-based approach if applicable).

### Phase 3: The Agentic Graph
**Goal**: Move from linear chains to cognitive graphs.
1.  **Define State**: Create `AgentState` (TypedDict) in `app/core/state.py`.
2.  **Build Graph**:
    *   Implement nodes in `app/core/graph_nodes.py`.
    *   Define edges and conditional logic (Reflection Loop).
3.  **Persistence**: Configure `PostgresCheckpointer` (or SQLite for local dev) to maintain thread state.

### Phase 4: The Feedback Loop
**Goal**: Automated optimization.
1.  **Feedback Endpoint**: Update API to accept `thumbs_up`/`thumbs_down` and `correction`.
2.  **DSPy Integration**:
    *   Refactor Generation Node to use `dspy.Module`.
    *   Create `scripts/optimize_agent.py` to run MIPRO on feedback data.

## 5. Technical Stack Recommendations

| Category | Tool | Justification |
| :--- | :--- | :--- |
| **Orchestration** | **LangGraph** | Essential for cyclic state management. |
| **Vector DB** | **Weaviate** | Best support for Hybrid Search and Multimodal. |
| **Optimization** | **DSPy** | Only mature framework for prompt compilation. |
| **LLM** | **GPT-4o** | Required for high-fidelity vision and reasoning. |
| **Re-ranker** | **Cohere** | Industry standard for RAG precision. |

## 6. Next Steps
1.  Approve this strategy.
2.  Execute **Phase 1** (Ingestion & Retrieval Upgrade).
3.  Run `*tech-spec` to generate detailed task tickets for Phase 1.

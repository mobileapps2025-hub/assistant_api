# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start dev server
python -m uvicorn app.main:app --reload

# Run tests
python -m pytest tests/
python -m pytest tests/test_chat_service.py::test_function_name -v

# Start local Weaviate
docker-compose up weaviate

# Trigger document ingestion (requires ADMIN_API_KEY)
curl -X POST http://localhost:8000/admin/ingest -H "X-Admin-Key: your-key"

# Train agent from feedback (requires ADMIN_API_KEY)
python train_agent.py
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | ✅ Yes | — | GPT-4o + embeddings; app refuses to start without it |
| `WEAVIATE_URL` | No | `http://localhost:8080` | Vector store |
| `WEAVIATE_API_KEY` | No | `""` | Weaviate auth (cloud deployments) |
| `COHERE_API_KEY` | No | `""` | Reranking; falls back to top-10 without it |
| `DATABASE_CONNECTION_STRING` | No | `""` | Async SQL Server; feedback system disabled without it |
| `ADMIN_API_KEY` | No | `""` | Protects `/admin/*` routes; returns 503 if unset |
| `CORS_ORIGINS` | No | localhost ports | Comma-separated allowed origins |

## Architecture

MCL Assistant is a FastAPI RAG app serving the MCL (Mobile Checklist) mobile app.

### Request flow

1. `POST /api/chat` → `ChatService.process_chat_request()` (`app/services/chat_service.py`)
2. If image attached → `VisionService` (GPT-4o vision)
3. Otherwise → **LangGraph state machine** (`app/graph/`):
   - `detect_language` → `retrieve_documents` → `grade_documents`
   - Relevant docs: `generate_answer`
   - Irrelevant / grading error: `rewrite_query` (max 1 retry) → `generate_answer` or `clarify_ambiguity`
4. Returns `{response, response_id, sources}`

### Key components

| Component | Location | Role |
|-----------|----------|------|
| LangGraph workflow | `app/graph/` | Agentic RAG state machine |
| VectorStoreService | `app/services/vector_store.py` | Weaviate hybrid search + Cohere reranking |
| ChatService | `app/services/chat_service.py` | Orchestrates text/vision routing |
| ContextAnalyzer | `app/core/context.py` | Infers user interface/device/expertise from query |
| IngestionService | `app/services/ingestion_service.py` | Loads PDF/MD/DOCX/PPTX from `app/documents/` into Weaviate |
| DSPy optimizer | `app/optimization/` | Prompt optimization via feedback; `compiled_rag.json` loaded at startup |
| Feedback loop | `app/core/database.py`, `app/routers/admin.py` | Stores feedback → CuratedQA → DSPy retraining |

### Admin routes (require `X-Admin-Key` header)

- `POST /admin/ingest` — re-ingest all documents from `app/documents/`
- `POST /admin/curated-qa` — add a curated Q&A pair
- `POST /admin/train` — trigger DSPy retraining from feedback

### Search strategy

Weaviate hybrid search (semantic + keyword, `alpha=0.5`) → Cohere reranker (threshold `>0.7`) → top-N chunks → GPT-4o.

### Self-improvement pipeline

User feedback → `Feedback` DB table → `CuratedQA` table → DSPy `BootstrapFewShot` → `app/optimization/compiled_rag.json`.

### Startup sequence (`app/main.py` lifespan)

1. Validate `OPENAI_API_KEY` (raises at import if missing)
2. Create async DB tables (if DB available)
3. Initialize Weaviate schema
4. If vector store empty → auto-ingest from `app/documents/`

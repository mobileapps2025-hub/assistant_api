# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Also read the repo-root `PROJECT_KNOWLEDGE.md` for the layered-rebuild decisions and module status.

## Commands

```bash
pip install -r requirements.txt           # runtime deps
python -m uvicorn app.main:app --reload    # dev server
python -m pytest tests/ -q                 # all tests
python -m pytest tests/test_routing.py -q  # a single file
python -m pytest tests/ --cov=app --cov-report=term-missing
```

Ragie ingestion is manual (Ragie dashboard, or the `spikes/ragie_spike.py` helper). There is
no local vector store and no startup ingestion.

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | ✅ Yes | — | GPT-4o (answers, routing, vision); app refuses to start without it |
| `RAGIE_API_KEY` | For KB answers | `""` | Ragie retrieval + image proxy; KNOWLEDGE answers degrade to a fallback without it |
| `RAGIE_PARTITION` | No | `mcl_spike` | Ragie partition holding the MCL docs |
| `RAGIE_TOP_K` | No | `6` | Chunks retrieved per query |
| `API_PUBLIC_URL` | No | `http://127.0.0.1:8001` | Public base URL used to build `/api/ragie/image` links the frontend can load |
| `DATABASE_CONNECTION_STRING` | No | `""` | Async SQL Server; feedback system disabled without it |
| `CORS_ORIGINS` | No | localhost ports | Comma-separated allowed origins |
| `MEMORIES_DIR` | No | `$HOME/data/memories` on Azure, else `app/memories` | Where per-user durable memory is stored. Defaults to Azure's persistent `$HOME` (survives redeploys) when `WEBSITE_HOSTNAME` is set; override to relocate. |
| `MEMORY_SELECT_THRESHOLD` | No | `8` | Above this many stored memories, recall runs a `gpt-4o-mini` relevance selector for the current question; at/below it, all memories are sent (no extra call). |
| `ENABLE_MCL_IMAGE_VALIDATION` | No | `false` | Pre-check uploaded images are MCL screens |
| `FLOW_TRACE` | No | `true` | Prints a human-readable, arrow-connected flow trace to stderr for manual testing (`app/core/flow.py`). Set `false` in production. |

## Architecture

A FastAPI agent for the MCL (Mobile Checklist) app, built in five layers (see
`PROJECT_KNOWLEDGE.md`). Retrieval is served by **Ragie** (managed RAG) — there is no
in-house vector store.

### Request flow (`app/services/chat_service.py`)

1. `POST /api/chat` → `ChatService.process_chat_request()`.
2. **Route** every message via `classify_route` (one deterministic `gpt-4o-mini` call;
   **vision-aware** — the screenshot is included when present) into one of:
   - **CHAT** → `_handle_chat` (direct reply; sees the image if attached).
   - **KNOWLEDGE** → `_handle_text_request` → `app/retrieval` (contextualize → Ragie retrieve → grounded, cited answer). **With an image** it runs `_answer_over_image` instead (`build_vision_query` → Ragie → answer over the screenshot, cited + enforced).
   - **PERSONAL** → `_handle_personal_request` → forced MCL tool call (needs a connected session); the image rides along as context.
4. Per-user **memory** is recalled once and injected into every path's system prompt.
5. **Enforcement** sanitizes answers (citations/images) and gates tool calls (deny-by-default allowlist).

### The five layers → packages

| Layer | Package | Role |
|-------|---------|------|
| 1 Instruction | `app/instructions/` | `get_system_prompt(mode, ...)` — one CORE identity + per-mode files |
| 2 Routing | `app/routing/` | `classify_route` → CHAT / KNOWLEDGE / PERSONAL (structured output) |
| 3 Memory | `app/services/memory_service.py` | per-user durable memory (`app/memories/{user_id}/`), capped recall |
| 4 Hooks/enforcement | `app/enforcement/` | citation/image sanitize + tool allowlist (deny-by-default) + audit |
| 5 Retrieval & tools | `app/retrieval/`, `app/tools.py`, `app/clients/` | Ragie retrieve+answer; MCL user tools |

### Other endpoints

- `POST /api/auth/session` — resolve identity from the MCL bearer token.
- `GET /api/ragie/image?document_id=&chunk_id=` — proxies a Ragie screenshot (auth'd) so the frontend can render it.
- `/api/memory/*` — list/get/save/store/recall/update/delete (scoped by `user_id`).
- `POST /api/vision/analyze-screenshot`, `/api/feedback`, `/health`.

### Startup sequence (`app/main.py` lifespan)

1. Validate `OPENAI_API_KEY` (raises at import if missing).
2. Create async DB tables (if DB available).
3. That's it — retrieval is Ragie; nothing to ingest locally.

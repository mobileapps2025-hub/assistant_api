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

### Knowledge base (in-house, replaces Ragie — Decision 20)

```bash
python ingest_kb.py extract --file "../docs/<guide>.pdf" ...   # cut screenshots, draft descriptions + procedures
python review_server.py                                        # human review at http://127.0.0.1:8010 (saves to kb_review/*.json)
python ingest_kb.py redescribe                                 # re-run the vision pass with the current prompt (skips what you reviewed)
python ingest_kb.py build                                      # veto gate -> app/kb_index/ (committed; deploys ship it)
cd ../codebase_agent && python research.py --gaps gaps.selected.json   # research the questions queued from the review tool
```

No ingestion runs on the server. Review state lives in `kb_review/{images.json,procedures.json}`
(commit it). **Review is currently PAUSED** — see `PROJECT_KNOWLEDGE.md` Decision 20 for where.

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | ✅ Yes | — | GPT-4o (answers, routing, vision); app refuses to start without it |
| `KB_TOP_K` | No | `6` (honors legacy `RAGIE_TOP_K`) | Units retrieved per query from the in-house index |
| `KB_INDEX_DIR` | No | `app/kb_index` | Where `index.json` + `embeddings.npz` live |
| `RAGIE_API_KEY` / `RAGIE_PARTITION` | No | `""` / `mcl_spike` | **Legacy** — only the old `/api/ragie/image` proxy (kept so old links don't 404) |
| `API_PUBLIC_URL` | No | `http://127.0.0.1:8001` | Public base URL used to build `/images/kb/...` screenshot links the frontend can load |
| `DATABASE_CONNECTION_STRING` | No | `""` | Async SQL Server; feedback system disabled without it |
| `CORS_ORIGINS` | No | localhost ports | Comma-separated allowed origins |
| `MEMORIES_DIR` | No | `$HOME/data/memories` on Azure, else `app/memories` | Where per-user durable memory is stored. Defaults to Azure's persistent `$HOME` (survives redeploys) when `WEBSITE_HOSTNAME` is set; override to relocate. |
| `MEMORY_SELECT_THRESHOLD` | No | `8` | Above this many stored memories, recall runs a `gpt-4o-mini` relevance selector for the current question; at/below it, all memories are sent (no extra call). |
| `ENABLE_MCL_IMAGE_VALIDATION` | No | `false` | Pre-check uploaded images are MCL screens |
| `GAP_INGEST_TOKEN` | No | `""` | Shared secret for `/api/gaps*` (the questions MarieClaire could not answer). Unset = those endpoints are closed. |
| `GAP_SINK_URL` | No | `""` | Set on a **local** MarieClaire to the central backend, so unanswered questions are forwarded over HTTP and no database credential lives locally. Unset = write straight to the database (the central backend). |
| `GAP_API_URL` | No | `http://127.0.0.1:8001` | Read by `review_server.py` to show the waiting questions. |
| `FLOW_TRACE` | No | `true` | Prints a human-readable, arrow-connected flow trace to stderr for manual testing (`app/core/flow.py`). Set `false` in production. |

## Architecture

A FastAPI agent for the MCL (Mobile Checklist) app, built in five layers (see
`PROJECT_KNOWLEDGE.md`). Retrieval is **in-house** (Decision 20): a committed embedding index
under `app/kb_index/` with typed units — `text`, `procedure` (whole step-by-step guides), and
`image` (approved screenshots). Screenshots reach answers only via `{{step:...}}`/`{{image:...}}`
markers the model writes; `answerer.render_markers` resolves them deterministically to approved
images (never model-written URLs).

### Request flow (`app/services/chat_service.py`)

1. `POST /api/chat` → `ChatService.process_chat_request()`.
2. **Detect language** (`detect_language`, gpt-4o-mini on the last ≤3 user messages) and **format the caller's `device`** (`{platform, form_factor, app_version}`); both feed prompt slots (`# LANGUAGE`, `# DEVICE`) on every path.
3. **Route** every message via `classify_route` (one deterministic `gpt-4o-mini` call;
   **vision-aware** — the screenshot is included when present; **capability-aware** — it receives `MCL_USER_TOOLS` so it knows the agent's real tools) into one of:
   - **CHAT** → `_handle_chat` (direct reply; also owns questions **about the assistant itself** — what it is / can do / its own prior words; carries the tool catalog so it answers from real capabilities). Sees the image if attached.
   - **KNOWLEDGE** → `_handle_text_request` → `app/retrieval` (contextualize → in-house KB retrieve → grounded, cited answer with marker-rendered screenshots). Only for questions about the **MCL product**. **With an image** it runs `_answer_over_image` instead (`build_vision_query` → KB retrieve → answer over the screenshot, cited + enforced).
   - **PERSONAL** → `_handle_personal_request` → forced MCL tool call (needs a connected session); the image rides along as context.
4. Per-user **memory** is recalled once (query-aware: a gpt-4o-mini selector narrows to relevant memories when the user has > `MEMORY_SELECT_THRESHOLD`) and injected into every path's system prompt.
5. **Enforcement** sanitizes answers (citations/images) and gates tool calls (deny-by-default allowlist).

### The five layers → packages

| Layer | Package | Role |
|-------|---------|------|
| 1 Instruction | `app/instructions/` | `get_system_prompt(mode, ...)` — one CORE identity + per-mode files |
| 2 Routing | `app/routing/` | `classify_route` (capability-aware — takes `tools_catalog`) → CHAT / KNOWLEDGE / PERSONAL; `detect_language` (structured output) |
| 3 Memory | `app/services/memory_service.py` | per-user durable memory (`app/memories/{user_id}/`), capped recall |
| 4 Hooks/enforcement | `app/enforcement/` | citation/image sanitize + tool allowlist (deny-by-default) + audit |
| 5 Retrieval & tools | `app/retrieval/`, `app/tools.py`, `app/clients/`, `ingest_kb.py`, `review_server.py` | in-house KB retrieve+answer (marker-rendered screenshots); MCL user tools |

### Other endpoints

- `POST /api/auth/session` — resolve identity from the MCL bearer token.
- `GET /images/kb/<hash>.png` — approved KB screenshots (static mount). `GET /api/ragie/image` is legacy only.
- `/api/memory/*` — list/get/save/store/recall/update/delete (scoped by `user_id`).
- `POST /api/vision/analyze-screenshot`, `/api/feedback`, `/health`.

### Startup sequence (`app/main.py` lifespan)

1. Validate `OPENAI_API_KEY` (raises at import if missing).
2. Create async DB tables (if DB available).
3. That's it — the KB index is loaded lazily from `app/kb_index/` on the first KNOWLEDGE query; nothing to ingest at startup.

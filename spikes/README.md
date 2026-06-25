# Ragie validation spike

**Purpose:** prove whether [Ragie](https://ragie.ai) is at least as good as the current
in-house RAG pipeline *before* we commit to replacing it (the future Layer 5 retrieval
module). This is **isolated and throwaway** — nothing here is imported by the running agent,
so the live app is untouched. If Ragie doesn't clear the bar, we delete this folder and keep
the current pipeline (**Plan B** — no rollback needed).

It reuses the Layer-1 instruction file (`get_system_prompt("rag")`) for answering, so the
**only** difference from the current pipeline is *where the chunks come from* — an
apples-to-apples comparison of retrieval quality.

## The corpus
The original source documents live in the repo-root **`docs/`** folder (24 PDFs, including
the image-rich `MCL Visual Guide *` / `Visual Guide * MCL APP` set and the FAQ PDFs). These
are what we upload to Ragie to test its native multimodal handling.

## Uploads are manual (by design)
You control exactly what goes into Ragie — nothing is auto-uploaded. Two options:
1. **Ragie dashboard (recommended):** upload the documents you want, add new ones anytime,
   delete/replace from their UI. Maximum control, zero code.
2. **This script, with explicit files/folder:**
   ```bash
   python spikes/ragie_spike.py upload --dir ../docs                       # all of docs/
   python spikes/ragie_spike.py upload --file "../docs/MCL Visual Guide Tasks.pdf"
   ```

> A **hidden in-app upload panel** (manage Ragie documents from your own admin area) is
> recorded as a future product feature in `PROJECT_KNOWLEDGE.md` — to design during the
> Layer 5 module, not part of this spike.

## Setup
```bash
cd assistant_api
pip install -r spikes/requirements.txt
export RAGIE_API_KEY=...        # PowerShell: $env:RAGIE_API_KEY="..."
# OPENAI_API_KEY must already be set (the app needs it anyway)
```

## Run (from assistant_api/)
```bash
python spikes/ragie_spike.py upload --dir ../docs        # or upload via the dashboard
python spikes/ragie_spike.py status                      # ingestion status
python spikes/ragie_spike.py ask "How do I export a checklist to Excel?"
python spikes/ragie_spike.py eval --out spikes/ragie_eval_report.md
```
`eval` compares against the current pipeline via `POST /api/chat` (start the backend first).
Add `--ragie-only` to skip the comparison.

## Go / no-go bar
- Retrieval returns on-topic chunks for the eval questions.
- Ragie answers are **≥** the current pipeline's on the eval set (correct, grounded, complete).
- Multimodal: image-bearing questions return useful results (watch the `image_links` flag /
  `links` field in the report).
- Latency and cost per query are acceptable.

If it passes → we design the full Layer 5 Ragie module. If not → Plan B (keep current
pipeline, proceed to Module 2 routing with the in-house RAG as the `MCL_QUERY` path).

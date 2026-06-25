#!/usr/bin/env python
"""Ragie validation spike — ISOLATED and THROWAWAY.

This does NOT touch the running agent. It exists only to answer one question: is Ragie at
least as good as the in-house RAG pipeline before we commit to replacing it? If the answer
is no, we delete this folder and keep the current pipeline (Plan B) — no rollback needed.

It deliberately reuses the Layer-1 instruction file (`get_system_prompt("rag")`) so the
ONLY thing that differs from the current pipeline is the retrieval source — making the
comparison apples-to-apples.

Subcommands
-----------
  upload   Upload PDFs/MD from one or more folders into a Ragie partition; poll until ready.
  status   List documents in the partition with their ingestion status.
  ask      Retrieve from Ragie for a single question and answer with GPT-4o (Ragie only).
  eval     Run the eval set through Ragie AND (if reachable) the current /api/chat pipeline,
           writing a side-by-side Markdown report.

Environment
-----------
  RAGIE_API_KEY    required
  OPENAI_API_KEY   required (answer generation)
  RAGIE_PARTITION  optional, default "mcl_spike"
  BACKEND_URL      optional, default http://localhost:8000 (for the side-by-side compare)

Usage (run from the assistant_api/ directory)
---------------------------------------------
  pip install -r spikes/requirements.txt
  # Uploads are MANUAL. Either upload via the Ragie dashboard, or name files explicitly:
  python spikes/ragie_spike.py upload --dir ../docs                 # the original source PDFs
  python spikes/ragie_spike.py upload --file "../docs/MCL Visual Guide Tasks.pdf"
  python spikes/ragie_spike.py status
  python spikes/ragie_spike.py ask "How do I export a checklist to Excel?"
  python spikes/ragie_spike.py eval --out spikes/ragie_eval_report.md
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make `app` importable (reuse the Layer-1 instruction file for prompt parity) and ensure
# sibling modules (eval_questions) import regardless of how the script is launched.
SELF_DIR = Path(__file__).resolve().parent
ASSISTANT_API_DIR = SELF_DIR.parent
sys.path.insert(0, str(ASSISTANT_API_DIR))
sys.path.insert(0, str(SELF_DIR))

# Windows consoles default to cp1252; chunk text contains Unicode (arrows, em-dashes, etc.).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load assistant_api/.env (OPENAI_API_KEY etc.). Does not override vars already in the
# environment, so an inline RAGIE_API_KEY=... still wins.
try:
    from dotenv import load_dotenv
    load_dotenv(ASSISTANT_API_DIR / ".env")
except ImportError:
    pass

PARTITION = os.getenv("RAGIE_PARTITION", "mcl_spike")
ANSWER_MODEL = "gpt-4o"
SUPPORTED_EXT = {".pdf", ".md", ".docx", ".txt"}
UPLOAD_RECORD = SELF_DIR / ".ragie_uploads.json"
REPO_ROOT = ASSISTANT_API_DIR.parent
ROOT_DOCS_DIR = REPO_ROOT / "docs"   # the original source PDFs (24, incl. image-rich guides)
READY_STATES = {"ready"}
TERMINAL_STATES = {"ready", "failed"}


# --------------------------------------------------------------------------- clients
def get_ragie():
    key = os.getenv("RAGIE_API_KEY")
    if not key:
        sys.exit("ERROR: RAGIE_API_KEY is not set. Create a Ragie account, then set it.")
    try:
        from ragie import Ragie
    except ImportError:
        sys.exit("ERROR: ragie SDK not installed. Run: pip install -r spikes/requirements.txt")
    return Ragie(auth=key)


def get_openai():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set.")
    from openai import OpenAI
    return OpenAI()


# --------------------------------------------------------------------------- upload
def _collect_files(dirs):
    files = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  (skip, missing) {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
                files.append(p)
    return files


def cmd_upload(args):
    # Uploads are deliberately MANUAL — no auto-batch. You name the files (or a folder).
    # Validate the selection before requiring the SDK/key, for clearer guidance.
    files = [Path(f) for f in (args.file or [])]
    for d in (args.dir or []):
        files.extend(_collect_files([d]))

    if not files:
        sys.exit(
            "Nothing to upload — uploads are manual on purpose.\n"
            "Pick one of:\n"
            "  - Upload the docs you want via the Ragie dashboard (full per-document control), or\n"
            "  - Pass explicit files / a folder, e.g.:\n"
            f'      python spikes/ragie_spike.py upload --file "{ROOT_DOCS_DIR / "MCL Visual Guide Tasks.pdf"}"\n'
            f'      python spikes/ragie_spike.py upload --dir "{ROOT_DOCS_DIR}"'
        )

    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        sys.exit("These files do not exist:\n  " + "\n  ".join(missing))

    client = get_ragie()
    print(f"Uploading {len(files)} file(s) to partition '{PARTITION}' (mode={args.mode}):")
    records = _load_records()
    new_ids = []
    for path in files:
        with open(path, "rb") as fh:
            res = client.documents.create(request={
                "file": {"file_name": path.name, "content": fh},
                "mode": args.mode,  # hi_res extracts images+tables; fast is text-only (~20x faster)
                "metadata": {"source": path.name, "spike": True},
                "partition": PARTITION,
            })
        records[res.id] = {"name": path.name, "status": res.status}
        new_ids.append(res.id)
        print(f"  + {path.name}  ->  id={res.id}  status={res.status}")
    _save_records(records)

    if not args.no_wait:
        _wait_until_ready(client, new_ids)


def _wait_until_ready(client, doc_ids, timeout_s=900, interval_s=10):
    print(f"\nWaiting for {len(doc_ids)} document(s) to finish ingestion (Ctrl+C to stop)...")
    deadline = time.time() + timeout_s
    pending = set(doc_ids)
    while pending and time.time() < deadline:
        for doc_id in list(pending):
            try:
                doc = client.documents.get(document_id=doc_id, partition=PARTITION)
            except Exception as e:  # noqa: BLE001 - spike: report and keep polling
                print(f"  ! {doc_id}: get failed ({e})")
                continue
            if doc.status in TERMINAL_STATES:
                flag = "OK" if doc.status in READY_STATES else "FAILED"
                print(f"  [{flag}] {doc_id} -> {doc.status}")
                pending.discard(doc_id)
        if pending:
            time.sleep(interval_s)
    if pending:
        print(f"\nStill not ready after {timeout_s}s: {sorted(pending)} (run `status` later).")
    else:
        print("\nAll documents reached a terminal state.")


def cmd_status(args):
    client = get_ragie()
    # SDK 2.0.0: list() -> ListDocumentsResponse(.result: DocumentList(.documents, .pagination)).
    # page_size 100 covers our corpus (~24); add pagination here if the corpus grows.
    res = client.documents.list(request={"partition": PARTITION, "page_size": 100})
    docs = res.result.documents
    for doc in docs:
        print(f"  {doc.id} — {doc.status} — {doc.name}")
    print(f"Total in partition '{PARTITION}': {len(docs)}")


# --------------------------------------------------------------------------- retrieve + answer
# Ragie free tier: 10 requests/minute. Space calls out and back off on 429.
_RAGIE_MIN_INTERVAL_S = 7.0
_last_ragie_call = [0.0]


def _throttle():
    elapsed = time.monotonic() - _last_ragie_call[0]
    if elapsed < _RAGIE_MIN_INTERVAL_S:
        time.sleep(_RAGIE_MIN_INTERVAL_S - elapsed)
    _last_ragie_call[0] = time.monotonic()


def retrieve(client, query, top_k=6, retries=4):
    from ragie import models
    for attempt in range(retries):
        _throttle()
        try:
            res = client.retrievals.retrieve(request={
                "query": query,
                "rerank": True,
                "top_k": top_k,
                "partition": PARTITION,
            })
            return res.scored_chunks
        except models.ErrorMessage as e:
            if getattr(e, "status_code", None) == 429 and attempt < retries - 1:
                print("    (rate-limited; backing off 20s)")
                time.sleep(20)
                continue
            raise


def _context_block(chunks):
    """Mirror the current pipeline's TEXTUAL CONTEXT format so the prompt is identical
    apart from where the chunks came from."""
    lines = []
    for c in chunks:
        source = getattr(c, "document_name", None) or "unknown"
        lines.append(f"[Source: {source}]: {c.text}")
    return "\n".join(lines)


def answer_with_ragie(client, oai, query, lang=None, top_k=6):
    from app.instructions import get_system_prompt

    chunks = retrieve(client, query, top_k=top_k)
    system_prompt = get_system_prompt("rag", language=lang)
    user_prompt = (
        "# TEXTUAL CONTEXT\n"
        f"{_context_block(chunks)}\n\n"
        f"User Question: {query}\n\n"
        "Answer as MarieClaire (cite sources inline with [Source: filename]):"
    )
    resp = oai.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        timeout=60,
    )
    return resp.choices[0].message.content.strip(), chunks


def _image_url(chunk):
    """Return the chunk's screenshot URL if Ragie extracted one (hi_res mode only).

    Ragie puts the retrievable image under the ``self_image`` link; it's an authenticated
    api.ragie.ai URL (needs the Bearer token to fetch), not a public image.
    """
    links = getattr(chunk, "links", None) or {}
    v = links.get("self_image") if isinstance(links, dict) else None
    return getattr(v, "href", None) if v is not None else None


def _fmt_chunks(chunks):
    out = []
    for c in chunks:
        img = _image_url(c)
        line = (
            f"    score={c.score:.3f} doc={getattr(c, 'document_name', '?')} "
            f"image={'yes' if img else 'no'}\n      {c.text[:160].strip()}..."
        )
        if img:
            line += f"\n      [image] {img}"
        out.append(line)
    return "\n".join(out)


def cmd_ask(args):
    client, oai = get_ragie(), get_openai()
    answer, chunks = answer_with_ragie(client, oai, args.query, lang=args.lang, top_k=args.top_k)
    print(f"\nQ: {args.query}\n")
    print(f"Retrieved {len(chunks)} chunk(s):")
    print(_fmt_chunks(chunks))
    print("\n--- Ragie answer ---\n")
    print(answer)


# --------------------------------------------------------------------------- current pipeline (compare)
def query_current_pipeline(query, backend_url):
    """POST the question to the running backend's /api/chat (no auth needed for MCL_QUERY).
    Best-effort: returns an error string if the backend isn't reachable."""
    payload = json.dumps({"messages": [{"role": "user", "content": query}]}).encode("utf-8")
    req = urllib.request.Request(
        f"{backend_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("response", "(no 'response' field)")
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return f"(current pipeline unavailable: {e})"


# --------------------------------------------------------------------------- eval report
def cmd_eval(args):
    from eval_questions import EVAL_QUESTIONS

    client, oai = get_ragie(), get_openai()
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    compare = not args.ragie_only

    rows = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q = item["q"]
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {q}")
        try:
            ragie_answer, chunks = answer_with_ragie(client, oai, q, lang=item.get("lang"), top_k=args.top_k)
        except Exception as e:  # noqa: BLE001 - spike: record the failure, keep going
            ragie_answer, chunks = f"(ERROR: {e})", []
            print(f"    ! {e}")
        current_answer = query_current_pipeline(q, backend_url) if compare else "(skipped)"
        rows.append({"item": item, "ragie_answer": ragie_answer, "chunks": chunks,
                     "current_answer": current_answer})

    report = _render_report(rows, compare, backend_url)
    out = Path(args.out)
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out}")


def _render_report(rows, compare, backend_url):
    lines = [
        "# Ragie spike — side-by-side evaluation",
        "",
        f"- Partition: `{PARTITION}`  |  Answer model: `{ANSWER_MODEL}`  |  rerank=on",
        f"- Compared against current pipeline: {'yes — ' + backend_url if compare else 'no (ragie-only)'}",
        "",
        "Judge each pair: is the Ragie answer at least as correct, grounded, and complete as "
        "the current one? Note any multimodal wins (image links retrieved).",
        "",
    ]
    for i, row in enumerate(rows, 1):
        item = row["item"]
        has_image = any(_image_url(c) for c in row["chunks"])
        lines += [
            f"## {i}. {item['q']}",
            f"_topic={item['topic']} · lang={item['lang']} · expects_visual={item['expects_visual']} · "
            f"ragie_chunks={len(row['chunks'])} · images={'yes' if has_image else 'no'}_",
            "",
            "**Retrieved chunks:**", "```", _fmt_chunks(row["chunks"]), "```",
            "",
            "**Ragie answer:**", "", row["ragie_answer"], "",
        ]
        if compare:
            lines += ["**Current pipeline answer:**", "", row["current_answer"], ""]
        lines += ["---", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- records helpers
def _load_records():
    if UPLOAD_RECORD.exists():
        return json.loads(UPLOAD_RECORD.read_text(encoding="utf-8"))
    return {}


def _save_records(records):
    UPLOAD_RECORD.write_text(json.dumps(records, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- cli
def main():
    parser = argparse.ArgumentParser(description="Ragie validation spike (isolated, throwaway).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upload", help="Manually upload specific files/a folder into Ragie.")
    p_up.add_argument("--file", action="append", help="A file to upload (repeatable).")
    p_up.add_argument("--dir", action="append",
                      help="A folder whose supported files to upload (repeatable).")
    p_up.add_argument("--mode", choices=["hi_res", "fast"], default="hi_res",
                      help="hi_res (default) extracts images+tables; fast is text-only (~20x faster).")
    p_up.add_argument("--no-wait", action="store_true", help="Don't poll for readiness.")
    p_up.set_defaults(func=cmd_upload)

    p_st = sub.add_parser("status", help="List documents + ingestion status in the partition.")
    p_st.set_defaults(func=cmd_status)

    p_ask = sub.add_parser("ask", help="Retrieve + answer a single question (Ragie only).")
    p_ask.add_argument("query")
    p_ask.add_argument("--lang", default=None, help="Force answer language (e.g. de).")
    p_ask.add_argument("--top-k", type=int, default=6)
    p_ask.set_defaults(func=cmd_ask)

    p_ev = sub.add_parser("eval", help="Run the eval set; write a side-by-side report.")
    p_ev.add_argument("--out", default=str(SELF_DIR / "ragie_eval_report.md"))
    p_ev.add_argument("--top-k", type=int, default=6)
    p_ev.add_argument("--ragie-only", action="store_true", help="Skip the current-pipeline compare.")
    p_ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

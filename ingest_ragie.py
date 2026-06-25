#!/usr/bin/env python
"""Re-ingest the MCL source documents into Ragie.

Run from the assistant_api/ directory. The Ragie API key is read from the environment
(RAGIE_API_KEY) — never pass it on the command line or hardcode it.

  python ingest_ragie.py                 # ingest ../docs into the mcl_docs partition (hi_res)
  python ingest_ragie.py --status        # list documents + ingestion status in the partition
  python ingest_ragie.py --purge         # delete every document in the partition first, then ingest

Why this exists: the original spike partition (mcl_spike) ended up with orphaned chunks —
the vector index still returned them but their underlying documents (and screenshots) had
been deleted, so every image link 404'd. Re-ingesting into a clean partition fixes both
text and image retrieval. After a successful run, point the backend at the new partition:
set RAGIE_PARTITION=mcl_docs (or update the default in app/core/config.py).
"""
import argparse
import os
import sys
import time
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parent
REPO_ROOT = SELF_DIR.parent
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_PARTITION = "mcl_docs"
SUPPORTED_EXT = {".pdf", ".md", ".docx", ".txt"}
TERMINAL_STATES = {"ready", "failed"}
READY_STATES = {"ready"}

# Ragie free tier allows 10 requests/minute; space calls out and back off on 429.
MIN_REQUEST_INTERVAL_S = 7.0
RATE_LIMIT_BACKOFF_S = 20.0
_last_request_at = [0.0]


def _reconfigure_console_for_unicode():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv(SELF_DIR / ".env")
    except ImportError:
        pass


def _require_api_key() -> str:
    key = os.getenv("RAGIE_API_KEY")
    if not key:
        sys.exit("ERROR: RAGIE_API_KEY is not set in the environment.")
    return key


def _ragie_client():
    try:
        from ragie import Ragie
    except ImportError:
        sys.exit("ERROR: ragie SDK not installed. Run: pip install -r requirements.txt")
    return Ragie(auth=_require_api_key())


def _throttle():
    elapsed = time.monotonic() - _last_request_at[0]
    if elapsed < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
    _last_request_at[0] = time.monotonic()


def _is_rate_limited(error) -> bool:
    return getattr(error, "status_code", None) == 429


def _collect_documents(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        sys.exit(f"ERROR: docs directory not found: {docs_dir}")
    return sorted(
        p for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def _list_documents(client, partition: str) -> list:
    res = client.documents.list(request={"partition": partition, "page_size": 100})
    return list(res.result.documents)


def _upload_one(client, path: Path, partition: str, mode: str) -> str:
    with open(path, "rb") as fh:
        res = client.documents.create(request={
            "file": {"file_name": path.name, "content": fh},
            "mode": mode,
            "metadata": {"source": path.name},
            "partition": partition,
        })
    return res.id


def _delete_one(client, document_id: str, partition: str):
    client.documents.delete(document_id=document_id, partition=partition)


def _wait_until_ready(client, doc_ids: list[str], partition: str,
                      timeout_s: int = 1800, interval_s: int = 10):
    print(f"\nWaiting for {len(doc_ids)} document(s) to finish ingestion (Ctrl+C to stop)...")
    deadline = time.monotonic() + timeout_s
    pending = set(doc_ids)
    while pending and time.monotonic() < deadline:
        for doc_id in list(pending):
            try:
                doc = client.documents.get(document_id=doc_id, partition=partition)
            except Exception as e:
                print(f"  ! {doc_id}: status check failed ({e})")
                continue
            if doc.status in TERMINAL_STATES:
                flag = "OK" if doc.status in READY_STATES else "FAILED"
                print(f"  [{flag}] {doc_id} -> {doc.status}")
                pending.discard(doc_id)
        if pending:
            time.sleep(interval_s)
    if pending:
        print(f"\nStill not ready after {timeout_s}s: {sorted(pending)} — re-run with --status later.")
    else:
        print("\nAll documents reached a terminal state.")


def _purge_partition(client, partition: str):
    docs = _list_documents(client, partition)
    if not docs:
        print(f"Partition '{partition}' is already empty — nothing to purge.")
        return
    print(f"Purging {len(docs)} document(s) from partition '{partition}':")
    for doc in docs:
        _throttle()
        try:
            _delete_one(client, doc.id, partition)
            print(f"  - deleted {doc.id}  ({doc.name})")
        except Exception as e:
            if _is_rate_limited(e):
                time.sleep(RATE_LIMIT_BACKOFF_S)
                _delete_one(client, doc.id, partition)
                print(f"  - deleted {doc.id}  ({doc.name})  [after backoff]")
            else:
                print(f"  ! failed to delete {doc.id}: {e}")


def _ingest(client, files: list[Path], partition: str, mode: str) -> list[str]:
    print(f"Ingesting {len(files)} file(s) into partition '{partition}' (mode={mode}):")
    uploaded = []
    for path in files:
        _throttle()
        try:
            doc_id = _upload_one(client, path, partition, mode)
        except Exception as e:
            if _is_rate_limited(e):
                time.sleep(RATE_LIMIT_BACKOFF_S)
                doc_id = _upload_one(client, path, partition, mode)
            else:
                print(f"  ! failed to upload {path.name}: {e}")
                continue
        uploaded.append(doc_id)
        print(f"  + {path.name}  ->  id={doc_id}")
    return uploaded


def cmd_status(client, partition: str):
    docs = _list_documents(client, partition)
    for doc in docs:
        print(f"  {doc.id} — {doc.status} — {doc.name}")
    print(f"Total in partition '{partition}': {len(docs)}")


def cmd_ingest(args, client):
    if args.purge:
        _purge_partition(client, args.partition)
    files = _collect_documents(Path(args.docs_dir))
    if not files:
        sys.exit(f"No supported documents found in {args.docs_dir}")
    uploaded = _ingest(client, files, args.partition, args.mode)
    if uploaded and not args.no_wait:
        _wait_until_ready(client, uploaded, args.partition)
    print(
        f"\nDone. To serve answers from this partition, set "
        f"RAGIE_PARTITION={args.partition} in the backend environment."
    )


def main():
    _reconfigure_console_for_unicode()
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Re-ingest MCL documents into Ragie.")
    parser.add_argument("--partition", default=os.getenv("RAGIE_PARTITION", DEFAULT_PARTITION),
                        help=f"Target Ragie partition (default {DEFAULT_PARTITION}).")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR),
                        help="Folder of source documents to ingest.")
    parser.add_argument("--mode", choices=["hi_res", "fast"], default="hi_res",
                        help="hi_res (default) extracts images+tables; fast is text-only.")
    parser.add_argument("--purge", action="store_true",
                        help="Delete every document in the partition before ingesting.")
    parser.add_argument("--no-wait", action="store_true", help="Don't poll for readiness.")
    parser.add_argument("--status", action="store_true",
                        help="Only list documents + status in the partition, then exit.")
    args = parser.parse_args()

    client = _ragie_client()
    if args.status:
        cmd_status(client, args.partition)
    else:
        cmd_ingest(args, client)


if __name__ == "__main__":
    main()

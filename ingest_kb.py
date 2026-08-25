"""Build the in-house KB index — screenshots structurally linked to procedure steps, behind a
human review gate.

    cd assistant_api
    python ingest_kb.py extract --file "../docs/Visual Guide Tasks MCL APP.pdf" [...]
    python review_server.py            # review at http://127.0.0.1:8010 (saves directly)
    python ingest_kb.py build

extract  Walks each PDF page in reading order (text + image blocks). Every real screenshot
         (icons filtered by size) becomes an *occurrence* record in kb_review/images.json
         (file saved once by content hash under static/images/kb/) with an AI-drafted
         semantic_description (retrieval/reasoning) and alt_text (user-facing). Each page is
         also read by the model to draft *procedures* (title + ordered steps + which screenshot
         belongs to which step) into kb_review/procedures.json. Raw page text (with image
         placeholders) goes to kb_review/extracted.json. Existing review work is never redone.
build    Strict gate. Index units:
           text       raw chunks; approved+safe images referenced as "[Figure <id>: alt]"
           procedure  approved procedures stored WHOLE, steps keep only approved+safe images
           image      every approved+safe image, embedded on its semantic_description
         Writes app/kb_index/{index.json,embeddings.npz}; deploys ship them.
"""
import argparse
import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import fitz
import numpy as np
from openai import RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.core.config import client  # noqa: E402

SELF_DIR = Path(__file__).resolve().parent
INDEX_DIR = SELF_DIR / "app" / "kb_index"
IMAGES_DIR = SELF_DIR / "static" / "images" / "kb"
REVIEW_DIR = SELF_DIR / "kb_review"
EXTRACTED_FILE = REVIEW_DIR / "extracted.json"
IMAGES_FILE = REVIEW_DIR / "images.json"
PROCEDURES_FILE = REVIEW_DIR / "procedures.json"

EMBED_MODEL = "text-embedding-3-small"
DRAFT_MODEL = "gpt-4o-mini"
MIN_IMAGE_PIXELS = 40_000
PLACEHOLDER = "⟦img:{}⟧"
PLACEHOLDER_RE = re.compile(r"⟦img:([a-z0-9-]+_p\d+_i\d+)⟧")
QUESTION_HEADING = re.compile(r"^\s*(?:\d+[.)]\s+)?[^\n]{8,120}\?\s*$", re.MULTILINE)
MIN_QUESTION_SPLITS = 2
CONTEXT_CHARS = 220

IMAGE_PROMPT = (
    "This screenshot comes from a user guide for MCL (Mobile Checklist), a retail checklist app "
    "(mobile app + web dashboard).\n"
    "ANNOTATION CONVENTION: a red rectangle/box/outline drawn on the screenshot is NOT part of the "
    "app UI — the guide author drew it to point at the element the reader must look at or tap. "
    "Treat whatever is inside the red box as the subject of the screenshot: name that element "
    "precisely (button label, menu item, field, icon) and describe the action it illustrates. "
    "Mention the rest of the screen only as context.\n"
    "Return JSON with two fields:\n"
    '"semantic_description": 2-3 sentences for search and reasoning — which screen this is, the '
    "highlighted element and what tapping/using it does, plus the other visible controls.\n"
    '"alt_text": one short sentence a user would read under the image, centered on the highlighted '
    "element (e.g. \"The Filter icon at the top of the Tasks list\").\n"
    "Be concrete; name UI elements. Do not invent what isn't visible."
)
PROCEDURE_PROMPT = (
    "You read one page of an MCL (Mobile Checklist) user guide. Screenshot positions are marked "
    "as ⟦img:ID⟧. Identify every step-by-step PROCEDURE the page explains (how to do something). "
    "For each, return the ordered steps in plain imperative sentences and attach to each step the "
    "IDs of the screenshots that illustrate exactly that step (usually the one right after it; "
    "a step may have none). Return JSON: {\"procedures\": [{\"procedure_id\": \"snake_case\", "
    "\"title\": \"...\", \"steps\": [{\"text\": \"...\", \"image_ids\": [\"...\"]}]}]}. "
    "If the page explains no procedure, return {\"procedures\": []}. Never invent steps."
)


# ----------------------------------------------------------------------------- helpers

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _ask_json(content) -> dict:
    # Screenshots are token-heavy; the org TPM limit trips easily -> back off and retry.
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=DRAFT_MODEL,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=90,
            )
            break
        except RateLimitError:
            wait = 5 * (attempt + 1)
            print(f"  rate limited — retrying in {wait}s")
            time.sleep(wait)
    else:
        raise SystemExit("OpenAI rate limit persisted after retries — re-run extract later (progress is saved).")
    try:
        return json.loads(response.choices[0].message.content or "{}")
    except ValueError:
        return {}


def _split_questions(text: str) -> list[str]:
    starts = [m.start() for m in QUESTION_HEADING.finditer(text)]
    if len(starts) < MIN_QUESTION_SPLITS:
        return [text]
    bounds = ([0] if starts[0] > 0 else []) + starts + [len(text)]
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]


# ----------------------------------------------------------------------------- extract

def _to_png(image_bytes: bytes) -> bytes:
    pix = fitz.Pixmap(image_bytes)
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png")


def _block_text(block: dict) -> str:
    return "\n".join(
        "".join(span["text"] for span in line["spans"]).rstrip() for line in block.get("lines", [])
    ).strip()


def _draft_image(png: bytes) -> dict:
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    drafted = _ask_json([
        {"type": "text", "text": IMAGE_PROMPT},
        # high: full-resolution understanding — menu labels and thin highlight outlines matter here
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
    ])
    return {
        "semantic_description": drafted.get("semantic_description", "").strip(),
        "alt_text": drafted.get("alt_text", "").strip(),
    }


def _new_image_record(image_id, png, doc_name, page_number, blocks, i) -> dict:
    digest = hashlib.sha256(png).hexdigest()[:12]
    (IMAGES_DIR / f"{digest}.png").write_bytes(png)
    before = " ".join(_block_text(b) for b in blocks[max(0, i - 2):i] if b["type"] == 0)
    after = " ".join(_block_text(b) for b in blocks[i + 1:i + 3] if b["type"] == 0)
    return {
        "image_id": image_id, "hash": digest, "path": f"kb/{digest}.png",
        "document": doc_name, "page": page_number,
        "context_before": before[-CONTEXT_CHARS:], "context_after": after[:CONTEXT_CHARS],
        **_draft_image(png),
        "keep": True, "safe_to_show": True, "approved": False,
    }


def _walk_page(page, doc_slug, doc_name, page_number, images) -> tuple[str, int]:
    """Page text in reading order with a placeholder per screenshot; registers new images."""
    parts, new, k = [], 0, 0
    blocks = page.get_text("dict")["blocks"]
    for i, block in enumerate(blocks):
        if block["type"] == 0:
            if text := _block_text(block):
                parts.append(text)
            continue
        if block.get("width", 0) * block.get("height", 0) < MIN_IMAGE_PIXELS:
            continue
        k += 1
        image_id = f"{doc_slug}_p{page_number}_i{k}"
        parts.append(PLACEHOLDER.format(image_id))
        if image_id not in images:
            images[image_id] = _new_image_record(image_id, _to_png(block["image"]), doc_name, page_number, blocks, i)
            new += 1
            print(f"  new image {image_id}: {images[image_id]['alt_text'][:70]}")
    return "\n".join(parts).strip(), new


def _draft_procedures(text: str, doc_slug: str, page_number: int, doc_name: str, procedures: dict) -> int:
    page_image_ids = set(PLACEHOLDER_RE.findall(text))
    if not page_image_ids and len(text) < 200:
        return 0
    new = 0
    for proc in _ask_json(f"{PROCEDURE_PROMPT}\n\nPAGE TEXT:\n{text}").get("procedures", []):
        procedure_id = f"{doc_slug}_{_slug(proc.get('procedure_id') or proc.get('title', 'procedure'))}"
        if procedure_id in procedures:
            continue
        steps = [
            {"text": s.get("text", "").strip(),
             "image_ids": [i for i in s.get("image_ids", []) if i in page_image_ids]}
            for s in proc.get("steps", []) if s.get("text", "").strip()
        ]
        if not steps:
            continue
        procedures[procedure_id] = {
            "procedure_id": procedure_id, "title": proc.get("title", "").strip(),
            "document": doc_name, "page": page_number, "steps": steps, "approved": False,
        }
        new += 1
        print(f"  new procedure {procedure_id}: {len(steps)} step(s)")
    return new


def _extract_pdf(pdf_path: Path, images: dict, procedures: dict) -> list[dict]:
    chunks = []
    doc_slug = _slug(pdf_path.stem)
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text, _ = _walk_page(page, doc_slug, pdf_path.name, page_number, images)
            if not text:
                continue
            _draft_procedures(text, doc_slug, page_number, pdf_path.name, procedures)
            for piece_number, piece in enumerate(_split_questions(text), start=1):
                if PLACEHOLDER_RE.sub("", piece).strip():
                    chunks.append({"id": f"{doc_slug}_p{page_number}_c{piece_number}",
                                   "document_name": pdf_path.name, "text": piece})
    return chunks


def cmd_extract(args) -> None:
    pdfs = [Path(p) for p in args.file if Path(p).exists()]
    if not pdfs:
        sys.exit("No existing PDF given via --file")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    images, procedures = _load_json(IMAGES_FILE, {}), _load_json(PROCEDURES_FILE, {})
    extracted = {c["id"]: c for c in _load_json(EXTRACTED_FILE, [])}

    for pdf in pdfs:
        chunks = _extract_pdf(pdf, images, procedures)
        extracted = {**{k: v for k, v in extracted.items() if v["document_name"] != pdf.name},
                     **{c["id"]: c for c in chunks}}
        print(f"{pdf.name}: {len(chunks)} text chunk(s)")
        # persist after every PDF so a crash (rate limit, network) never loses drafted work
        _save_json(EXTRACTED_FILE, list(extracted.values()))
        _save_json(IMAGES_FILE, images)
        _save_json(PROCEDURES_FILE, procedures)
    pending_img = sum(1 for i in images.values() if not i["approved"])
    pending_proc = sum(1 for p in procedures.values() if not p["approved"])
    print(f"\n{len(images)} image(s) ({pending_img} awaiting review), "
          f"{len(procedures)} procedure(s) ({pending_proc} awaiting review)")
    print("Next: python review_server.py  ->  then: python ingest_kb.py build")


# ----------------------------------------------------------------------------- build

def _showable(images: dict) -> dict:
    return {i: rec for i, rec in images.items() if rec["approved"] and rec["keep"] and rec["safe_to_show"]}


def _image_ref(rec: dict) -> dict:
    return {"id": rec["image_id"], "path": rec["path"], "alt": rec["alt_text"]}


def _text_units(extracted: list[dict], showable: dict) -> list[dict]:
    units = []
    for chunk in extracted:
        refs = []

        def replace(match):
            rec = showable.get(match.group(1))
            if not rec:
                return ""
            refs.append(_image_ref(rec))
            return f"[Figure {rec['image_id']}: {rec['alt_text']}]"

        text = re.sub(r"\n{3,}", "\n\n", PLACEHOLDER_RE.sub(replace, chunk["text"])).strip()
        if text:
            units.append({"kind": "text", "id": chunk["id"], "document_name": chunk["document_name"],
                          "text": text, "images": refs, "steps": []})
    return units


def _procedure_units(procedures: dict, showable: dict) -> list[dict]:
    units = []
    for proc in procedures.values():
        if not proc["approved"]:
            continue
        steps = [{"text": s["text"], "images": [_image_ref(showable[i]) for i in s["image_ids"] if i in showable]}
                 for s in proc["steps"]]
        text = f"Procedure: {proc['title']}\n" + "\n".join(f"{n}. {s['text']}" for n, s in enumerate(steps, 1))
        units.append({"kind": "procedure", "id": proc["procedure_id"], "document_name": proc["document"],
                      "text": text, "title": proc["title"], "steps": steps,
                      "images": [img for s in steps for img in s["images"]]})
    return units


def _image_units(showable: dict) -> list[dict]:
    return [{"kind": "image", "id": rec["image_id"], "document_name": rec["document"],
             "text": rec["semantic_description"] or rec["alt_text"], "images": [_image_ref(rec)], "steps": []}
            for rec in showable.values()]


def _embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), 512):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[start:start + 512])
        vectors.extend(item.embedding for item in response.data)
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def cmd_build(args) -> None:
    if not EXTRACTED_FILE.exists():
        sys.exit("Nothing extracted yet — run: python ingest_kb.py extract --file ...")
    images, procedures = _load_json(IMAGES_FILE, {}), _load_json(PROCEDURES_FILE, {})
    showable = _showable(images)
    units = _text_units(_load_json(EXTRACTED_FILE, []), showable) + \
        _procedure_units(procedures, showable) + _image_units(showable)

    excluded_img = len(images) - len(showable)
    excluded_proc = sum(1 for p in procedures.values() if not p["approved"])
    if excluded_img or excluded_proc:
        print(f"Strict gate: {excluded_img} image(s) and {excluded_proc} procedure(s) not approved -> excluded")

    print(f"Embedding {len(units)} units with {EMBED_MODEL}...")
    embeddings = _embed([u["text"] for u in units])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(INDEX_DIR / "index.json", {"model": EMBED_MODEL, "units": units})
    np.savez_compressed(INDEX_DIR / "embeddings.npz", embeddings=embeddings)
    kinds = {k: sum(1 for u in units if u["kind"] == k) for k in ("text", "procedure", "image")}
    print(f"Wrote {len(units)} units {kinds} -> {INDEX_DIR}")


# ----------------------------------------------------------------------------- cli

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--file", nargs="+", required=True, help="PDF(s) to extract")
    extract.set_defaults(func=cmd_extract)
    sub.add_parser("build").set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

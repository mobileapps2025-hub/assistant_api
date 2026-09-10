"""Local, offline review tool for KB screenshots and procedures.

    cd assistant_api && python review_server.py      # then open http://127.0.0.1:8010

Edits save straight into kb_review/images.json and kb_review/procedures.json (no
download-and-replace). Binds to localhost only. After reviewing: python ingest_kb.py build
"""
import json
import os
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from datetime import datetime, timezone

load_dotenv(Path(__file__).resolve().parent / ".env")

SELF_DIR = Path(__file__).resolve().parent
REVIEW_DIR = SELF_DIR / "kb_review"
IMAGES_FILE = REVIEW_DIR / "images.json"
PROCEDURES_FILE = REVIEW_DIR / "procedures.json"
DOCS_DIR = REVIEW_DIR / "docs"          # one folder per researched gap: draft.md, evidence.json, report.md
DISCARDED_DIR = REVIEW_DIR / "discarded"
DISCARDED_GUIDES = REVIEW_DIR / "discarded_guides.json"
IMAGES_ROOT = SELF_DIR / "static" / "images"
DOC_DECISIONS = ("pending", "approved", "rejected", "needs_product_owner")

# The central list of questions MarieClaire could not answer. Read through the backend's
# protected endpoint, so no database credential lives here either.
GAP_API_URL = os.getenv("GAP_API_URL", "http://127.0.0.1:8001").rstrip("/")
GAP_INGEST_TOKEN = os.getenv("GAP_INGEST_TOKEN", "").strip()
SELECTED_GAPS_FILE = SELF_DIR.parent / "codebase_agent" / "gaps.selected.json"

app = FastAPI(title="KB review")
app.mount("/images", StaticFiles(directory=SELF_DIR / "static" / "images"), name="images")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


@app.get("/api/state")
def state():
    return {"images": _load(IMAGES_FILE), "procedures": _load(PROCEDURES_FILE)}


@app.put("/api/images/{image_id}")
def save_image(image_id: str, body: dict):
    images = _load(IMAGES_FILE)
    if image_id not in images:
        raise HTTPException(404)
    editable = ("semantic_description", "alt_text", "keep", "safe_to_show", "approved")
    images[image_id] = {**images[image_id], **{k: body[k] for k in editable if k in body}}
    _save(IMAGES_FILE, images)
    return images[image_id]


@app.post("/api/images/discard")
def discard_images(body: dict):
    """Throw pictures out of the process: the file moves to kb_review/discarded/, every review
    entry sharing that file is removed, and procedure steps stop pointing at it."""
    images = _load(IMAGES_FILE)
    doomed_paths = {images[i]["path"] for i in body.get("image_ids", []) if i in images}
    if not doomed_paths:
        raise HTTPException(400, "no known images to discard")
    doomed_ids = {i for i, rec in images.items() if rec["path"] in doomed_paths}

    DISCARDED_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in doomed_paths:
        source = IMAGES_ROOT / path
        if source.exists():
            source.replace(DISCARDED_DIR / Path(path).name)
            moved += 1

    _save(IMAGES_FILE, {i: rec for i, rec in images.items() if i not in doomed_ids})

    procedures = _load(PROCEDURES_FILE)
    for procedure in procedures.values():
        for step in procedure.get("steps", []):
            step["image_ids"] = [i for i in step.get("image_ids", []) if i not in doomed_ids]
    _save(PROCEDURES_FILE, procedures)

    return {"files": moved, "entries": len(doomed_ids)}


@app.post("/api/procedures/discard")
def discard_procedures(body: dict):
    """Throw step-by-step guides out of the process. The removed guides are appended to
    kb_review/discarded_guides.json so a mis-click is recoverable."""
    procedures = _load(PROCEDURES_FILE)
    doomed = [i for i in body.get("procedure_ids", []) if i in procedures]
    if not doomed:
        raise HTTPException(400, "no known guides to discard")
    graveyard = _load(DISCARDED_GUIDES)
    graveyard.update({i: procedures.pop(i) for i in doomed})
    _save(DISCARDED_GUIDES, graveyard)
    _save(PROCEDURES_FILE, procedures)
    return {"removed": len(doomed), "left": len(procedures)}


@app.put("/api/procedures/{procedure_id}")
def save_procedure(procedure_id: str, body: dict):
    procedures = _load(PROCEDURES_FILE)
    if procedure_id not in procedures:
        raise HTTPException(404)
    editable = ("title", "steps", "approved")
    procedures[procedure_id] = {**procedures[procedure_id], **{k: body[k] for k in editable if k in body}}
    _save(PROCEDURES_FILE, procedures)
    return procedures[procedure_id]


@app.post("/api/procedures")
def create_procedure(body: dict):
    procedures = _load(PROCEDURES_FILE)
    procedure_id = body.get("procedure_id", "").strip()
    if not procedure_id or procedure_id in procedures:
        raise HTTPException(400, "procedure_id missing or already exists")
    procedures[procedure_id] = {
        "procedure_id": procedure_id, "title": body.get("title", ""), "document": body.get("document", ""),
        "page": body.get("page", 0), "steps": [{"text": "", "image_ids": []}], "approved": False,
    }
    _save(PROCEDURES_FILE, procedures)
    return procedures[procedure_id]


@app.delete("/api/procedures/{procedure_id}")
def delete_procedure(procedure_id: str):
    procedures = _load(PROCEDURES_FILE)
    procedures.pop(procedure_id, None)
    _save(PROCEDURES_FILE, procedures)
    return {"ok": True}


# --- Docs: drafts researched by the codebase worker (codebase_agent/research.py) ---

def _doc_dirs():
    return sorted(d for d in DOCS_DIR.iterdir() if (d / "evidence.json").exists()) if DOCS_DIR.exists() else []


def _load_doc(folder: Path) -> dict:
    evidence = json.loads((folder / "evidence.json").read_text(encoding="utf-8"))
    draft = (folder / "draft.md").read_text(encoding="utf-8") if (folder / "draft.md").exists() else ""
    return {**evidence, "draft_md": draft, "has_report": (folder / "report.md").exists()}


@app.get("/api/docs")
def list_docs():
    return {d.name: _load_doc(d) for d in _doc_dirs()}


@app.get("/api/docs/{gap_id}/report")
def doc_report(gap_id: str):
    path = DOCS_DIR / gap_id / "report.md"
    if not path.exists():
        raise HTTPException(404)
    return {"report": path.read_text(encoding="utf-8")}


@app.put("/api/docs/{gap_id}")
def save_doc(gap_id: str, body: dict):
    folder = DOCS_DIR / gap_id
    if not (folder / "evidence.json").exists():
        raise HTTPException(404)
    if body.get("decision") not in DOC_DECISIONS:
        raise HTTPException(400, f"decision must be one of {DOC_DECISIONS}")
    evidence = json.loads((folder / "evidence.json").read_text(encoding="utf-8"))
    evidence["review"] = {
        "decision": body["decision"],
        "reviewed_by": (body.get("reviewed_by") or "").strip() or None,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": body.get("notes", ""),
    }
    (folder / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    if "draft_md" in body:
        (folder / "draft.md").write_text(body["draft_md"], encoding="utf-8")
    return _load_doc(folder)



# --- Questions: the unanswered questions waiting to be researched ---

def _gap_headers() -> dict:
    return {"Authorization": f"Bearer {GAP_INGEST_TOKEN}"} if GAP_INGEST_TOKEN else {}


@app.get("/api/questions")
def list_questions():
    """Proxy the central list. Returns an explanation rather than failing when unreachable."""
    if not GAP_INGEST_TOKEN:
        return {"gaps": [], "problem": "Set GAP_INGEST_TOKEN (and GAP_API_URL) to read the central list."}
    try:
        with httpx.Client(timeout=10) as http:
            response = http.get(f"{GAP_API_URL}/api/gaps", headers=_gap_headers())
        if response.status_code >= 400:
            return {"gaps": [], "problem": f"{GAP_API_URL} answered {response.status_code}."}
        return {"gaps": response.json().get("gaps", []), "problem": None}
    except Exception as err:
        return {"gaps": [], "problem": f"Could not reach {GAP_API_URL}: {err}"}


@app.post("/api/questions/{gap_id}/discard")
def discard_question(gap_id: int):
    with httpx.Client(timeout=10) as http:
        response = http.patch(f"{GAP_API_URL}/api/gaps/{gap_id}",
                              json={"status": "discarded"}, headers=_gap_headers())
    if response.status_code >= 400:
        raise HTTPException(response.status_code, response.text[:200])
    return {"ok": True}


@app.post("/api/questions/select")
def select_questions(body: dict):
    """Queue the chosen questions for the local codebase agent. Nothing runs automatically —
    this only writes the list, and the reviewer starts the run from their own terminal."""
    chosen = body.get("gaps") or []
    if not chosen:
        raise HTTPException(400, "no questions selected")
    queued = [
        {"id": f"gap-{g['id']}", "question": g["question"], "language": g.get("language") or "English"}
        for g in chosen if g.get("question")
    ]
    SELECTED_GAPS_FILE.write_text(json.dumps(queued, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"queued": len(queued), "file": str(SELECTED_GAPS_FILE),
            "command": f"cd codebase_agent && python research.py --gaps gaps.selected.json --max {len(queued)}"}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>KB review</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f5;color:#111}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:1}
 header b{font-size:15px} .tab{cursor:pointer;padding:6px 10px;border-radius:6px} .tab.on{background:#0f766e;color:#fff}
 .card{display:grid;grid-template-columns:minmax(300px,46%) 1fr;gap:16px;background:#fff;margin:14px 16px;padding:14px;border-radius:8px;border:2px solid #e5e5e5}
 .card.ok{border-color:#0f766e} .card.off{opacity:.55}
 img{max-width:100%;border:1px solid #ddd;border-radius:4px}
 .src{font-size:12px;color:#666;margin-bottom:6px} .ctx{font-size:12px;color:#555;margin:6px 0;white-space:pre-wrap}
 textarea,input[type=text]{width:100%;font:inherit;padding:8px;border:1px solid #ccc;border-radius:6px;box-sizing:border-box}
 textarea{min-height:64px} label{display:inline-block;margin:8px 14px 0 0;font-weight:600}
 .flags{margin-top:6px} .saved{color:#0f766e;font-size:12px;margin-left:8px}
 .proc{background:#fff;margin:14px 16px;padding:14px;border-radius:8px;border:2px solid #e5e5e5} .proc.ok{border-color:#0f766e}
 .step{display:grid;grid-template-columns:28px 1fr 260px;gap:10px;align-items:start;margin:8px 0}
 .step .n{font-weight:700;padding-top:8px} .thumbs img{max-height:70px;margin:2px;border-radius:3px}
 button{padding:6px 12px;border:0;border-radius:6px;background:#0f766e;color:#fff;font-weight:600;cursor:pointer} button.warn{background:#b91c1c}
 h4{margin:4px 0}
 .doc{background:#fff;margin:14px 16px;padding:14px;border-radius:8px;border:2px solid #e5e5e5}
 .doc.approved{border-color:#0f766e} .doc.rejected{border-color:#b91c1c;opacity:.7} .doc.needs_product_owner{border-color:#d97706}
 .doc .q{font-size:15px;font-weight:600;margin:4px 0 8px} .doc .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .doc textarea.draft{min-height:260px;font-family:ui-monospace,Consolas,monospace;font-size:13px}
 .ev{font-size:13px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;max-height:420px;overflow:auto}
 .ev ul{margin:4px 0 8px 18px;padding:0} .ev code{font-size:12px;background:#eef2f7;padding:1px 4px;border-radius:3px}
 .badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;background:#e2e8f0;margin-right:6px}
 .badge.partial{background:#fef3c7} .badge.documented{background:#d1fae5} .badge.not_documentable,.badge.needs_product_owner{background:#fee2e2}
 select{font:inherit;padding:6px;border:1px solid #ccc;border-radius:6px}
 .picbar{position:sticky;top:44px;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;display:flex;gap:12px;align-items:center;z-index:1}
 .picbar .hint{font-size:13px;color:#555}
 .picgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--tw,200px),1fr));gap:10px;padding:10px 16px}
 .pic{position:relative;background:#fff;border:2px solid #e5e5e5;border-radius:8px;overflow:hidden;cursor:pointer;user-select:none}
 .pic img{display:block;width:100%;height:var(--th,130px);object-fit:cover;object-position:top center;
   border:0;border-radius:0;border-bottom:1px solid #eee;background:#fafafa}
 .pic .cap{padding:6px 8px 2px;font-size:11.5px;line-height:1.35;color:#333;
   display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
 .pic .where{padding:0 8px 7px;font-size:10.5px;color:#999}
 .pic .zoom{position:absolute;top:5px;right:5px;background:rgba(255,255,255,.94);border:1px solid #ccc;border-radius:6px;
   width:24px;height:24px;line-height:22px;text-align:center;text-decoration:none;font-size:13px;color:#333;opacity:0;transition:opacity .12s}
 .pic:hover .zoom{opacity:1}
 .pic.marked{border-color:#b91c1c;background:#fff1f0}
 .pic.marked img{opacity:.35}
 .pic.marked::after{content:"✕";position:absolute;top:calc(var(--th,130px)/2);left:50%;transform:translate(-50%,-50%);
   font-size:44px;color:#b91c1c;font-weight:700;text-shadow:0 0 12px #fff}
 .pic .dup{display:inline-block;font-size:10px;background:#fef3c7;color:#8a6d00;padding:1px 5px;border-radius:9px;margin-left:4px}
 .docgroup{margin:18px 16px 0;font-size:13.5px;color:#444;display:flex;align-items:center;gap:10px}
 .docgroup .mini{font:inherit;font-size:12px;padding:2px 9px;background:#fff;color:#0f766e;border:1px solid #b9ddd9;border-radius:99px;cursor:pointer}
 .q{background:#fff;border:2px solid #e5e5e5;border-radius:8px;padding:11px 13px;margin:8px 16px;
    display:grid;grid-template-columns:26px 1fr auto;gap:11px;align-items:center}
 .q.picked{border-color:#0f766e;background:#f2fbf9}
 .qtext{font-size:14px}
 .qmeta{font-size:11.5px;color:#999;margin-top:3px}
 .qtimes{font-size:11px;font-weight:700;padding:3px 9px;border-radius:99px;background:#f1f1f3;color:#555;white-space:nowrap}
 .qtimes.hot{background:#fde8c8;color:#8a4d00}
 .qcmd{margin:10px 16px;padding:11px 13px;background:#0f2320;color:#d7f5ee;border-radius:8px;
       font-family:ui-monospace,Consolas,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-all}
 .warnbox{margin:10px 16px;padding:11px 13px;background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;font-size:13px}
 .picbar select{font:inherit;padding:4px 8px}
 .glist{padding:10px 16px;display:grid;gap:8px}
 .g{background:#fff;border:2px solid #e5e5e5;border-radius:8px;padding:10px 12px;cursor:pointer;user-select:none;
    display:grid;grid-template-columns:1fr auto;gap:10px;align-items:start}
 .g:hover{border-color:#cfcfcf}
 .g.marked{border-color:#b91c1c;background:#fff1f0}
 .g.marked .gtitle{text-decoration:line-through;color:#b91c1c}
 .gtitle{font-weight:600;font-size:14px}
 .gsrc{font-size:11.5px;color:#999;margin-top:2px}
 .gsteps{margin:7px 0 0;padding-left:20px;font-size:12.5px;color:#444}
 .gsteps li{margin:2px 0}
 .gsteps .has{color:#0f766e;font-size:11px;margin-left:5px}
 .gtags{display:flex;flex-direction:column;gap:4px;align-items:flex-end;min-width:96px}
 .t{font-size:10.5px;padding:2px 7px;border-radius:99px;white-space:nowrap}
 .t.one{background:#fdf3d3;color:#8a6d00} .t.noimg{background:#f1f1f3;color:#666}
 .t.img{background:#e6f4ec;color:#1a7f4b} .t.many{background:#eceaff;color:#4b3ecc}
 .gthumbs{display:flex;gap:3px;flex-wrap:wrap;justify-content:flex-end;margin-top:4px}
 .gthumbs img{height:34px;width:auto;border:1px solid #ddd;border-radius:3px;object-fit:cover;object-position:top}
</style></head><body>
<header><b>KB review</b>
 <span class="tab" data-t="pictures" onclick="show('pictures')">Pictures <span id="qc"></span></span>
 <span class="tab on" data-t="images" onclick="show('images')">Images <span id="ic"></span></span>
 <span class="tab" data-t="questions" onclick="show('questions')">Questions <span id="qsc"></span></span>
 <span class="tab" data-t="guides" onclick="show('guides')">Guides <span id="gc"></span></span>
 <span class="tab" data-t="procedures" onclick="show('procedures')">Procedures <span id="pc"></span></span>
 <span class="tab" data-t="docs" onclick="show('docs')">Docs <span id="dc"></span></span>
 <button onclick="newProc()">New procedure</button>
 <span style="margin-left:auto;font-size:13px">Reviewer: <input type="text" id="reviewer" placeholder="your name" style="width:140px;padding:4px 6px"></span>
</header>
<div id="pictures" style="display:none">
 <div class="picbar">
  <span class="hint">Click to throw a picture out. Shift-click to select a run of them. Hover for the magnifier.</span>
  <label style="font-weight:400;font-size:13px">Size
   <select id="picsize" onchange="setPicSize(this.value)">
    <option value="150">Small</option><option value="200" selected>Medium</option>
    <option value="280">Large</option><option value="400">Huge</option></select></label>
  <span style="margin-left:auto"><b id="markcount">0</b> marked</span>
  <button onclick="clearMarks()">Clear</button>
  <button class="warn" onclick="deleteMarked()">Delete marked</button>
 </div>
 <div id="picgrid"></div>
</div>
<div id="questions" style="display:none">
 <div class="picbar">
  <span class="hint">Questions MarieClaire could not answer. Tick the ones worth researching.</span>
  <span style="margin-left:auto"><b id="qpicked">0</b> selected</span>
  <button onclick="refreshQuestions()">Refresh</button>
  <button onclick="queueQuestions()">Queue for research</button>
 </div>
 <div id="qbody"></div>
</div>
<div id="guides" style="display:none">
 <div class="picbar">
  <span class="hint">Click a guide to throw it out. Shift-click for a run.</span>
  <label style="font-weight:400;font-size:13px">Show
   <select id="gfilter" onchange="renderGuides()">
    <option value="all">Everything</option>
    <option value="one">Only one step</option>
    <option value="noimg">No pictures</option>
    <option value="img">Has pictures</option>
   </select></label>
  <span style="margin-left:auto"><b id="gmarkcount">0</b> marked</span>
  <button onclick="clearGuideMarks()">Clear</button>
  <button class="warn" onclick="deleteMarkedGuides()">Delete marked</button>
 </div>
 <div id="glist" class="glist"></div>
</div>
<div id="images"></div><div id="procedures" style="display:none"></div><div id="docs" style="display:none"></div>
<script>
let S;
const esc = s => (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function load(){ S = await (await fetch('/api/state')).json(); S.docs = await (await fetch('/api/docs')).json();
  refreshQuestions();
  const rv=document.getElementById('reviewer'); try{ rv.value = localStorage.getItem('kb-reviewer')||''; }catch(e){}
  let sz='200'; try{ sz = localStorage.getItem('kb-picsize')||'200'; }catch(e){}
  document.getElementById('picsize').value = sz; setPicSize(sz);
  rv.addEventListener('change', ()=>{ try{ localStorage.setItem('kb-reviewer', rv.value); }catch(e){} }); render(); }
function show(t){ document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  for (const k of ['pictures','questions','guides','images','procedures','docs']) document.getElementById(k).style.display = t===k?'':'none'; }

// --- Pictures tab: one card per picture FILE (several guides can share one file) ---
const MARKED = new Set();

function uniquePictures(){
  const byFile = new Map();
  Object.values(S.images).forEach(im => {
    if (!byFile.has(im.path)) byFile.set(im.path, []);
    byFile.get(im.path).push(im);
  });
  return [...byFile.entries()].sort((a,b) =>
    a[1][0].document.localeCompare(b[1][0].document) || a[1][0].page - b[1][0].page);
}

let PIC_ORDER = [];      // paths in display order, for shift-click ranges
let LAST_CLICKED = null;

function renderPictures(){
  const host = document.getElementById('picgrid');
  host.innerHTML = '';
  PIC_ORDER = [];
  const groups = new Map();
  uniquePictures().forEach(([path, entries]) => {
    if (!groups.has(entries[0].document)) groups.set(entries[0].document, []);
    groups.get(entries[0].document).push([path, entries]);
  });

  groups.forEach((pics, doc) => {
    const head = document.createElement('div');
    head.className = 'docgroup';
    head.innerHTML = `<b>${esc(doc)}</b> <span style="color:#888">${pics.length}</span>`;
    const all = document.createElement('button');
    all.className = 'mini';
    all.textContent = 'mark all';
    all.onclick = () => markGroup(pics.map(p => p[0]));
    head.appendChild(all);
    host.appendChild(head);

    const grid = document.createElement('div');
    grid.className = 'picgrid';
    pics.forEach(([path, entries]) => {
      PIC_ORDER.push(path);
      const first = entries[0];
      const d = document.createElement('div');
      d.className = 'pic' + (MARKED.has(path) ? ' marked' : '');
      d.dataset.path = path;
      d.onclick = e => clickPicture(path, e.shiftKey);
      d.innerHTML = `<a class="zoom" href="/images/${path}" target="_blank" onclick="event.stopPropagation()">&#128269;</a>
        <img loading="lazy" src="/images/${path}">
        <div class="cap">${esc(first.alt_text)}</div>
        <div class="where">page ${first.page}${entries.length>1?`<span class="dup">${entries.length} places</span>`:''}</div>`;
      grid.appendChild(d);
    });
    host.appendChild(grid);
  });
  updateMarkCount();
}

function paint(){
  document.querySelectorAll('.pic').forEach(el => el.classList.toggle('marked', MARKED.has(el.dataset.path)));
  updateMarkCount();
}

function clickPicture(path, shift){
  if (shift && LAST_CLICKED){
    const a = PIC_ORDER.indexOf(LAST_CLICKED), b = PIC_ORDER.indexOf(path);
    if (a > -1 && b > -1) PIC_ORDER.slice(Math.min(a,b), Math.max(a,b)+1).forEach(p => MARKED.add(p));
  } else {
    MARKED.has(path) ? MARKED.delete(path) : MARKED.add(path);
  }
  LAST_CLICKED = path;
  paint();
}

function markGroup(paths){
  const allMarked = paths.every(p => MARKED.has(p));
  paths.forEach(p => allMarked ? MARKED.delete(p) : MARKED.add(p));
  paint();
}

function clearMarks(){ MARKED.clear(); LAST_CLICKED = null; paint(); }

function setPicSize(px){
  document.documentElement.style.setProperty('--tw', px + 'px');
  document.documentElement.style.setProperty('--th', Math.round(px * 0.65) + 'px');
  try { localStorage.setItem('kb-picsize', px); } catch(e){}
}

function updateMarkCount(){ document.getElementById('markcount').textContent = MARKED.size; }

// --- Guides tab: the auto-written step-by-step guides ---
const GMARKED = new Set();
let GUIDE_ORDER = [], GLAST = null;

function guideFacts(g){
  const steps = g.steps || [];
  const pics = steps.reduce((n,s) => n + ((s.image_ids||[]).length), 0);
  return {steps: steps.length, pics};
}

function renderGuides(){
  const host = document.getElementById('glist');
  const mode = document.getElementById('gfilter').value;
  host.innerHTML = '';
  GUIDE_ORDER = [];
  Object.values(S.procedures)
    .sort((a,b) => a.document.localeCompare(b.document) || (a.page||0)-(b.page||0))
    .forEach(g => {
      const f = guideFacts(g);
      if (mode === 'one'   && f.steps > 1) return;
      if (mode === 'noimg' && f.pics > 0)  return;
      if (mode === 'img'   && f.pics === 0) return;
      GUIDE_ORDER.push(g.procedure_id);

      const tags = [];
      if (f.steps <= 1) tags.push('<span class="t one">one step</span>');
      if (f.steps > 3)  tags.push(`<span class="t many">${f.steps} steps</span>`);
      tags.push(f.pics ? `<span class="t img">${f.pics} picture${f.pics>1?'s':''}</span>`
                       : '<span class="t noimg">no pictures</span>');

      const thumbs = (g.steps||[]).flatMap(s => s.image_ids||[])
        .filter(i => S.images[i])
        .map(i => `<img loading="lazy" src="/images/${S.images[i].path}">`).join('');

      const steps = (g.steps||[]).map(s =>
        `<li>${esc(s.text)}${(s.image_ids||[]).length ? '<span class="has">has picture</span>' : ''}</li>`).join('');

      const d = document.createElement('div');
      d.className = 'g' + (GMARKED.has(g.procedure_id) ? ' marked' : '');
      d.dataset.gid = g.procedure_id;
      d.onclick = e => clickGuide(g.procedure_id, e.shiftKey);
      d.innerHTML = `<div><div class="gtitle">${esc(g.title || g.procedure_id)}</div>
        <div class="gsrc">${esc(g.document)} &middot; page ${g.page||'?'}</div>
        <ol class="gsteps">${steps}</ol></div>
        <div class="gtags">${tags.join('')}<div class="gthumbs">${thumbs}</div></div>`;
      host.appendChild(d);
    });
  updateGuideCount();
}

function paintGuides(){
  document.querySelectorAll('.g').forEach(el => el.classList.toggle('marked', GMARKED.has(el.dataset.gid)));
  updateGuideCount();
}

function clickGuide(id, shift){
  if (shift && GLAST){
    const a = GUIDE_ORDER.indexOf(GLAST), b = GUIDE_ORDER.indexOf(id);
    if (a > -1 && b > -1) GUIDE_ORDER.slice(Math.min(a,b), Math.max(a,b)+1).forEach(x => GMARKED.add(x));
  } else {
    GMARKED.has(id) ? GMARKED.delete(id) : GMARKED.add(id);
  }
  GLAST = id;
  paintGuides();
}

function clearGuideMarks(){ GMARKED.clear(); GLAST = null; paintGuides(); }
function updateGuideCount(){ document.getElementById('gmarkcount').textContent = GMARKED.size; }

// --- Questions tab: unanswered questions from the central list ---
let QUESTIONS = [], QPICKED = new Set();

async function refreshQuestions(){
  const r = await fetch('/api/questions');
  const data = await r.json();
  QUESTIONS = data.gaps || [];
  QPICKED.clear();
  renderQuestions(data.problem);
}

function renderQuestions(problem){
  const host = document.getElementById('qbody');
  host.innerHTML = '';
  document.getElementById('qsc').textContent = QUESTIONS.length ? `(${QUESTIONS.length})` : '';
  if (problem){
    host.innerHTML = `<div class="warnbox">${esc(problem)}</div>`;
    return;
  }
  if (!QUESTIONS.length){
    host.innerHTML = '<div class="warnbox">Nothing waiting. Every question so far was answered from the library.</div>';
    return;
  }
  QUESTIONS.forEach(g => {
    const d = document.createElement('div');
    d.className = 'q' + (QPICKED.has(g.id) ? ' picked' : '');
    d.innerHTML = `<input type="checkbox" ${QPICKED.has(g.id)?'checked':''} onchange="pickQuestion(${g.id}, this.checked)">
      <div><div class="qtext">${esc(g.question)}</div>
      <div class="qmeta">${esc(g.language||'')} &middot; first asked ${esc((g.first_asked_at||'').slice(0,10))}</div></div>
      <div style="display:flex;gap:8px;align-items:center">
        <span class="qtimes ${g.times_asked>=3?'hot':''}">asked ${g.times_asked}&times;</span>
        <button class="warn" onclick="discardQuestion(${g.id})">Discard</button>
      </div>`;
    host.appendChild(d);
  });
  updateQPicked();
}

function pickQuestion(id, on){
  on ? QPICKED.add(id) : QPICKED.delete(id);
  renderQuestions(null);
}

function updateQPicked(){ document.getElementById('qpicked').textContent = QPICKED.size; }

async function discardQuestion(id){
  if (!confirm('Discard this question? It leaves the waiting list.')) return;
  const r = await fetch(`/api/questions/${id}/discard`, {method:'POST'});
  if (!r.ok) return alert('Could not discard: ' + await r.text());
  await refreshQuestions();
}

async function queueQuestions(){
  if (!QPICKED.size) return alert('Tick the questions you want researched first.');
  const chosen = QUESTIONS.filter(g => QPICKED.has(g.id));
  const r = await fetch('/api/questions/select', {method:'POST', headers:{'Content-Type':'application/json'},
                                                  body: JSON.stringify({gaps: chosen})});
  if (!r.ok) return alert('Could not queue: ' + await r.text());
  const out = await r.json();
  document.getElementById('qbody').insertAdjacentHTML('afterbegin',
    `<div class="qcmd">${out.queued} question(s) queued. Run this in your terminal:

${esc(out.command)}</div>`);
}


async function deleteMarkedGuides(){
  if (!GMARKED.size) return alert('No guides marked yet.');
  if (!confirm(`Delete ${GMARKED.size} guide(s)?

They are saved in kb_review/discarded_guides.json and the bot stops using them.`)) return;
  const r = await fetch('/api/procedures/discard', {method:'POST', headers:{'Content-Type':'application/json'},
                                                    body: JSON.stringify({procedure_ids: [...GMARKED]})});
  if (!r.ok) return alert('Could not delete: ' + await r.text());
  const out = await r.json();
  GMARKED.clear();
  await load();
  show('guides');
  alert(`Deleted ${out.removed}. ${out.left} guides left.

Now run:  python ingest_kb.py build`);
}


async function deleteMarked(){
  if (!MARKED.size) return alert('No pictures marked yet. Click the ones you want to throw out.');
  const ids = Object.values(S.images).filter(im => MARKED.has(im.path)).map(im => im.image_id);
  if (!confirm(`Delete ${MARKED.size} picture(s)?\n\nThey move to kb_review/discarded/ and the bot stops using them.`)) return;
  const r = await fetch('/api/images/discard', {method:'POST', headers:{'Content-Type':'application/json'},
                                                body: JSON.stringify({image_ids: ids})});
  if (!r.ok) return alert('Could not delete: ' + await r.text());
  const out = await r.json();
  MARKED.clear();
  await load();
  show('pictures');
  alert(`Deleted ${out.files} picture(s). Now run:  python ingest_kb.py build`);
}
function counts(){ const im=Object.values(S.images), pr=Object.values(S.procedures), dc=Object.values(S.docs||{});
  document.getElementById('ic').textContent = `(${im.filter(i=>i.approved).length}/${im.length})`;
  document.getElementById('pc').textContent = `(${pr.filter(p=>p.approved).length}/${pr.length})`;
  document.getElementById('dc').textContent = `(${dc.filter(d=>d.review.decision!=='pending').length}/${dc.length} decided)`;
  document.getElementById('qc').textContent = `(${uniquePictures().length})`;
  document.getElementById('gc').textContent = `(${pr.length})`; }
function render(){ counts(); renderPictures(); renderGuides(); renderImages(); renderProcs(); renderDocs(); }

function renderImages(){
  const el = document.getElementById('images'); el.innerHTML='';
  Object.values(S.images).sort((a,b)=>a.document.localeCompare(b.document)||a.page-b.page||a.image_id.localeCompare(b.image_id)).forEach(im=>{
    const d = document.createElement('div'); d.className='card'+(im.approved?' ok':'')+(im.keep?'':' off'); d.id='img-'+im.image_id;
    d.innerHTML = `<img src="/images/${im.path}">
      <div><div class="src"><b>${im.image_id}</b> — ${esc(im.document)} p${im.page}</div>
      <div class="ctx"><b>Before:</b> ${esc(im.context_before)}</div>
      <h4>Semantic description (search/reasoning)</h4><textarea id="sd-${im.image_id}">${esc(im.semantic_description)}</textarea>
      <h4>Alt text (shown to users)</h4><input type="text" id="alt-${im.image_id}" value="${esc(im.alt_text)}">
      <div class="ctx"><b>After:</b> ${esc(im.context_after)}</div>
      <div class="flags">
        <label><input type="checkbox" id="keep-${im.image_id}" ${im.keep?'checked':''}> Keep</label>
        <label><input type="checkbox" id="safe-${im.image_id}" ${im.safe_to_show?'checked':''}> Safe to show</label>
        <label><input type="checkbox" id="ok-${im.image_id}" ${im.approved?'checked':''}> Approved</label>
        <button onclick="saveImage('${im.image_id}')">Save</button><span class="saved" id="sv-${im.image_id}"></span>
      </div></div>`;
    el.appendChild(d);
  });
}
async function saveImage(id){
  const body = { semantic_description: v('sd-'+id), alt_text: v('alt-'+id), keep: c('keep-'+id), safe_to_show: c('safe-'+id), approved: c('ok-'+id) };
  S.images[id] = await (await fetch('/api/images/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  counts(); const d=document.getElementById('img-'+id); d.classList.toggle('ok',S.images[id].approved); d.classList.toggle('off',!S.images[id].keep);
  flash('sv-'+id);
}

function renderProcs(){
  const el = document.getElementById('procedures'); el.innerHTML='';
  Object.values(S.procedures).forEach(p=>{
    const d = document.createElement('div'); d.className='proc'+(p.approved?' ok':''); d.id='proc-'+p.procedure_id;
    const steps = p.steps.map((s,i)=>`<div class="step"><div class="n">${i+1}</div>
      <textarea id="st-${p.procedure_id}-${i}">${esc(s.text)}</textarea>
      <div><input type="text" id="si-${p.procedure_id}-${i}" value="${esc(s.image_ids.join(', '))}" placeholder="image ids, comma-separated">
      <div class="thumbs">${s.image_ids.map(id=>S.images[id]?`<img title="${id}" src="/images/${S.images[id].path}">`:`<span style="color:#b91c1c">${id}?</span>`).join('')}</div></div></div>`).join('');
    d.innerHTML = `<div class="src"><b>${p.procedure_id}</b> — ${esc(p.document)} p${p.page}</div>
      <h4>Title</h4><input type="text" id="pt-${p.procedure_id}" value="${esc(p.title)}">
      <h4>Steps (text · screenshot ids shown after that step)</h4>${steps}
      <div class="flags"><label><input type="checkbox" id="pok-${p.procedure_id}" ${p.approved?'checked':''}> Approved</label>
      <button onclick="addStep('${p.procedure_id}')">Add step</button>
      <button onclick="saveProc('${p.procedure_id}')">Save</button>
      <button class="warn" onclick="delProc('${p.procedure_id}')">Delete procedure</button><span class="saved" id="psv-${p.procedure_id}"></span></div>`;
    el.appendChild(d);
  });
}
async function saveProc(id){
  const p = S.procedures[id];
  const steps = p.steps.map((s,i)=>({ text: v(`st-${id}-${i}`), image_ids: v(`si-${id}-${i}`).split(',').map(x=>x.trim()).filter(Boolean) })).filter(s=>s.text);
  const body = { title: v('pt-'+id), steps, approved: c('pok-'+id) };
  S.procedures[id] = await (await fetch('/api/procedures/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  renderProcs(); counts(); show('procedures'); flash('psv-'+id);
}
async function delProc(id){ await fetch('/api/procedures/'+id,{method:'DELETE'}); delete S.procedures[id]; renderProcs(); counts(); }
function addStep(id){ saveProcLocal(id); S.procedures[id].steps.push({text:'', image_ids:[]}); renderProcs(); show('procedures'); }
function saveProcLocal(id){ const p=S.procedures[id]; p.title=v('pt-'+id);
  p.steps = p.steps.map((s,i)=>({ text: v(`st-${id}-${i}`), image_ids: v(`si-${id}-${i}`).split(',').map(x=>x.trim()).filter(Boolean) })); }
async function newProc(){
  const id = prompt('Procedure id (snake_case, e.g. dashboard_create_task):'); if(!id) return;
  const title = prompt('Title (e.g. Create a task on the Dashboard):') || id;
  const doc = prompt('Source document filename (as in the image cards):') || '';
  const r = await fetch('/api/procedures',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({procedure_id:id,title,document:doc})});
  if(!r.ok){ alert('Could not create (id missing or already exists)'); return; }
  S.procedures[id] = await r.json(); renderProcs(); counts(); show('procedures');
  document.getElementById('proc-'+id)?.scrollIntoView();
}
function renderDocs(){
  const el = document.getElementById('docs'); el.innerHTML='';
  const docs = Object.values(S.docs||{});
  if(!docs.length){ el.innerHTML = '<div class="doc">No researched drafts yet. Run: <code>python codebase_agent/research.py --gaps ...</code></div>'; return; }
  docs.forEach(d=>{
    const id = d.gap_id, rv = d.review||{decision:'pending'};
    const card = document.createElement('div'); card.className = 'doc '+(rv.decision||'pending'); card.id='doc-'+id;
    const sources = (d.sources||[]).map(s=>`${s.name}@${(s.commit_sha||'').slice(0,8)}${s.sha_verified_by_mcl_team?'':' (unverified)'}`).join(', ');
    const files = (d.evidence_files||[]).map(f=>`<li><code>${esc(f.repo)}/${esc(f.path)}</code> ${esc(f.lines||'')} — ${esc(f.why||'')}</li>`).join('') || '<li><i>none cited</i></li>';
    const unc = (d.uncertainties||[]).map(u=>`<li>${esc(u)}</li>`).join('') || '<li><i>none</i></li>';
    card.innerHTML = `
      <div class="src"><b>${esc(id)}</b> · <span class="badge ${esc(d.status)}">${esc(d.status)}</span><span class="badge">${esc(d.platform||'unknown')}</span> sources: ${esc(sources)}
        ${rv.reviewed_by?` · reviewed by ${esc(rv.reviewed_by)} ${esc((rv.reviewed_at||'').slice(0,10))}`:''}</div>
      <div class="q">Q: ${esc(d.question)}</div>
      <div class="cols">
        <div><h4>Draft (what users will read) — editable</h4><textarea class="draft" id="dd-${id}">${esc(d.draft_md||'')}</textarea></div>
        <div><h4>Evidence (reviewers only — never enters the KB)</h4><div class="ev">
          <b>Title:</b> ${esc(d.title||'')}<br><b>Summary:</b> ${esc(d.summary_for_reviewer||'')}
          ${d.abstain_reason?`<br><b>Abstain reason:</b> ${esc(d.abstain_reason)}`:''}
          <br><b>Uncertainties:</b><ul>${unc}</ul><b>Code read:</b><ul>${files}</ul>
          ${d.has_report?`<button onclick="showReport('${id}')">Full research report</button><pre id="rep-${id}" style="display:none;white-space:pre-wrap;font-size:12px"></pre>`:''}
        </div></div>
      </div>
      <div class="flags">
        Decision: <select id="dec-${id}">${['pending','approved','rejected','needs_product_owner'].map(o=>`<option ${o===rv.decision?'selected':''}>${o}</option>`).join('')}</select>
        <input type="text" id="dn-${id}" placeholder="review notes (why / what to confirm)" value="${esc(rv.notes||'')}" style="width:46%;margin-left:8px">
        <button onclick="saveDoc('${id}')">Save</button><span class="saved" id="dsv-${id}"></span>
      </div>`;
    el.appendChild(card);
  });
}
async function saveDoc(id){
  const body = { draft_md: document.getElementById('dd-'+id).value, decision: v('dec-'+id), notes: v('dn-'+id), reviewed_by: v('reviewer') };
  const r = await fetch('/api/docs/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){ alert('Save failed: '+(await r.text())); return; }
  S.docs[id] = await r.json(); counts();
  const card=document.getElementById('doc-'+id); card.className='doc '+S.docs[id].review.decision; flash('dsv-'+id);
}
async function showReport(id){ const pre=document.getElementById('rep-'+id);
  if(pre.style.display==='none'){ pre.textContent = (await (await fetch(`/api/docs/${id}/report`)).json()).report; pre.style.display=''; } else pre.style.display='none'; }
const v = id => document.getElementById(id).value.trim(), c = id => document.getElementById(id).checked;
function flash(id){ const e=document.getElementById(id); if(e){ e.textContent='saved'; setTimeout(()=>e.textContent='',1200);} }
load();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)

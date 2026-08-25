"""Local, offline review tool for KB screenshots and procedures.

    cd assistant_api && python review_server.py      # then open http://127.0.0.1:8010

Edits save straight into kb_review/images.json and kb_review/procedures.json (no
download-and-replace). Binds to localhost only. After reviewing: python ingest_kb.py build
"""
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

SELF_DIR = Path(__file__).resolve().parent
REVIEW_DIR = SELF_DIR / "kb_review"
IMAGES_FILE = REVIEW_DIR / "images.json"
PROCEDURES_FILE = REVIEW_DIR / "procedures.json"

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
</style></head><body>
<header><b>KB review</b>
 <span class="tab on" data-t="images" onclick="show('images')">Images <span id="ic"></span></span>
 <span class="tab" data-t="procedures" onclick="show('procedures')">Procedures <span id="pc"></span></span>
 <button onclick="newProc()">New procedure</button>
</header>
<div id="images"></div><div id="procedures" style="display:none"></div>
<script>
let S;
const esc = s => (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function load(){ S = await (await fetch('/api/state')).json(); render(); }
function show(t){ document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  document.getElementById('images').style.display = t==='images'?'':'none';
  document.getElementById('procedures').style.display = t==='procedures'?'':'none'; }
function counts(){ const im=Object.values(S.images), pr=Object.values(S.procedures);
  document.getElementById('ic').textContent = `(${im.filter(i=>i.approved).length}/${im.length})`;
  document.getElementById('pc').textContent = `(${pr.filter(p=>p.approved).length}/${pr.length})`; }
function render(){ counts(); renderImages(); renderProcs(); }

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
const v = id => document.getElementById(id).value.trim(), c = id => document.getElementById(id).checked;
function flash(id){ const e=document.getElementById(id); if(e){ e.textContent='saved'; setTimeout(()=>e.textContent='',1200);} }
load();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)

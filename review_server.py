"""Local, offline review tool for MarieClaire's knowledge.

    cd assistant_api && python review_server.py      # then open http://127.0.0.1:8010

Two tabs:
  Questions — the questions MarieClaire could not answer (from the central list). Hand them to
              Claude (say "answer the pending questions" in a Claude Code chat, or /answer-gaps).
  Drafts    — the pages Claude wrote for those questions, waiting for your approval. Web pages
              carry screenshots; code answers (behaviour/rules) are the same, without images.
              Approve publishes into the knowledge base the agent uses; reject sends the
              question back to be tried again.

Reads the central list through the backend's protected endpoint, so no database credential
lives here. Binds to localhost only.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

load_dotenv(Path(__file__).resolve().parent / ".env")

SELF_DIR = Path(__file__).resolve().parent
PENDING = SELF_DIR / "kb_review" / "pending_web"     # draft pages awaiting approval: <gap-id>.json
MAP_DIR = SELF_DIR / "app" / "documents" / "new_app_map"

GAP_API_URL = os.getenv("GAP_API_URL", "http://127.0.0.1:8001").rstrip("/")
GAP_INGEST_TOKEN = os.getenv("GAP_INGEST_TOKEN", "").strip()
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", "http://127.0.0.1:8001").rstrip("/")

app = FastAPI(title="MarieClaire review")


def _gap_headers() -> dict:
    return {"Authorization": f"Bearer {GAP_INGEST_TOKEN}"} if GAP_INGEST_TOKEN else {}


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _mark_gap(gap_id: str, status: str) -> None:
    digits = "".join(ch for ch in str(gap_id) if ch.isdigit())
    if not digits:
        return
    try:
        with httpx.Client(timeout=10) as http:
            http.patch(f"{GAP_API_URL}/api/gaps/{digits}", json={"status": status}, headers=_gap_headers())
    except Exception:
        pass   # closing the gap is best-effort; the page still publishes


# --- Questions: the unanswered questions from the central list ---

@app.get("/api/questions")
def list_questions():
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


@app.post("/api/questions/{gap_id}/status")
def set_question_status(gap_id: int, body: dict):
    with httpx.Client(timeout=10) as http:
        response = http.patch(f"{GAP_API_URL}/api/gaps/{gap_id}",
                              json={"status": body.get("status")}, headers=_gap_headers())
    if response.status_code >= 400:
        raise HTTPException(response.status_code, response.text[:200])
    return {"ok": True}


# --- Drafts: pages Claude wrote (web + code), awaiting approval ---

@app.get("/api/drafts")
def list_drafts():
    if not PENDING.exists():
        return {"pages": []}
    pages = []
    for path in sorted(PENDING.glob("*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        for shot in page.get("screenshots", []):
            shot["url"] = f"{API_PUBLIC_URL}/images/kb/webmap/{shot.get('file', '')}"
        page["_id"] = path.stem
        pages.append(page)
    return {"pages": pages}


@app.post("/api/drafts/{draft_id}/approve")
def approve_draft(draft_id: str):
    src = PENDING / f"{draft_id}.json"
    if not src.exists():
        raise HTTPException(404, "no such draft")
    page = json.loads(src.read_text(encoding="utf-8"))
    page.pop("_id", None)
    page.pop("gap_id", None)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    (MAP_DIR / f"{_slug(page['screen'])}.json").write_text(json.dumps(page, indent=2, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run([sys.executable, "ingest_kb.py", "add-web-map"],
                            cwd=SELF_DIR, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise HTTPException(500, f"publish failed: {result.stderr[-300:]}")
    src.unlink()
    _mark_gap(draft_id, "researched")
    return {"ok": True, "published": page["screen"]}


@app.post("/api/drafts/{draft_id}/reject")
def reject_draft(draft_id: str):
    src = PENDING / f"{draft_id}.json"
    if src.exists():
        src.unlink()
    _mark_gap(draft_id, "pending")   # back onto the open list to be walked again
    return {"ok": True}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MarieClaire review</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<style>
 :root{--bg:#f6f7f9;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#0f766e;
       --web:#0369a1;--web-bg:#e0f2fe;--code:#7c3aed;--code-bg:#f3e8ff;--warn:#b91c1c;--hot:#c2410c}
 *{box-sizing:border-box}
 body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
 header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);
        padding:14px 22px;display:flex;gap:18px;align-items:center;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 .brand{font-weight:700;font-size:17px;letter-spacing:-.01em}
 .brand small{display:block;font-weight:400;font-size:12px;color:var(--muted)}
 nav{display:flex;gap:6px;margin-left:8px}
 .tab{cursor:pointer;padding:8px 14px;border-radius:999px;font-weight:600;font-size:14px;color:var(--muted);border:1px solid transparent}
 .tab:hover{background:#f1f5f9}.tab.on{background:var(--accent);color:#fff}
 .tab .count{opacity:.75;font-weight:500;margin-left:4px}
 main{max-width:960px;margin:22px auto;padding:0 22px}
 .hint{color:var(--muted);font-size:13.5px;margin:0 0 16px}
 .empty,.warnbox{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px;color:var(--muted);text-align:center}
 .warnbox{border-color:#fecaca;background:#fef2f2;color:var(--warn);text-align:left}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:12px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
 .qtop{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
 .badge{font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;padding:3px 9px;border-radius:999px;background:#f1f5f9;color:var(--muted)}
 .badge.web{background:var(--web-bg);color:var(--web)} .badge.code{background:var(--code-bg);color:var(--code)} .badge.app{background:#f1f5f9;color:var(--muted)}
 .badge.fix{background:#fef3c7;color:#92400e}
 .note{font-size:13px;color:#92400e;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:8px 10px;margin:0 0 10px}
 .flagged{font-size:12px;color:var(--muted);margin:6px 0 4px}
 .bubble{background:#f1f5f9;border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:0 0 12px;font-size:14px;line-height:1.55}
 .bubble p{margin:0 0 .5rem}.bubble p:last-child{margin-bottom:0}
 .bubble ol,.bubble ul{margin:.25rem 0 .5rem;padding-left:1.25rem}.bubble li{margin:.15rem 0}
 .bubble img{max-width:280px;border:1px solid var(--line);border-radius:8px;margin:.4rem 0;display:block}
 .bubble code{font-size:.85em;background:#e2e8f0;padding:.05rem .3rem;border-radius:.25rem}
 .role{font-size:12.5px;color:var(--muted)} .role b{color:var(--ink);font-weight:600}
 .times{margin-left:auto;font-size:12.5px;color:var(--muted)} .times.hot{color:var(--hot);font-weight:700}
 .qtext{font-size:15.5px;line-height:1.5;margin:2px 0 12px}
 .qmeta{font-size:12px;color:var(--muted);margin-bottom:12px}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 button{font:inherit;font-size:13.5px;font-weight:600;cursor:pointer;border:1px solid var(--line);background:#fff;color:var(--ink);padding:7px 13px;border-radius:8px}
 button:hover{background:#f8fafc}
 button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.primary:hover{filter:brightness(1.05)}
 button.ghost{color:var(--warn);border-color:#fecaca}button.ghost:hover{background:#fef2f2}
 .ok{color:var(--accent);font-size:12.5px;font-weight:600;margin-left:6px}
 .procs{margin:6px 0 10px}.proc{margin:6px 0}.proc ol{margin:4px 0 0;padding-left:20px}.proc li{margin:2px 0;font-size:14px}
 .shots{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 12px}
 .shots figure{margin:0;max-width:320px}.shots img{width:100%;border:1px solid var(--line);border-radius:8px}
 .shots figcaption{font-size:12px;color:var(--muted);margin-top:4px}
</style></head><body>
<header>
  <div class="brand">MarieClaire <small>knowledge review</small></div>
  <nav>
    <span class="tab on" data-t="questions" onclick="show('questions')">Questions<span class="count" id="qc"></span></span>
    <span class="tab" data-t="drafts" onclick="show('drafts')">Drafts<span class="count" id="wc"></span></span>
  </nav>
</header>
<main>
  <section id="questions">
    <p class="hint">Questions MarieClaire could not answer. In a Claude Code chat say <b>“answer the pending questions”</b> (or run <code>/answer-gaps</code>); Claude walks the screens, or reads the code when a screen can't answer, and the drafts appear under Drafts.</p>
    <div class="row" style="margin-bottom:14px"><button onclick="refreshQuestions()">Refresh</button></div>
    <div id="qbody"></div>
  </section>
  <section id="drafts" style="display:none">
    <p class="hint">Pages Claude wrote for the pending questions. Web pages carry screenshots; code answers don't. Approve to publish into the knowledge base; reject to send the question back to be tried again.</p>
    <div id="wbody"></div>
  </section>
</main>
<script>
const esc = s => (s??'').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function show(t){ document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  ['questions','drafts'].forEach(s=>document.getElementById(s).style.display = s===t?'':'none'); }

// --- Questions ---
let QUESTIONS = [];
async function refreshQuestions(){
  const data = await (await fetch('/api/questions')).json();
  QUESTIONS = data.gaps || []; renderQuestions(data.problem);
}
function renderQuestions(problem){
  const host = document.getElementById('qbody'); host.innerHTML='';
  document.getElementById('qc').textContent = QUESTIONS.length ? ` ${QUESTIONS.length}` : '';
  if(problem){ host.innerHTML = `<div class="warnbox">${esc(problem)}</div>`; return; }
  if(!QUESTIONS.length){ host.innerHTML = '<div class="empty">Nothing waiting. Every question so far was answered from the library.</div>'; return; }
  QUESTIONS.forEach(g=>{
    const surf=(g.surface||'other').toLowerCase(), badge=surf==='web'?'web':(surf==='app'?'app':'');
    const isFix = g.kind==='correction';
    const card=document.createElement('div'); card.className='card';
    card.innerHTML = `
      <div class="qtop">
        ${isFix?'<span class="badge fix">correction</span>':''}
        <span class="badge ${badge}">${esc(surf)}</span>
        <span class="role">role <b>${esc(g.role||'—')}</b></span>
        <span class="times ${g.times_asked>=3?'hot':''}">${isFix?'reported':'asked'} ${g.times_asked}×</span></div>
      <div class="qtext">${esc(g.question)}</div>
      ${isFix ? renderFlagged(g.note) : ''}
      <div class="qmeta">#${g.id} · ${esc(g.language||'')} · first ${esc((g.first_asked_at||'').slice(0,10))}</div>
      <div class="row">
        <button class="primary" onclick="copyForClaude(${g.id})">Copy for Claude</button>
        <button class="ghost" onclick="setStatus(${g.id},'discarded')">Discard</button>
        <span class="ok" id="qok-${g.id}"></span></div>`;
    host.appendChild(card);
  });
}
function renderFlagged(note){
  if(!note) return '';
  const i = note.indexOf('answer given:');
  const complaint = (i < 0 ? '' : note.slice(0, i)).replace(/\\|\\|\\s*$/, '').trim();
  const answer = i < 0 ? note : note.slice(i + 'answer given:'.length).trim();
  const bubble = answer ? `<div class="flagged">The answer the user reported as wrong:</div><div class="bubble">${marked.parse(answer)}</div>` : '';
  const said = complaint ? `<div class="note">User said: ${esc(complaint)}</div>` : '';
  return said + bubble;
}
async function copyForClaude(id){
  const g = QUESTIONS.find(x=>x.id===id); if(!g) return;
  await navigator.clipboard.writeText(`Please answer this pending MarieClaire question (walk the web screen, or read the code if no screen answers it).\n#${g.id} [${g.surface||'?'}, role ${g.role||'—'}]: ${g.question}`);
  const e=document.getElementById('qok-'+id); e.textContent='copied — paste it to Claude'; setTimeout(()=>e.textContent='',2500);
}
async function setStatus(id,status){
  if(status==='discarded' && !confirm('Discard this question? It leaves the waiting list.')) return;
  const r = await fetch(`/api/questions/${id}/status`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status})});
  if(!r.ok) return alert('Failed: '+await r.text());
  refreshQuestions();
}

// --- Drafts ---
async function refreshDrafts(){
  const pages = (await (await fetch('/api/drafts')).json()).pages || [];
  const el = document.getElementById('wbody'); el.innerHTML='';
  document.getElementById('wc').textContent = pages.length ? ` ${pages.length}` : '';
  if(!pages.length){ el.innerHTML = '<div class="empty">No drafts waiting. In a Claude Code chat, say “answer the pending questions”.</div>'; return; }
  pages.forEach(p=>{
    const shots=(p.screenshots||[]).map(s=>`<figure><img src="${esc(s.url)}" alt="${esc(s.alt)}" loading="lazy"><figcaption>${esc(s.alt)}</figcaption></figure>`).join('');
    const procs=(p.procedures||[]).map(pr=>`<div class="proc"><b>${esc(pr.title)}</b><ol>${(pr.steps||[]).map(s=>`<li>${esc(s.text)}</li>`).join('')}</ol></div>`).join('');
    const kind = shots ? 'web' : 'code';
    const card=document.createElement('div'); card.className='card';
    card.innerHTML = `
      <div class="qtop"><span class="badge ${kind}">${kind}</span><span class="role"><b>${esc(p.screen)}</b> · ${esc(p.route||'')}</span></div>
      <div class="qtext">${esc(p.overview||'')}</div>
      ${procs?`<div class="procs">${procs}</div>`:''}
      ${shots?`<div class="shots">${shots}</div>`:''}
      <div class="row">
        <button class="primary" onclick="approve('${p._id}')">Approve &amp; publish</button>
        <button class="ghost" onclick="reject('${p._id}')">Reject</button>
        <span class="ok" id="wok-${p._id}"></span></div>`;
    el.appendChild(card);
  });
}
async function approve(id){
  const e=document.getElementById('wok-'+id); e.textContent='publishing…';
  const r = await fetch(`/api/drafts/${id}/approve`,{method:'POST'});
  if(!r.ok){ e.textContent=''; return alert('Publish failed: '+await r.text()); }
  refreshDrafts(); refreshQuestions();
}
async function reject(id){
  if(!confirm('Reject this draft? The question goes back to be tried again.')) return;
  await fetch(`/api/drafts/${id}/reject`,{method:'POST'});
  refreshDrafts(); refreshQuestions();
}

refreshQuestions(); refreshDrafts();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)

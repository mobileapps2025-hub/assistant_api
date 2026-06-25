"""Layer 5 — the retrieval pipeline: contextualize → retrieve → answer."""
from typing import Any, Dict, List, Optional

from app.core.flow import flow
from app.retrieval.answerer import answer
from app.retrieval.contextualizer import contextualize
from app.retrieval.retriever import retrieve

HISTORY_TURNS = 6


def _text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(item.get("text", "") for item in content if item.get("type") == "text").strip()
    return str(content or "").strip()


def _history_text(messages: List[Dict[str, Any]]) -> str:
    prior = messages[:-1] if messages else []
    recent = [m for m in prior[-HISTORY_TURNS:] if m.get("role") in ("user", "assistant")]
    if not recent:
        return ""
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {_text(m)}" for m in recent]
    return "# Conversation History\n" + "\n".join(lines)


def run(query: str, messages: List[Dict[str, Any]], *, language: Optional[str] = None,
        memory: Optional[str] = None) -> Dict[str, Any]:
    contextualized = contextualize(query, messages)
    flow(f"🔁 query → '{contextualized[:50]}'")
    chunks = retrieve(contextualized)
    flow(f"📄 retrieved {len(chunks)} chunk(s) from Ragie")
    return answer(contextualized, chunks, language=language,
                  history_text=_history_text(messages), memory=memory)

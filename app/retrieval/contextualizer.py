"""Layer 5 — query contextualization (Decision 11).

Ragie is stateless, so a follow-up like "and how do I delete it?" must be rewritten into a
self-contained query before retrieval. Ported from the old graph `contextualize_query` node.
"""
from typing import Any, Dict, List

from app.core.config import client
from app.core.logging import get_logger

logger = get_logger(__name__)

CONTEXTUALIZE_MODEL = "gpt-4o-mini"
HISTORY_TURNS = 4

_SYSTEM_PROMPT = """You are a query preprocessing assistant for the MCL knowledge base,
which covers BOTH the MCL mobile app and the MCL Dashboard (web admin).
Rewrite the user's latest question into a self-contained search query that can retrieve
the right documents without conversation context.

Rules:
- Resolve all pronouns ("it", "that", "them", "this") to the actual MCL entity.
- Expand follow-up questions into full standalone queries.
- If already self-contained, return it unchanged.
- Do NOT add platform, device, or surface words (e.g. "mobile app", "Dashboard", "iOS",
  "Android", "tablet", "web") that the user did not explicitly mention.
- Output ONLY the rewritten query, no explanation.
- Always output in English."""


def _text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(item.get("text", "") for item in content if item.get("type") == "text").strip()
    return str(content or "").strip()


def contextualize(query: str, messages: List[Dict[str, Any]]) -> str:
    prior = messages[:-1] if messages else []
    has_prior_user_turn = any(m.get("role") == "user" and _text(m) for m in prior)
    if len(messages) <= 1 or not has_prior_user_turn:
        return query

    recent = [m for m in prior[-HISTORY_TURNS:] if m.get("role") in ("user", "assistant")]
    history = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {_text(m)}" for m in recent
    )
    user_prompt = f"Conversation history (most recent first):\n{history}\n\nLatest user question: {query}"

    try:
        response = client.chat.completions.create(
            model=CONTEXTUALIZE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            timeout=15,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"[RETRIEVAL] contextualized '{query[:50]}' -> '{rewritten[:50]}'")
        return rewritten or query
    except Exception as e:
        logger.warning(f"[RETRIEVAL] contextualize failed, using original query: {e}")
        return query


_VISION_QUERY_PROMPT = """You are preparing a documentation search for the MCL (Mobile
Checklist) app. The user has shared a screenshot and possibly a message. Using BOTH the
screenshot and the conversation, write ONE standalone English search query that captures
exactly what MCL help the user needs. Resolve references like "here", "this", "that", "it"
by what is visible on the screen. Output ONLY the query — no preamble, no explanation."""


def build_vision_query(messages: List[Dict[str, Any]], *, history_turns: int = 6) -> str:
    """Turn a screenshot + conversation into one standalone text query for Ragie.

    Ragie is text-only, so the image is read here (by a vision model) and described into the
    query. Returns "" on failure so the caller can fall back to the raw user text.
    """
    recent = [
        m for m in messages[-history_turns:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not recent:
        return ""
    try:
        response = client.chat.completions.create(
            model=CONTEXTUALIZE_MODEL,
            messages=[{"role": "system", "content": _VISION_QUERY_PROMPT}, *recent],
            temperature=0,
            timeout=20,
        )
        query = (response.choices[0].message.content or "").strip()
        logger.info(f"[RETRIEVAL] vision query: '{query[:80]}'")
        return query
    except Exception as e:
        logger.warning(f"[RETRIEVAL] vision query build failed: {e}")
        return ""

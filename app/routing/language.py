"""Layer 2 — detect the language the answer should be written in.

Guided by the user's most recent messages (newest wins), so an English-quoting knowledge
base or the bot's own English replies can't pull the answer language off the user's. Short,
signal-free messages ("ok", "ja", an emoji) fall back to the most recent message that does
carry language signal.
"""
from typing import Any, Dict, List

from app.core.config import client
from app.core.logging import get_logger

logger = get_logger(__name__)

DETECT_MODEL = "gpt-4o-mini"
USER_TURNS = 3
DEFAULT_LANGUAGE = "English"

_SYSTEM_PROMPT = """You detect the language a reply should be written in for an MCL support chat.
You are given the user's most recent messages, oldest first, newest last.
Return the language of the NEWEST message that carries real language signal. If the newest
message is too short or ambiguous to tell (e.g. "ok", "ja", a number, an emoji, a bare URL),
fall back to the most recent earlier message that does carry signal.
Output ONLY the English name of the language, e.g. "German", "English", "Spanish"."""


def _text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(i.get("text", "") for i in content if i.get("type") == "text").strip()
    return str(content or "").strip()


def detect_language(messages: List[Dict[str, Any]]) -> str:
    user_texts = [t for t in (_text(m) for m in messages if m.get("role") == "user") if t]
    if not user_texts:
        return DEFAULT_LANGUAGE

    recent = user_texts[-USER_TURNS:]
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(recent))
    try:
        response = client.chat.completions.create(
            model=DETECT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
            temperature=0,
            timeout=10,
        )
        language = (response.choices[0].message.content or "").strip()
        logger.info(f"[LANGUAGE] detected '{language}' from {len(recent)} user message(s)")
        return language or DEFAULT_LANGUAGE
    except Exception as e:
        logger.warning(f"[LANGUAGE] detect failed, defaulting to {DEFAULT_LANGUAGE}: {e}")
        return DEFAULT_LANGUAGE

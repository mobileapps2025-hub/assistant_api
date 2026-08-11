"""Localize short, fixed, canned UI/status strings into the user's language.

These strings are Python literals that never pass through the model, so without this they stay
English. English passes through untouched; other languages get one cheap gpt-4o-mini
translation, cached — the vocabulary is ~a dozen fixed phrases, so each translates once per
language per process, then it's free. Falls back to the original English if translation fails
(e.g. the API is down) — the right behavior, since most of these strings are error messages.
"""
from functools import lru_cache

from app.core.config import client
from app.core.logging import get_logger

logger = get_logger(__name__)

_MODEL = "gpt-4o-mini"
_SYSTEM = (
    "Translate the user's text to {language}. Keep it short and natural. Keep product terms "
    "like 'MCL', 'Dashboard', 'Checklist', 'Task' in English. Output only the translation."
)


def _is_english(language: str) -> bool:
    return not language or language.strip().lower().startswith("en")


@lru_cache(maxsize=256)
def _translate(text: str, language: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM.format(language=language)},
                {"role": "user", "content": text},
            ],
            temperature=0,
            timeout=10,
        )
        return (resp.choices[0].message.content or "").strip() or text
    except Exception as e:
        logger.warning(f"[LOCALIZE] translate failed ({language}); using English: {e}")
        return text


def localize(text: str, language: str = "English") -> str:
    return text if _is_english(language) else _translate(text, language)

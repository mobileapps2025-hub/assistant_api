"""Layer 3 — the questions MarieClaire could not answer.

Recorded only when the knowledge search reported ``missing_information``: never by matching
words in the answer. Repeats collapse onto one row so the list is also a priority order.

Two ways in, chosen by configuration and never both:
  * ``GAP_SINK_URL`` set  -> forward to the central backend's protected endpoint. This is how a
    local MarieClaire reports, so no database credential ever leaves Azure.
  * otherwise             -> write straight to the database (this *is* the central backend).
"""
import hashlib
import re
import unicodedata
from typing import Optional

import httpx
from sqlalchemy import select, update

from app.core.config import GAP_INGEST_TOKEN, GAP_SINK_URL
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_QUESTION_CHARS = 500
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def fingerprint(question: str) -> str:
    """Stable id for a question, so the same ask in different wording collapses onto one row."""
    folded = unicodedata.normalize("NFKD", question).casefold()
    folded = _PUNCTUATION.sub(" ", folded)
    return hashlib.sha256(_WHITESPACE.sub(" ", folded).strip().encode()).hexdigest()


def _clean(question: str) -> str:
    return _WHITESPACE.sub(" ", question).strip()[:MAX_QUESTION_CHARS]


async def record_gap(question: str, language: Optional[str] = None) -> bool:
    """Log one unanswerable question. Never raises — a failure here must not break a reply."""
    question = _clean(question or "")
    if not question:
        return False
    try:
        if GAP_SINK_URL:
            return await _forward(question, language)
        return await _store(question, language)
    except Exception as err:
        logger.warning(f"[GAPS] could not record question: {err}")
        return False


async def _forward(question: str, language: Optional[str]) -> bool:
    headers = {"Authorization": f"Bearer {GAP_INGEST_TOKEN}"} if GAP_INGEST_TOKEN else {}
    async with httpx.AsyncClient(timeout=8) as http:
        response = await http.post(
            GAP_SINK_URL.rstrip("/") + "/api/gaps",
            json={"question": question, "language": language},
            headers=headers,
        )
    if response.status_code >= 400:
        logger.warning(f"[GAPS] sink refused the question: {response.status_code}")
        return False
    logger.info("[GAPS] question forwarded to the central list")
    return True


async def _store(question: str, language: Optional[str]) -> bool:
    from app.core.config import AsyncSessionLocal
    from app.core.database import DocumentationGap

    if AsyncSessionLocal is None:
        logger.warning("[GAPS] no database available — question not recorded")
        return False

    digest = fingerprint(question)
    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(select(DocumentationGap).where(DocumentationGap.fingerprint == digest))
        ).scalar_one_or_none()
        if existing:
            await session.execute(
                update(DocumentationGap)
                .where(DocumentationGap.id == existing.id)
                .values(times_asked=existing.times_asked + 1, last_asked_at=_now())
            )
            logger.info(f"[GAPS] question asked again ({existing.times_asked + 1}x)")
        else:
            session.add(DocumentationGap(fingerprint=digest, question=question, language=language))
            logger.info("[GAPS] new question recorded")
        await session.commit()
    return True


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)

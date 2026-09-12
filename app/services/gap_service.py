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


def fingerprint(question: str, kind: str = "missing") -> str:
    """Stable id for a question, so the same ask in different wording collapses onto one row.
    A correction and a missing report of the same question are kept separate by kind."""
    folded = unicodedata.normalize("NFKD", question).casefold()
    folded = _PUNCTUATION.sub(" ", folded)
    return hashlib.sha256(f"{kind}:{_WHITESPACE.sub(' ', folded).strip()}".encode()).hexdigest()


def _clean(question: str) -> str:
    return _WHITESPACE.sub(" ", question).strip()[:MAX_QUESTION_CHARS]


async def record_gap(question: str, language: Optional[str] = None,
                     surface: Optional[str] = None, role: Optional[str] = None,
                     kind: str = "missing", note: Optional[str] = None) -> bool:
    """Log an unanswered question (kind='missing') or a wrong-answer report (kind='correction').
    Never raises — a failure here must not break a reply."""
    question = _clean(question or "")
    if not question:
        return False
    try:
        if GAP_SINK_URL:
            return await _forward(question, language, surface, role, kind, note)
        return await _store(question, language, surface, role, kind, note)
    except Exception as err:
        logger.warning(f"[GAPS] could not record question: {err}")
        return False


async def discard_correction(question: str) -> bool:
    """Withdraw a correction a user filed (misclick, or they were wrong). Sets it discarded so
    nothing is acted on. Central (DB) only — corrections arrive through the platform, not a sink."""
    question = _clean(question or "")
    if not question:
        return False
    try:
        from app.core.config import AsyncSessionLocal
        from app.core.database import DocumentationGap
        if AsyncSessionLocal is None:
            return False
        digest = fingerprint(question, "correction")
        async with AsyncSessionLocal() as session:
            existing = (
                await session.execute(select(DocumentationGap).where(DocumentationGap.fingerprint == digest))
            ).scalar_one_or_none()
            if existing is None or existing.status == "discarded":
                return False
            await session.execute(
                update(DocumentationGap).where(DocumentationGap.id == existing.id).values(status="discarded"))
            await session.commit()
        logger.info("[GAPS] correction withdrawn")
        return True
    except Exception as err:
        logger.warning(f"[GAPS] could not withdraw correction: {err}")
        return False


async def _forward(question: str, language: Optional[str], surface: Optional[str], role: Optional[str],
                   kind: str, note: Optional[str]) -> bool:
    headers = {"Authorization": f"Bearer {GAP_INGEST_TOKEN}"} if GAP_INGEST_TOKEN else {}
    async with httpx.AsyncClient(timeout=8) as http:
        response = await http.post(
            GAP_SINK_URL.rstrip("/") + "/api/gaps",
            json={"question": question, "language": language, "surface": surface, "role": role,
                  "kind": kind, "note": note},
            headers=headers,
        )
    if response.status_code >= 400:
        logger.warning(f"[GAPS] sink refused the question: {response.status_code}")
        return False
    logger.info("[GAPS] question forwarded to the central list")
    return True


async def _store(question: str, language: Optional[str], surface: Optional[str], role: Optional[str],
                 kind: str, note: Optional[str]) -> bool:
    from app.core.config import AsyncSessionLocal
    from app.core.database import DocumentationGap

    if AsyncSessionLocal is None:
        logger.warning("[GAPS] no database available — question not recorded")
        return False

    digest = fingerprint(question, kind)
    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(select(DocumentationGap).where(DocumentationGap.fingerprint == digest))
        ).scalar_one_or_none()
        if existing:
            await session.execute(
                update(DocumentationGap)
                .where(DocumentationGap.id == existing.id)
                .values(times_asked=existing.times_asked + 1, last_asked_at=_now(),
                        status="pending", note=note or existing.note)   # a repeat re-opens it
            )
            logger.info(f"[GAPS] {kind} reported again ({existing.times_asked + 1}x)")
        else:
            session.add(DocumentationGap(
                fingerprint=digest, question=question, language=language, surface=surface,
                role=role, kind=kind, note=note))
            logger.info(f"[GAPS] new {kind} recorded")
        await session.commit()
    return True


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)

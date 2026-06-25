"""Layer 4 — post-answer guardrails (sanitize + log).

Deterministic checks the model can't bypass: a cited source that wasn't retrieved this turn
is stripped; an embedded image link that wasn't provided this turn is stripped. Both log.
"""
import re
from typing import Iterable, List, Tuple

from app.core.flow import flow
from app.core.logging import get_logger

logger = get_logger(__name__)

_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+?)\]", re.IGNORECASE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _source_known(cited: str, allowed: Iterable[str]) -> bool:
    c = cited.strip().lower()
    return any(c == a.lower() or c in a.lower() or a.lower() in c for a in allowed if a)


def enforce_citations(answer: str, allowed_sources: Iterable[str]) -> Tuple[str, List[str]]:
    allowed = list(allowed_sources)
    fabricated: List[str] = []

    def replace(match: "re.Match") -> str:
        cited = match.group(1).strip()
        if _source_known(cited, allowed):
            return match.group(0)
        fabricated.append(cited)
        return ""

    sanitized = _SOURCE_RE.sub(replace, answer)
    if fabricated:
        logger.warning(f"[ENFORCE] stripped {len(fabricated)} fabricated citation(s): {fabricated}")
    return sanitized, fabricated


def enforce_image_refs(answer: str, allowed_urls: Iterable[str]) -> Tuple[str, List[str]]:
    allowed = set(allowed_urls)
    removed: List[str] = []

    def replace(match: "re.Match") -> str:
        url = match.group(1).strip()
        if url in allowed:
            return match.group(0)
        removed.append(url)
        return ""

    sanitized = _IMAGE_RE.sub(replace, answer)
    if removed:
        logger.warning(f"[ENFORCE] stripped {len(removed)} unverified image link(s): {removed}")
    return sanitized, removed


def enforce_answer(answer: str, *, allowed_sources: Iterable[str], allowed_image_urls: Iterable[str]) -> str:
    answer, fabricated = enforce_citations(answer, allowed_sources)
    answer, removed = enforce_image_refs(answer, allowed_image_urls)
    flow(f"🛡 enforce: removed {len(fabricated)} bad citation(s), {len(removed)} bad image(s)")
    return re.sub(r"\n{3,}", "\n\n", answer).strip()

"""Layer 4 — pending confirmations for dangerous (write/destructive) actions.

A write/destructive tool call is not executed immediately: it's parked here until the user
approves it via /api/chat/confirm. In-memory with a short TTL — a pending confirmation lives
for seconds, so losing them on restart is acceptable (single-instance today). No token is
stored; the approve call re-supplies auth and we match on user_id.
"""
import time
import uuid
from typing import Any, Dict, List, Optional

TTL_SECONDS = 300

_PENDING: Dict[str, Dict[str, Any]] = {}


def _prune() -> None:
    cutoff = time.time() - TTL_SECONDS
    for cid in [c for c, p in _PENDING.items() if p["created_at"] < cutoff]:
        del _PENDING[cid]


def create_pending(
    tool: str,
    args: Dict[str, Any],
    user_id: Optional[str],
    summary: str,
    risk: str,
    language: str = "English",
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    _prune()
    cid = f"cfm_{uuid.uuid4().hex[:8]}"
    _PENDING[cid] = {
        "tool": tool, "args": args, "user_id": user_id, "summary": summary,
        "risk": risk, "language": language, "messages": messages or [],
        "created_at": time.time(),
    }
    return cid


def _valid(pending: Optional[Dict[str, Any]], user_id: Optional[str]) -> bool:
    return bool(pending and pending["user_id"] == user_id)


def peek_pending(confirmation_id: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the pending action without removing it (e.g. to read its language on reject)."""
    _prune()
    pending = _PENDING.get(confirmation_id)
    return pending if _valid(pending, user_id) else None


def take_pending(confirmation_id: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return and remove the pending action if it exists, hasn't expired, and belongs to the user."""
    _prune()
    pending = _PENDING.get(confirmation_id)
    if not _valid(pending, user_id):
        return None
    del _PENDING[confirmation_id]
    return pending

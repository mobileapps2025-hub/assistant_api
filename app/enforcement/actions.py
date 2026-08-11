"""Recent assistant action context for follow-up questions.

This is intentionally lightweight and in-memory, like pending confirmations. It gives the
next few turns a factual record of what the assistant actually executed, including defaults
that are otherwise hidden in handler code.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

TTL_SECONDS = 1800
MAX_ACTIONS_PER_KEY = 5

_ACTIONS: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}


def _key(session_id: Optional[str], user_id: Optional[str]) -> Tuple[str, str]:
    return (session_id or "", user_id or "")


def _prune() -> None:
    cutoff = time.time() - TTL_SECONDS
    for key in list(_ACTIONS):
        _ACTIONS[key] = [a for a in _ACTIONS[key] if a["created_at"] >= cutoff]
        if not _ACTIONS[key]:
            del _ACTIONS[key]


def _value(value: Any) -> str:
    if value is None or value == "":
        return "none"
    return str(value)


def _describe_action(tool: str, args: Dict[str, Any], summary: str) -> str:
    if tool == "add_task":
        description = _value(args.get("description"))
        due_date = _value(args.get("due_date"))
        market_id = _value(args.get("market_id"))
        assigned_user_id = args.get("assigned_user_id")
        explicit_target = bool(args.get("market_id") or assigned_user_id)
        assignment = _value(assigned_user_id) if assigned_user_id else (
            "current user" if not explicit_target else "not explicitly assigned"
        )
        return (
            f"{summary} Values used: description={description}; due_date={due_date}; "
            f"market={market_id}; assigned_user={assignment}; task_type=standard MCL task type."
        )
    return f"{summary} Arguments: {args}"


def record_action(
    session_id: Optional[str],
    user_id: Optional[str],
    tool: str,
    args: Dict[str, Any],
    summary: str,
) -> None:
    _prune()
    key = _key(session_id, user_id)
    actions = _ACTIONS.setdefault(key, [])
    actions.append({
        "tool": tool,
        "args": args,
        "summary": summary,
        "description": _describe_action(tool, args or {}, summary),
        "created_at": time.time(),
    })
    del actions[:-MAX_ACTIONS_PER_KEY]


def recall_action_context(session_id: Optional[str], user_id: Optional[str]) -> str:
    _prune()
    actions = _ACTIONS.get(_key(session_id, user_id), [])
    if not actions:
        return ""
    lines = ["# RECENT ASSISTANT ACTIONS"]
    for action in actions[-MAX_ACTIONS_PER_KEY:]:
        lines.append(f"- {action['tool']}: {action['description']}")
    return "\n".join(lines)


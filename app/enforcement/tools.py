"""Layer 4 — pre-tool security guardrail (block + audit).

The agent will touch production data, so tool execution is deny-by-default: only an explicit
allowlist of read-safe tools may run. Adding a tool to ``app/tools.py`` does NOT grant
execution — it must be added here and classified. Write/destructive tools stay denied until
explicitly allowed (none exist yet; the ACTION route is deferred). Every decision is audited.
"""
from dataclasses import dataclass
from typing import Optional

from app.core.logging import get_logger
from app.models import AuthContext

logger = get_logger(__name__)

READ_TOOLS = frozenset({
    "get_user_info",
    "get_user_markets",
    "get_user_checklists",
    "get_open_task_count",
})
ALLOWED_TOOLS = READ_TOOLS


@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    reason: str


def check_tool_call(tool_name: str, auth_context: Optional[AuthContext] = None) -> ToolDecision:
    if tool_name in ALLOWED_TOOLS:
        decision = ToolDecision(True, "allowlisted read tool")
    else:
        decision = ToolDecision(False, "denied: not in allowlist (deny-by-default)")
    user = auth_context.user_id if auth_context else None
    logger.info(
        f"[AUDIT] tool={tool_name} user={user} allowed={decision.allowed} reason={decision.reason}"
    )
    return decision

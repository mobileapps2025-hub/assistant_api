"""Layer 2 — Router/Classifier.

Decides which single path handles a text input — CHAT, KNOWLEDGE, or PERSONAL — using recent
conversation history, before any retrieval, tool call, or answer. It never answers.

One deterministic LLM call with structured output; no keyword heuristics. On any failure it
falls back to KNOWLEDGE, the safest grounded path.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from app.core.config import client
from app.core.logging import get_logger

logger = get_logger(__name__)

Route = Literal["CHAT", "KNOWLEDGE", "PERSONAL"]

ROUTING_MODEL = "gpt-4o-mini"
DEFAULT_ROUTE: Route = "KNOWLEDGE"
HISTORY_TURNS = 6

ROUTING_SYSTEM_PROMPT = """You route a user's message to exactly one handler. Read the whole conversation, but classify the user's LATEST message; earlier turns are context only (use them to resolve follow-ups like "and how do I delete it?").
PERSONAL — the user wants their OWN live data, which the assistant must fetch from the MCL service API using the user's session token: their profile/account, their company or role, the markets assigned to them, their own checklists, or how many open tasks they have. Typical phrasing: "my ...", "do I have ...", "who am I", "how many tasks do I have".
KNOWLEDGE — a GENERAL question about how the MCL app works that does NOT need the user's own data or any live service call, answerable from the documentation: how to use a feature, what something is or means, troubleshooting, platform/device differences, dashboards, sync issues.
CHAT — the user is just talking: greeting, thanks, small talk, testing the bot, asking about YOU (the bot), casual conversation.
Pick the single best route for the latest message."""

_ROUTE_SCHEMA = {
    "name": "route_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["CHAT", "KNOWLEDGE", "PERSONAL"]},
            "reason": {"type": "string"},
        },
        "required": ["route", "reason"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str


def _turn_content(message: Dict[str, Any]) -> Any:
    """Text for normal turns; the full multimodal content (text + image) when a turn carries
    a screenshot, so the classifier can route image messages by what's on screen."""
    content = message.get("content")
    if isinstance(content, list):
        if any(item.get("type") == "image_url" for item in content):
            return content
        return " ".join(item.get("text", "") for item in content if item.get("type") == "text").strip()
    return str(content or "").strip()


def _recent_turns(messages: List[Dict[str, Any]], history_turns: int) -> List[Dict[str, Any]]:
    turns = []
    for message in messages[-history_turns:]:
        if message.get("role") not in ("user", "assistant"):
            continue
        content = _turn_content(message)
        if content:
            turns.append({"role": message["role"], "content": content})
    return turns


def classify_route(messages: List[Dict[str, Any]], *, history_turns: int = HISTORY_TURNS) -> RouteDecision:
    turns = _recent_turns(messages, history_turns)
    if not turns or turns[-1]["role"] != "user":
        return RouteDecision("CHAT", "no user message to classify")

    try:
        response = client.chat.completions.create(
            model=ROUTING_MODEL,
            messages=[{"role": "system", "content": ROUTING_SYSTEM_PROMPT}, *turns],
            response_format={"type": "json_schema", "json_schema": _ROUTE_SCHEMA},
            temperature=0,
            timeout=10,
        )
        payload = json.loads(response.choices[0].message.content)
        return RouteDecision(payload["route"], payload.get("reason", ""))
    except Exception as e:
        logger.warning(f"[ROUTE] classification failed, defaulting to {DEFAULT_ROUTE}: {e}")
        return RouteDecision(DEFAULT_ROUTE, f"fallback: {e}")

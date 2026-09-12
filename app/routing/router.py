"""Layer 2 — Preflight classifier.

Labels the latest input as CHAT, KNOWLEDGE, or PERSONAL using recent conversation history.
It never answers and no longer locks normal text turns into one downstream path; the manager
agent uses this label for observability, auth pre-checks, and image routing.

One deterministic LLM call with structured output; no keyword heuristics. On any failure it
falls back to KNOWLEDGE, the safest product-support label.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from app.core.config import client
from app.core.logging import get_logger

logger = get_logger(__name__)

Route = Literal["CHAT", "ASSISTANT", "KNOWLEDGE", "PERSONAL"]

ROUTING_MODEL = "gpt-4o-mini"
DEFAULT_ROUTE: Route = "KNOWLEDGE"
HISTORY_TURNS = 6

_ROUTING_SYSTEM_TEMPLATE = """You preflight-classify a user's message for MarieClaire, an MCL (Mobile Checklist) support assistant.

Your job is to label the latest message, not answer it and not choose the final answer strategy. A manager agent will later decide whether to answer directly, search documentation, or use live MCL tools.

MarieClaire can:
- answer general MCL how-to questions by searching documentation,
- look up or change the user's OWN live data via these tools: {tools},
- chat, and explain what she herself is, can do, just did, or assumed.

Read the whole conversation, but classify the user's LATEST message; earlier turns are context (use them to resolve follow-ups like "and how do I delete it?" and references like "what do you mean by that?").

The four labels are about the user's INTENT — where the answer should come from:

CHAT — pure small talk with no MCL content: greetings, thanks, testing, how-are-you.
ASSISTANT — the message is about MarieClaire HERSELF: who she is, what she can or cannot do, her scope or limits ("what can you do", "what data can you get me", "can you delete a task?"), a reference back to her own words ("what do you mean by 'my data'?"), or what she just did, assumed, or which defaults she used. She answers these from her own self-knowledge, not from MCL documentation.
KNOWLEDGE — the user wants to know or do something in the MCL PRODUCT: how a feature works, steps to accomplish a task, what something means in MCL, troubleshooting, platform/device differences, dashboards, sync. "Can you help me create/set up/do X" belongs here — the user wants to be walked through the task, not told whether the assistant is capable.
PERSONAL — the user wants their OWN live data or an action on it that MarieClaire must reach via the tools above: their profile/account, company or role, assigned markets, their own checklists, their open task count, their company's actual configuration (its departments, its checklist questions, how its markets map to departments), or a create/edit/note/delete on their records. Typical phrasing: "my ...", "our ...", "do I have ...", "who am I". Choose PERSONAL when the user wants their company's concrete configuration or records; choose KNOWLEDGE when they ask how a feature works in general.

The key line: "what CAN you do" is ASSISTANT (about her); "help me DO it" is KNOWLEDGE or PERSONAL (about the task). Choose PERSONAL over KNOWLEDGE when the task is the user's own data or an action the tools above perform.
Pick the single best label for the latest message."""

_DEFAULT_TOOLS_SUMMARY = "the user's profile, their markets, their checklists, and their open task count"


def _render_tools(tools_catalog: List[Any]) -> str:
    names = []
    for tool in tools_catalog or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if name:
            names.append(name)
    return ", ".join(names) if names else _DEFAULT_TOOLS_SUMMARY


def _system_prompt(tools_catalog: List[Any]) -> str:
    return _ROUTING_SYSTEM_TEMPLATE.format(tools=_render_tools(tools_catalog))

_ROUTE_SCHEMA = {
    "name": "preflight_label_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["CHAT", "ASSISTANT", "KNOWLEDGE", "PERSONAL"]},
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
    a screenshot, so the classifier can label image messages by what's on screen."""
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


def classify_route(messages: List[Dict[str, Any]], *, history_turns: int = HISTORY_TURNS,
                   tools_catalog: List[Any] = None) -> RouteDecision:
    turns = _recent_turns(messages, history_turns)
    if not turns or turns[-1]["role"] != "user":
        return RouteDecision("CHAT", "no user message to classify")

    try:
        response = client.chat.completions.create(
            model=ROUTING_MODEL,
            messages=[{"role": "system", "content": _system_prompt(tools_catalog)}, *turns],
            response_format={"type": "json_schema", "json_schema": _ROUTE_SCHEMA},
            temperature=0,
            timeout=10,
        )
        payload = json.loads(response.choices[0].message.content)
        return RouteDecision(payload["route"], payload.get("reason", ""))
    except Exception as e:
        logger.warning(f"[PREFLIGHT] classification failed, defaulting to {DEFAULT_ROUTE}: {e}")
        return RouteDecision(DEFAULT_ROUTE, f"fallback: {e}")

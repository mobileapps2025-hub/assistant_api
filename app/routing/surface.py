"""Layer 2 — surface detection: is this how-to question about the web platform or the mobile app?

The two products are different: the web platform was rebuilt, the mobile app was not. So a
how-to answer must come from the matching knowledge. We start from where the user actually is
(the caller tells us) and only switch when the question clearly names the other product.

One small structured call, leaning to the caller's surface; on any doubt or failure it keeps
that surface.
"""
import json
from typing import Any, Dict, List, Optional

from app.core.config import client
from app.core.logging import get_logger
from app.kb_surfaces import APP, APP_SEARCH, WEB, WEB_SEARCH
from app.models import AuthContext, Device

logger = get_logger(__name__)

SURFACE_MODEL = "gpt-4o-mini"
HISTORY_TURNS = 6

_MOBILE_PLATFORMS = {"ios", "android"}

_SYSTEM = """You decide whether a user's MCL how-to question is about the WEB platform or the MOBILE APP.

MCL has two separate products:
- WEB — the browser platform (also called the dashboard): Overview, Reports, Data analysis, Tasks, Photo management, Safe counting; managing markets, departments, users; creating and configuring checklists.
- APP — the phone/tablet mobile app: running and completing checklists, syncing the device, on-site inspections.

The user is currently on: {base}. Strongly default to that surface — change only on a clear signal.

The bare word "app" is NOT a signal: people call the web platform "the app" too. "in the app", "in this app", "the app" alone keep the current surface.
Choose APP only on an explicit MOBILE cue: "on my phone", "on my tablet", "on my device", "the mobile app", "the phone app", "offline/sync on the device".
Choose WEB only on an explicit WEB cue: "in the browser", "on the web platform", "on my computer/desktop".
Also switch if the feature named lives only on the other product.
Use earlier turns to resolve follow-ups ("how do I create one from there?").

When in doubt, keep the current surface ({base}).

Return JSON: {{"surface": "web"}} or {{"surface": "app"}}."""


def base_surface(auth_context: Optional[AuthContext], device: Optional[Device]) -> str:
    if auth_context is not None and getattr(auth_context, "platform_turn", None):
        return WEB
    platform = (device.platform or "").lower() if device else ""
    if platform in _MOBILE_PLATFORMS:
        return APP
    if platform == "web":
        return WEB
    return APP


def _recent(messages: List[Dict[str, Any]]) -> str:
    prior = [m for m in messages[-HISTORY_TURNS:] if m.get("role") in ("user", "assistant")]
    lines = []
    for m in prior:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(i.get("text", "") for i in content if i.get("type") == "text")
        text = str(content or "").strip()
        if text:
            lines.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


def classify_surface(query: str, messages: List[Dict[str, Any]], base: str) -> str:
    if not query.strip():
        return base
    try:
        response = client.chat.completions.create(
            model=SURFACE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM.format(base=base)},
                {"role": "user", "content": f"{_recent(messages)}\n\nLatest: {query}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=15,
        )
        surface = json.loads(response.choices[0].message.content or "{}").get("surface", base)
        return surface if surface in (WEB, APP) else base
    except Exception as e:
        logger.error(f"[SURFACE] detection failed, keeping base '{base}': {e}")
        return base


def search_surfaces(surface: str):
    return WEB_SEARCH if surface == WEB else APP_SEARCH

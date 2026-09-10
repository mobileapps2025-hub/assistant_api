"""After MCL.Api executes (or the user dismisses) a proposal, MarieClaire writes the closing line.

No tools here: the outcome is already known, she only has to tell the user what happened in
their language and suggest the natural next step.
"""
import json
from typing import Any, Dict, List, Optional

from app.core.config import client
from app.core.flow import flow
from app.core.logging import get_logger
from app.instructions import get_system_prompt

logger = get_logger(__name__)

CLOSURE_MODEL = "gpt-4o"

CLOSURE_PROMPT = """You are MarieClaire right after the user confirmed or dismissed an action you proposed.
MCL has already carried it out (or not). Give the user useful closure in ONE or TWO short sentences,
in their language:
- executed: say what was created (mention the task number if given) and offer one natural next step.
- cancelled: acknowledge that nothing was changed, briefly.
- failed / refused: say plainly it did not happen and why, in everyday words, and what they can do.
Never claim anything beyond the outcome you were given. No headings, no lists."""


def write_closure(
    language: str,
    history: List[Dict[str, str]],
    proposal: Dict[str, Any],
    outcome: Dict[str, Any],
) -> Optional[str]:
    event = {
        "proposal": {"operation": proposal.get("operation"), "summary": proposal.get("summary"),
                     "args": proposal.get("args") or {}},
        "outcome": outcome,
    }
    messages = [
        {"role": "system", "content": get_system_prompt("agent", language=language)},
        {"role": "system", "content": CLOSURE_PROMPT},
        *[{"role": m["role"], "content": m["content"]} for m in history[-12:] if m.get("content")],
        {"role": "user", "content": "# ACTION RESULT\n" + json.dumps(event, ensure_ascii=False)},
    ]
    flow(f"🧾 closure for {proposal.get('operation')}: {outcome.get('status')} ({outcome.get('code')})")
    try:
        response = client.chat.completions.create(
            model=CLOSURE_MODEL, messages=messages, temperature=0, timeout=20,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception as err:
        logger.warning(f"[PLATFORM] closure failed: {err}")
        return None

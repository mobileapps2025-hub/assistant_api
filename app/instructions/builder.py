"""Layer 1 — Instruction File.

Single source of truth for MarieClaire's foundational instructions. Composes the system
prompt for each consumer as:

    CORE identity  +  mode addendum  +  optional dynamic slots
                                         (tool catalog, language directive, memory context)

The CORE identity (``core.md``) is always the foundation of every prompt and takes
precedence. Mode files (``chat.md`` / ``tools.md`` / ``rag.md``) only add mode-specific
behaviour on top of it.

This module is Layer 1 and stays decoupled from Layer 5: it never imports the tool
registry. The caller passes the live tool catalog in via ``tools_catalog`` so the prompt
always reflects the real toolset without duplicating it here.
"""
import contextvars
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Literal, Optional

Mode = Literal["chat", "tools", "rag", "vision", "agent"]

_INSTRUCTIONS_DIR = Path(__file__).parent
_VALID_MODES = ("chat", "tools", "rag", "vision", "agent")

# Modes that can receive retrieved KB context, and so may emit screenshot markers.
_SCREENSHOT_MODES = ("rag", "agent")

# The user's "today", resolved per request from their browser timezone and set once by the
# caller (see set_request_date). Read by every prompt so relative dates ("tomorrow") ground
# correctly. Falls back to the server date when the caller sets nothing.
_request_date: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_date", default=None
)


def set_request_date(formatted_date: Optional[str]) -> None:
    _request_date.set(formatted_date or None)


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    """Load and cache an instruction markdown file by stem (e.g. ``core``, ``rag``).

    Cached so each file is read from disk only once per process.
    """
    return (_INSTRUCTIONS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def _tools_block(tools_catalog: List[Any]) -> str:
    """Render the live tool catalog (OpenAI tool schema list) into a prompt section.

    Reads each tool's ``name`` and ``description`` from the standard
    ``{"type": "function", "function": {...}}`` shape; tolerates a flat ``{name, ...}``
    dict as a fallback. The rendered list — not any hardcoded copy — is what the agent
    treats as its source of truth for available tools.
    """
    lines = [
        "# YOUR AVAILABLE TOOLS",
        "These are the live tools you can use to look up the user's own MCL data. This "
        "list is the single source of truth — if a tool is not listed here, you do not "
        "have it.",
    ]
    for tool in tools_catalog:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        description = (fn.get("description") or "").strip()
        lines.append(f"- **{name}**: {description}")
    return "\n".join(lines)


def _language_directive(language: str) -> str:
    lang = language.strip().upper()
    return (
        "# LANGUAGE\n"
        f"Write your entire answer in **{lang}** — the language of the user's current message. "
        f"Do this even if earlier messages, the conversation so far, your own previous replies, "
        f"or the retrieved documentation were in another language: match **{lang}**, never the "
        f'history. Canonical MCL terms (e.g. "Dashboard", "Checklist", "Task") may stay in English.'
    )


def _device_directive(device: str) -> str:
    return (
        "# DEVICE\n"
        f"The user is on **{device.strip()}**. Answer for this platform and do not ask which "
        "platform they are on unless their question is genuinely ambiguous across platforms."
    )


def _memory_block(memory: str) -> str:
    return f"# MEMORY CONTEXT\n{memory.strip()}"


def _current_date_directive(current_date: Optional[str] = None) -> str:
    today = current_date or _request_date.get() or date.today().strftime("%A, %Y-%m-%d")
    return (
        "# CURRENT DATE\n"
        f"Today is **{today}**. Resolve every relative date (\"today\", \"tomorrow\", "
        "\"next Friday\") against this. Never treat any other date as today."
    )


def get_system_prompt(
    mode: Mode,
    *,
    language: Optional[str] = None,
    device: Optional[str] = None,
    memory: Optional[str] = None,
    tools_catalog: Optional[List[Any]] = None,
    current_date: Optional[str] = None,
) -> str:
    """Compose the full system prompt for a given consumer mode.

    Always starts from the CORE identity, then appends the mode-specific addendum, then any
    provided dynamic slots (tool catalog, language directive, memory context). Pure and
    deterministic for a given set of inputs.

    Args:
        mode: Which consumer the prompt is for — ``"chat"``, ``"tools"`` or ``"rag"``.
        language: When set, append a directive forcing the answer language.
        memory: When set, append the user's recalled memory context.
        tools_catalog: When set (a list of OpenAI tool schemas), append the live tool
            catalog so the agent can name exactly the tools it actually has.

    Raises:
        ValueError: If ``mode`` is not a recognised mode.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown instruction mode '{mode}'. Valid modes: {', '.join(_VALID_MODES)}."
        )

    sections = [_load("core"), _load(mode)]
    if mode in _SCREENSHOT_MODES:
        sections.append(_load("screenshots"))
    sections.append(_current_date_directive(current_date))
    if tools_catalog:
        sections.append(_tools_block(tools_catalog))
    if language:
        sections.append(_language_directive(language))
    if device:
        sections.append(_device_directive(device))
    if memory:
        sections.append(_memory_block(memory))
    return "\n\n".join(sections)

"""Capability tags — which permission a piece of knowledge needs. Mirrors surface filtering.

A unit tagged `everyone` is shown to all users. Any other tag is shown only when the user's
capability set (derived by MCL.Api from the company's role permissions) contains it. When the
turn carries no capabilities (a legacy mobile client), only `everyone` units are visible, so
nothing role-restricted leaks to a caller we cannot check.
"""
from typing import Iterable, Optional

EVERYONE = "everyone"


def capability_from_requires(requires: Optional[str], explicit: Optional[str] = None) -> str:
    """The capability a screen needs. An explicit tag wins; otherwise derive it from the
    human `requires` line the pages already carry."""
    if explicit:
        return explicit
    text = (requires or "").lower()
    if "checklist editor" in text:
        return "checklists.edit"
    if "company administrator" in text:
        return "company.admin"
    return EVERYONE


def visible(unit_capability: str, user_capabilities: Iterable[str]) -> bool:
    return unit_capability == EVERYONE or unit_capability in set(user_capabilities or ())

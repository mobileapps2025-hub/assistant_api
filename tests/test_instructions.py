"""Unit tests for Layer 1 — the Instruction File module (app/instructions).

These are the agreed pass/fail bar for Module 1: the composer must always start from the
CORE identity, add exactly the right mode addendum, and inject the optional dynamic slots
(tool catalog, language, memory) only when asked. Pure string assembly — no network, no key.
"""
import pytest

from datetime import date

from app import instructions
from app.instructions import builder, get_system_prompt, set_request_date
from app.tools import MCL_USER_TOOLS

# Stable substrings that identify each composed section.
CORE_MARKER = "MCL Support Specialist"
MODE_MARKERS = {
    "chat": "Casual conversation",
    "tools": "live MCL lookup",
    "rag": "SOURCE-BASED TRUTH",
    "vision": "Screenshot help",
}
# Unique to the *injected* tool catalog block (the mode files only mention the header in
# prose, so we key off the block's descriptive sentence, not the header).
CATALOG_SENTENCE = "These are the live tools you can use to look up"
SAMPLE_CATALOG = [
    {"type": "function", "function": {"name": "get_user_info", "description": "Profile."}},
    {"type": "function", "function": {"name": "get_open_task_count", "description": "Tasks."}},
]


@pytest.mark.parametrize("mode", ["chat", "tools", "rag", "vision"])
def test_core_identity_present_in_every_mode(mode):
    assert CORE_MARKER in get_system_prompt(mode)


@pytest.mark.parametrize("mode", ["chat", "tools", "rag", "vision"])
def test_mode_includes_only_its_own_addendum(mode):
    prompt = get_system_prompt(mode)
    assert MODE_MARKERS[mode] in prompt
    for other, marker in MODE_MARKERS.items():
        if other != mode:
            assert marker not in prompt


def test_language_directive_injected_only_when_requested():
    with_lang = get_system_prompt("rag", language="de")
    assert "# LANGUAGE" in with_lang
    assert "DE" in with_lang  # uppercased

    without_lang = get_system_prompt("rag")
    assert "# LANGUAGE" not in without_lang


def test_memory_block_injected_only_when_requested():
    with_memory = get_system_prompt("chat", memory="The user's name is Tomas.")
    assert "# MEMORY CONTEXT" in with_memory
    assert "The user's name is Tomas." in with_memory


def test_device_directive_injected_only_when_requested():
    with_device = get_system_prompt("rag", device="iOS phone")
    assert "# DEVICE" in with_device
    assert "iOS phone" in with_device

    assert "# DEVICE" not in get_system_prompt("rag")

    assert "# MEMORY CONTEXT" not in get_system_prompt("chat")


def test_tools_catalog_rendered_from_given_list():
    prompt = get_system_prompt("tools", tools_catalog=SAMPLE_CATALOG)
    assert CATALOG_SENTENCE in prompt
    assert "- **get_user_info**: Profile." in prompt
    assert "- **get_open_task_count**: Tasks." in prompt
    # The rendered block is absent when no catalog is supplied.
    assert CATALOG_SENTENCE not in get_system_prompt("tools")


def test_tools_catalog_is_not_hardcoded():
    """A tool that exists only in the passed catalog must surface — proving the prompt
    reflects the live registry rather than a hardcoded copy."""
    catalog = SAMPLE_CATALOG + [
        {"type": "function", "function": {"name": "get_future_widget", "description": "New."}}
    ]
    assert "get_future_widget" in get_system_prompt("tools", tools_catalog=catalog)


def test_real_registry_tools_all_appear():
    """Every tool in the real MCL_USER_TOOLS registry is named in the tools prompt."""
    prompt = get_system_prompt("tools", tools_catalog=MCL_USER_TOOLS)
    for tool in MCL_USER_TOOLS:
        assert tool["function"]["name"] in prompt


@pytest.mark.parametrize("mode", ["chat", "tools", "rag", "vision"])
def test_current_date_slot_always_present(mode):
    # Always on — the agent must never be left without a "today".
    assert "# CURRENT DATE" in get_system_prompt(mode)


def test_current_date_uses_explicit_override():
    prompt = get_system_prompt("chat", current_date="Wednesday, 2026-08-04")
    assert "Today is **Wednesday, 2026-08-04**" in prompt


def test_current_date_reads_request_contextvar():
    set_request_date("Monday, 2030-01-01")
    try:
        assert "Today is **Monday, 2030-01-01**" in get_system_prompt("rag")
    finally:
        set_request_date(None)


def test_explicit_current_date_beats_contextvar():
    set_request_date("Monday, 2030-01-01")
    try:
        prompt = get_system_prompt("chat", current_date="Wednesday, 2026-08-04")
        assert "2026-08-04" in prompt and "2030-01-01" not in prompt
    finally:
        set_request_date(None)


def test_current_date_falls_back_to_server_date_when_unset():
    set_request_date(None)
    today = date.today().strftime("%A, %Y-%m-%d")
    assert f"Today is **{today}**" in get_system_prompt("chat")


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        get_system_prompt("nonsense")


def test_instruction_files_are_cached():
    builder._load.cache_clear()
    builder._load("core")
    builder._load("core")
    info = builder._load.cache_info()
    assert info.hits >= 1  # second call served from cache
    assert info.misses == 1  # file read from disk exactly once


def test_public_api_surface():
    assert hasattr(instructions, "get_system_prompt")
    assert hasattr(instructions, "Mode")

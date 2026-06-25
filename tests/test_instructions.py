"""Unit tests for Layer 1 — the Instruction File module (app/instructions).

These are the agreed pass/fail bar for Module 1: the composer must always start from the
CORE identity, add exactly the right mode addendum, and inject the optional dynamic slots
(tool catalog, language, memory) only when asked. Pure string assembly — no network, no key.
"""
import pytest

from app import instructions
from app.instructions import builder, get_system_prompt
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

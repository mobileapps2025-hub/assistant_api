"""Unit tests for Layer 4 — Hooks & enforcement (app/enforcement).

Deterministic guardrails: fabricated citations and unverified image links are stripped;
tool execution is deny-by-default (only allowlisted read tools run; write/destructive denied).
"""
from app.enforcement import (
    ALLOWED_TOOLS,
    check_tool_call,
    enforce_answer,
    enforce_citations,
    enforce_image_refs,
)


# --- citation grounding ---

def test_known_citation_is_kept():
    answer = "Tap the + button [Source: wizard.pdf]."
    out, fabricated = enforce_citations(answer, {"wizard.pdf"})
    assert out == answer
    assert fabricated == []


def test_fabricated_citation_is_stripped():
    answer = "You can also teleport [Source: invented.pdf]."
    out, fabricated = enforce_citations(answer, {"wizard.pdf"})
    assert "[Source: invented.pdf]" not in out
    assert fabricated == ["invented.pdf"]


def test_citation_match_is_case_and_substring_tolerant():
    answer = "Export to Excel [Source: Mcl Dashboard Faq Revised.pdf]."
    out, fabricated = enforce_citations(answer, {"Mcl Dashboard Faq Revised.pdf"})
    assert fabricated == []
    assert "[Source: Mcl Dashboard Faq Revised.pdf]" in out


# --- image references ---

def test_allowed_image_kept_unverified_stripped():
    allowed = "http://host/api/ragie/image?document_id=D&chunk_id=C"
    answer = f"Here it is:\n![ok]({allowed})\n![fake](http://evil/x.png)"
    out, removed = enforce_image_refs(answer, {allowed})
    assert allowed in out
    assert "http://evil/x.png" not in out
    assert removed == ["http://evil/x.png"]


def test_enforce_answer_combines_both():
    allowed_url = "http://host/api/ragie/image?document_id=D&chunk_id=C"
    answer = (
        "Real step [Source: wizard.pdf].\n"
        "Fake step [Source: nope.pdf].\n"
        f"![real]({allowed_url})\n![fake](http://evil/x.png)"
    )
    out = enforce_answer(answer, allowed_sources={"wizard.pdf"}, allowed_image_urls={allowed_url})
    assert "[Source: wizard.pdf]" in out
    assert "nope.pdf" not in out
    assert allowed_url in out
    assert "evil" not in out


# --- tool security ---

def test_allowlisted_read_tools_pass():
    for name in ALLOWED_TOOLS:
        assert check_tool_call(name).allowed is True


def test_unknown_tool_denied_by_default():
    decision = check_tool_call("delete_everything")
    assert decision.allowed is False
    assert "deny-by-default" in decision.reason


def test_write_style_tool_denied():
    for name in ("delete_user", "update_checklist", "create_task", "drop_table"):
        assert check_tool_call(name).allowed is False

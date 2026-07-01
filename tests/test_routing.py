"""Unit tests for Layer 2 — the Router/Classifier (app/routing).

The OpenAI call is mocked: these verify the module's plumbing — structured-output parsing,
history assembly, the no-call short-circuits, and the error/parse fallback to KNOWLEDGE.
Real-LLM routing accuracy + determinism live in tests/routing_eval.py (run on demand).
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from app import routing
from app.routing import RouteDecision, classify_route


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def _decision_json(route: str, reason: str = "r") -> str:
    return json.dumps({"route": route, "reason": reason})


@pytest.mark.parametrize("route", ["CHAT", "KNOWLEDGE", "PERSONAL"])
def test_returns_route_from_structured_output(route):
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response(_decision_json(route))
        decision = classify_route([{"role": "user", "content": "anything"}])
    assert decision.route == route


def test_reason_is_captured():
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response(
            _decision_json("KNOWLEDGE", "general how-to")
        )
        decision = classify_route([{"role": "user", "content": "how do I create a checklist"}])
    assert decision.reason == "general how-to"


def test_no_user_message_returns_chat_without_calling_model():
    with patch("app.routing.router.client") as mock_client:
        decision = classify_route([{"role": "assistant", "content": "hi there"}])
    assert decision.route == "CHAT"
    mock_client.chat.completions.create.assert_not_called()


def test_empty_messages_returns_chat_without_calling_model():
    with patch("app.routing.router.client") as mock_client:
        decision = classify_route([])
    assert decision.route == "CHAT"
    mock_client.chat.completions.create.assert_not_called()


def test_blank_user_text_returns_chat_without_calling_model():
    with patch("app.routing.router.client") as mock_client:
        decision = classify_route([{"role": "user", "content": "   "}])
    assert decision.route == "CHAT"
    mock_client.chat.completions.create.assert_not_called()


def test_recent_history_is_sent_with_system_prompt():
    messages = [
        {"role": "user", "content": "how do I create a checklist"},
        {"role": "assistant", "content": "Use the Checklist Wizard."},
        {"role": "user", "content": "and how do I delete it?"},
    ]
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response(_decision_json("KNOWLEDGE"))
        classify_route(messages)
        sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["role"] == "system"
    assert [m["content"] for m in sent[1:]] == [
        "how do I create a checklist",
        "Use the Checklist Wizard.",
        "and how do I delete it?",
    ]


def test_history_turns_limit_truncates_old_turns():
    messages = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response(_decision_json("CHAT"))
        classify_route(messages, history_turns=2)
        sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["content"] for m in sent[1:]] == ["q8", "q9"]


def test_api_error_defaults_to_knowledge():
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        decision = classify_route([{"role": "user", "content": "q"}])
    assert decision.route == "KNOWLEDGE"
    assert decision.reason.startswith("fallback")


def test_invalid_json_defaults_to_knowledge():
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("not json at all")
        decision = classify_route([{"role": "user", "content": "q"}])
    assert decision.route == "KNOWLEDGE"
    assert decision.reason.startswith("fallback")


def test_multimodal_text_is_extracted():
    message = {"role": "user", "content": [{"type": "text", "text": "who am I"}]}
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response(_decision_json("PERSONAL"))
        classify_route([message])
        sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[-1]["content"] == "who am I"


def test_tool_catalog_and_capability_rule_reach_system_prompt():
    catalog = [{"type": "function", "function": {"name": "get_open_task_count", "description": "..."}}]
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response(_decision_json("CHAT"))
        classify_route([{"role": "user", "content": "what data can you get me?"}], tools_catalog=catalog)
        system_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "get_open_task_count" in system_prompt          # real capabilities injected
    assert "ASSISTANT" in system_prompt                    # capability-vs-product rule present


def test_system_prompt_has_default_tools_summary_without_catalog():
    with patch("app.routing.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response(_decision_json("CHAT"))
        classify_route([{"role": "user", "content": "hi"}])
        system_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "open task count" in system_prompt


def test_public_api_surface():
    assert hasattr(routing, "classify_route")
    assert hasattr(routing, "Route")
    assert hasattr(routing, "RouteDecision")
    assert isinstance(RouteDecision("CHAT", "x"), RouteDecision)

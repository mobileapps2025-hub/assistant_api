"""Unit tests for Layer 2 — language detection (app/routing/language)."""
from unittest.mock import MagicMock, patch

from app.routing.language import detect_language


def _response(content):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_no_user_messages_defaults_without_llm():
    with patch("app.routing.language.client") as mock_client:
        assert detect_language([{"role": "assistant", "content": "hi"}]) == "English"
    mock_client.chat.completions.create.assert_not_called()


def test_sends_only_recent_user_messages_newest_last():
    messages = [
        {"role": "user", "content": "erste Frage"},
        {"role": "assistant", "content": "an English reply that must not bias detection"},
        {"role": "user", "content": "zweite Frage"},
        {"role": "user", "content": "dritte Frage"},
        {"role": "user", "content": "vierte Frage"},
    ]
    with patch("app.routing.language.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("German")
        result = detect_language(messages)
        sent = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert result == "German"
    assert "erste Frage" not in sent  # only the last 3 user turns
    assert "English reply" not in sent  # assistant turns excluded
    assert sent.strip().endswith("vierte Frage")  # newest last


def test_llm_failure_defaults_to_english():
    with patch("app.routing.language.client") as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        assert detect_language([{"role": "user", "content": "hallo"}]) == "English"

"""Unit tests for the canned-string localizer (app/core/localize)."""
from unittest.mock import MagicMock, patch

from app.core import localize as loc


def test_english_passthrough_makes_no_call():
    with patch("app.core.localize.client") as client:
        for lang in ("English", "en", "EN", ""):
            assert loc.localize("Hello", lang) == "Hello"
        client.chat.completions.create.assert_not_called()


def test_non_english_translates_and_caches():
    loc._translate.cache_clear()
    with patch("app.core.localize.client") as client:
        resp = MagicMock()
        resp.choices[0].message.content = "Hola"
        client.chat.completions.create.return_value = resp
        assert loc.localize("Hello", "Spanish") == "Hola"
        assert loc.localize("Hello", "Spanish") == "Hola"          # served from cache
        client.chat.completions.create.assert_called_once()        # translated once, not twice


def test_translate_failure_falls_back_to_english():
    loc._translate.cache_clear()
    with patch("app.core.localize.client") as client:
        client.chat.completions.create.side_effect = RuntimeError("api down")
        assert loc.localize("Please try again", "German") == "Please try again"

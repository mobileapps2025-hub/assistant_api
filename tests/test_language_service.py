import pytest
from unittest.mock import MagicMock, patch
from lingua import Language
from app.services.language_service import LanguageService


@pytest.fixture
def language_service():
    """Build a LanguageService with a mocked Lingua detector."""
    with patch("app.services.language_service.LanguageDetectorBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder_cls.from_languages.return_value = mock_builder
        mock_detector = MagicMock()
        mock_builder.build.return_value = mock_detector

        service = LanguageService()
        service.detector = mock_detector  # Keep reference for test assertions
        return service


class TestLanguageServiceDetection:

    def test_returns_en_for_empty_string(self, language_service):
        result = language_service.detect_language("")
        assert result == "en"
        language_service.detector.detect_language_of.assert_not_called()

    def test_detects_english(self, language_service):
        language_service.detector.compute_language_confidence_values.return_value = [
            MagicMock(language=Language.ENGLISH, value=0.95)
        ]

        result = language_service.detect_language("How do I create a checklist?")
        assert result == "en"

    def test_detects_german(self, language_service):
        language_service.detector.compute_language_confidence_values.return_value = [
            MagicMock(language=Language.GERMAN, value=0.95)
        ]

        result = language_service.detect_language("Wie erstelle ich eine Checkliste?")
        assert result == "de"

    def test_defaults_to_en_when_detection_returns_none(self, language_service):
        language_service.detector.compute_language_confidence_values.return_value = []

        result = language_service.detect_language("???")
        assert result == "en"

    def test_defaults_to_en_on_exception(self, language_service):
        language_service.detector.compute_language_confidence_values.side_effect = Exception("detector crash")

        result = language_service.detect_language("some text")
        assert result == "en"

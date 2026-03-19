import logging
from typing import Optional
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

class LanguageService:
    def __init__(self):
        # We focus on the languages relevant to the project to improve accuracy and performance
        # adding English, German, Spanish, French, Italian, Portuguese
        languages = [
            Language.ENGLISH, 
            Language.GERMAN, 
            Language.SPANISH, 
            Language.FRENCH, 
            Language.ITALIAN, 
            Language.PORTUGUESE
        ]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
        logger.info("Language Service initialized with Lingua (EN, DE, ES, FR, IT, PT)")

    # Minimum confidence Lingua must have to classify a query as non-English.
    # Lingua can misclassify short technical queries (e.g. "How do I sync data on Android?")
    # as German because loanwords like "sync", "data", "Android" appear in both languages.
    # Requiring ≥0.75 confidence for non-English prevents these false positives while still
    # correctly detecting clearly non-English text (e.g. "Wie erstelle ich eine Checkliste?").
    NON_ENGLISH_CONFIDENCE_THRESHOLD = 0.75

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text.
        Returns the ISO 639-1 code (e.g., 'en', 'de', 'es').
        Defaults to 'en' if detection fails, text is empty, or confidence is too low.
        """
        if not text:
            return "en"

        try:
            confidence_values = self.detector.compute_language_confidence_values(text)

            if not confidence_values:
                logger.warning(f"Language detection returned no results for: '{text[:40]}'. Defaulting to 'en'.")
                return "en"

            top = confidence_values[0]
            top_lang = top.language
            top_confidence = top.value

            # If top language is already English, return immediately
            if top_lang == Language.ENGLISH:
                logger.info(f"[TRACE] language_service: detected='en' confidence={top_confidence:.3f} text='{text[:40]}'")
                return "en"

            # For non-English: require high confidence before committing.
            # Technical/short queries with loanwords can fool Lingua into non-English
            # classifications with low confidence.
            if top_confidence < self.NON_ENGLISH_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"[TRACE] language_service: top='{top_lang.iso_code_639_1.name.lower()}' "
                    f"confidence={top_confidence:.3f} < threshold={self.NON_ENGLISH_CONFIDENCE_THRESHOLD} "
                    f"→ defaulting to 'en' | text='{text[:40]}'"
                )
                return "en"

            detected = top_lang.iso_code_639_1.name.lower()
            logger.info(f"[TRACE] language_service: detected='{detected}' confidence={top_confidence:.3f} text='{text[:40]}'")
            return detected

        except Exception as e:
            logger.error(f"Error during language detection: {e}")
            return "en"

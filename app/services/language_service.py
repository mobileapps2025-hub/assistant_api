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

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text.
        Returns the ISO 639-1 code (e.g., 'en', 'de', 'es').
        Defaults to 'en' if detection fails or text is empty.
        """
        if not text:
            return "en"
            
        try:
            language = self.detector.detect_language_of(text)
            if language:
                # Convert Lingua language enum to ISO 639-1 string (lowercase)
                # Lingua returns e.g. Language.ENGLISH
                # We can map it or use iso_code_639_1.name.lower()
                return language.iso_code_639_1.name.lower()
            else:
                logger.warning(f"Language detection failed for text: '{text[:20]}...'. Defaulting to 'en'.")
                return "en"
        except Exception as e:
            logger.error(f"Error during language detection: {e}")
            return "en"

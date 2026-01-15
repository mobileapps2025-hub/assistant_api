from app.services.language_service import LanguageService

def test_language_detection():
    print("Initializing LanguageService...")
    service = LanguageService()
    
    test_cases = [
        ("Hello, how are you?", "en"),
        ("Hola, como estas?", "es"),
        ("Was kostet das?", "de"),
        ("Bonjour, comment ça va?", "fr"),
        ("Ciao, come stai?", "it"),
        ("Obrigado", "pt"), # Might be ambiguous, let's see
        ("This is a mixed sentence but mostly English.", "en"),
        ("Das ist ein Test.", "de")
    ]
    
    print("\nRunning detection tests:")
    for text, expected in test_cases:
        detected = service.detect_language(text)
        status = "✅" if detected == expected else f"❌ (Expected {expected})"
        print(f"[{status}] Text: '{text}' -> Detected: {detected}")

if __name__ == "__main__":
    test_language_detection()

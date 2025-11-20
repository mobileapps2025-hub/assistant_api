"""Utilities for deriving situational context from chat transcripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


INTERFACE_KEYWORDS: Dict[str, List[str]] = {
    "app": [
        "app",
        "mobile",
        "handy",
        "telefon",
        "smartphone",
        "phone",
        "tablet",
        "ios",
        "android",
        "screen",
        "screenshot",
        "ipad",
        "iphone",
        "samsung",
        "pixel",
        "huawei",
        "portrait",
        "landscape",
    ],
    "web": [
        "web",
        "browser",
        "desktop",
        "portal",
        "dashboard",
        "site",
        "pc",
        "webui",
        "web-ui",
        "chrome",
        "firefox",
        "edge",
        "safari",
    ],
}

LAYOUT_KEYWORDS: Dict[str, List[str]] = {
    "phone": ["phone", "smartphone", "handy", "portrait", "iphone", "android"],
    "tablet": ["tablet", "ipad", "landscape"],
}

OS_KEYWORDS: Dict[str, List[str]] = {
    "ios": ["ios", "iphone", "ipad", "apple"],
    "android": ["android", "samsung", "pixel", "huawei"],
}

NOVICE_SIGNALS = [
    "new",
    "first time",
    "nicht sicher",
    "kann nicht",
    "don't know",
    "hilfe",
    "help",
    "wo finde",
]

ADVANCED_SIGNALS = [
    "again",
    "already",
    "erneut",
    "advanced",
    "power user",
    "expert",
]

QUESTION_WORDS = [
    "how",
    "what",
    "where",
    "wie",
    "was",
    "wo",
    "kann",
    "could",
    "can",
]

CLARIFICATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "interface": {
        "en": "Are you currently using the MCL mobile app or the web dashboard?",
        "de": "Verwendest du gerade die MCL App oder das Web-Dashboard?",
    },
    "layout": {
        "en": "Are you on a phone-sized layout or on a tablet?",
        "de": "Nutzt du ein Smartphone-Layout oder ein Tablet?",
    },
    "os": {
        "en": "If you are in the app, is it on iOS or Android?",
        "de": "Falls du in der App bist: Läuft sie auf iOS oder Android?",
    },
    "intent": {
        "en": "Could you briefly describe what you want to achieve so I can guide you step by step?",
        "de": "Kannst du kurz beschreiben, was du genau erreichen möchtest? Dann kann ich dich Schritt für Schritt führen.",
    },
}

CONFIDENCE_NORMALIZER = 2.0
MIN_CONFIDENCE_FOR_ASSUMPTION = 0.4


def _flatten_user_text(messages: List[Dict[str, object]]) -> str:
    """Concatenate all user text snippets from the transcript."""
    chunks: List[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_val = item.get("text")
                    if text_val:
                        chunks.append(str(text_val))
        elif isinstance(content, str):
            chunks.append(content)
    return " \n".join(chunks).strip()


def _score_keywords(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    score = 0.0
    for keyword in keywords:
        if keyword and keyword in text_lower:
            score += 1.0
    return score


def _normalize(score: float) -> float:
    return max(0.0, min(1.0, score / CONFIDENCE_NORMALIZER))


@dataclass
class ContextAnalysis:
    interface: Optional[str] = None
    interface_confidence: float = 0.0
    layout: Optional[str] = None
    layout_confidence: float = 0.0
    os: Optional[str] = None
    os_confidence: float = 0.0
    user_expertise: Optional[str] = None
    expertise_confidence: float = 0.0
    intent_clarity: str = "unknown"
    latest_question: str = ""
    clarification_keys: List[str] = field(default_factory=list)

    def needs_clarification(self) -> bool:
        return bool(self.clarification_keys)

    def build_summary(self) -> str:
        parts: List[str] = []
        if self.interface:
            parts.append(
                f"Interface guess: {self.interface} ({self.interface_confidence*100:.0f}% confidence)"
            )
        else:
            parts.append("Interface: unknown")

        if self.layout:
            parts.append(
                f"Layout: {self.layout} ({self.layout_confidence*100:.0f}% confidence)"
            )

        if self.os:
            parts.append(f"OS: {self.os} ({self.os_confidence*100:.0f}% confidence)")

        if self.user_expertise:
            parts.append(
                f"User experience level: {self.user_expertise} ({self.expertise_confidence*100:.0f}% confidence)"
            )

        if self.intent_clarity != "clear":
            parts.append(f"Intent clarity: {self.intent_clarity}")

        if not parts:
            return "No situational signals identified."
        return " | ".join(parts)

    def prompt_block(self, language: str = "en") -> str:
        summary = self.build_summary()
        if language == "de":
            header = "Situationsanalyse:"
            guidance_label = "Vorgaben:"
            guardrails = [
                "Bestätige Annahmen mit weniger als 65 % Sicherheit zuerst mit dem Benutzer.",
                "Beschreibe nicht gleichzeitig App- und Web-Lösungen, außer der Benutzer bittet ausdrücklich darum.",
            ]
        else:
            header = "Situational context assessment:"
            guidance_label = "Guidance:"
            guardrails = [
                "Confirm with the user before relying on any assumption with confidence below 65%.",
                "Do not describe both mobile and web solutions simultaneously unless the user asks for a comparison.",
            ]

        prompt_lines = [header, summary, guidance_label]
        prompt_lines.extend(f"- {rule}" for rule in guardrails)
        return "\n".join(prompt_lines)

    def clarification_prompt(self, language: str = "en") -> str:
        if not self.needs_clarification():
            return ""
        lang = language if language in {"en", "de"} else "en"
        questions: List[str] = []
        for key in self.clarification_keys[:2]:
            question = CLARIFICATION_TEMPLATES.get(key, {}).get(lang)
            if question:
                questions.append(question)

        if not questions:
            return ""

        intro = {
            "en": "To give you actionable guidance I just need a quick clarification:",
            "de": "Damit ich dir konkret helfen kann, brauche ich noch kurz folgende Info:",
        }[lang]

        bullet_lines = [f"{idx+1}. {question}" for idx, question in enumerate(questions)]
        return intro + "\n" + "\n".join(bullet_lines)


def analyze_situational_context(messages: List[Dict[str, object]]) -> ContextAnalysis:
    """Derive situational context signals from the full conversation history."""
    analysis = ContextAnalysis()

    flattened_text = _flatten_user_text(messages)
    if not flattened_text:
        analysis.clarification_keys.append("intent")
        analysis.intent_clarity = "missing"
        return analysis

    latest_user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                text_items = [item.get("text", "") for item in content if isinstance(item, dict)]
                latest_user_message = " ".join(text_items).strip()
            elif isinstance(content, str):
                latest_user_message = content.strip()
            break
    analysis.latest_question = latest_user_message

    # Helper to calculate score with boost for latest message
    def get_boosted_score(keywords: List[str]) -> float:
        base_score = _score_keywords(flattened_text, keywords)
        # Boost if keywords appear in the latest message to prioritize immediate user intent/answers
        if _score_keywords(latest_user_message, keywords) > 0:
            return base_score + 3.0
        return base_score

    # Interface detection
    interface_scores = {
        name: get_boosted_score(keywords)
        for name, keywords in INTERFACE_KEYWORDS.items()
    }
    interface_label = max(interface_scores, key=interface_scores.get)
    interface_conf = _normalize(interface_scores[interface_label])

    if interface_conf >= MIN_CONFIDENCE_FOR_ASSUMPTION:
        analysis.interface = interface_label
        analysis.interface_confidence = interface_conf
    else:
        analysis.clarification_keys.append("interface")

    # Layout detection only matters for app
    if analysis.interface == "app":
        layout_scores = {
            name: get_boosted_score(keywords)
            for name, keywords in LAYOUT_KEYWORDS.items()
        }
        layout_label = max(layout_scores, key=layout_scores.get)
        layout_conf = _normalize(layout_scores[layout_label])
        if layout_conf >= MIN_CONFIDENCE_FOR_ASSUMPTION:
            analysis.layout = layout_label
            analysis.layout_confidence = layout_conf
        else:
            analysis.clarification_keys.append("layout")

        os_scores = {
            name: get_boosted_score(keywords)
            for name, keywords in OS_KEYWORDS.items()
        }
        os_label = max(os_scores, key=os_scores.get)
        os_conf = _normalize(os_scores[os_label])
        if os_conf >= MIN_CONFIDENCE_FOR_ASSUMPTION:
            analysis.os = os_label
            analysis.os_confidence = os_conf
        else:
            analysis.clarification_keys.append("os")

    # Expertise
    novice_score = _score_keywords(flattened_text, NOVICE_SIGNALS)
    advanced_score = _score_keywords(flattened_text, ADVANCED_SIGNALS)
    if novice_score > advanced_score and novice_score:
        analysis.user_expertise = "novice"
        analysis.expertise_confidence = _normalize(novice_score)
    elif advanced_score > novice_score and advanced_score:
        analysis.user_expertise = "advanced"
        analysis.expertise_confidence = _normalize(advanced_score)

    # Intent clarity heuristics
    latest_lower = latest_user_message.lower()
    
    # Check if we have a valid question anywhere in the history
    has_question_in_history = any(word in flattened_text.lower() for word in QUESTION_WORDS)
    
    if (not latest_user_message or len(latest_user_message) < 10) and not has_question_in_history:
        analysis.intent_clarity = "missing"
        if "intent" not in analysis.clarification_keys:
            analysis.clarification_keys.append("intent")
    elif not any(word in latest_lower for word in QUESTION_WORDS) and not has_question_in_history:
        analysis.intent_clarity = "ambiguous"
        if "intent" not in analysis.clarification_keys:
            analysis.clarification_keys.append("intent")
    else:
        analysis.intent_clarity = "clear"

    # Avoid redundant clarifications when interface is unknown
    if "interface" in analysis.clarification_keys:
        analysis.clarification_keys = ["interface"]

    return analysis

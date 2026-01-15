# Technical Research Report: Optimal Implementation for Automatic Language Detection and Localization

**Date:** January 15, 2026
**Prepared by:** BMad
**Project Context:** The current MCL AI Agent (Spotplan) responds in English even when queried in Spanish or German, unless explicitly prompted. The goal is to auto-detect the user's language and respond in that same language, utilizing knowledge that may exist in various languages but delivering it in the user's preferred idiom.

---

## Executive Summary

{{recommendations}}

### Key Recommendation

**Primary Choice:** [Technology/Pattern Name]

**Rationale:** [2-3 sentence summary]

**Key Benefits:**

- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

---

## 1. Research Objectives

### Technical Question

What is the optimal implementation strategy for automatic language detection and response localization in the MCL AI Agent?

### Project Context

The Agent currently functions primarily in English. Users asking questions in other languages (e.g., Spanish, German) receive English responses unless they specifically instruct otherwise. The system essentially understands the intent but defaults to English for output. We need a robust mechanism to:
1. Identify the input language.
2. Ensure the Retrieval Augmented Generation (RAG) process respects this context or translates correctly.
3. Generate the final response in the user's native language automatically.

### Requirements and Constraints

#### Functional Requirements

- **Language Detection:** System must strictly identify the input language (e.g., 'es', 'de', 'en') from the user's query string.
- **Context Handling:** The system must maintain this language preference for the session or turn.
- **Response Generation:** The LLM must be instructed (via system prompt or context) to reply in the detected language.
- **Knowledge Retrieval:** The search/retrieval mechanism should ideally find relevant documents even if the query is in a different language than the docs (Cross-lingual search) OR translate the query for search.

#### Non-Functional Requirements

- **Latency:** Detection must be fast (<100ms) to avoids slowing down the chat interface.
- **Accuracy:** High accuracy in detection to avoid jarring user experiences (e.g., replying in German to a Spanish query).
- **Maintainability:** Minimal changes to the core workflow logic.

#### Technical Constraints

- **Stack:** Python, FastAPI, Docker.
- **Offline Preference:** Prefer local libraries for detection (e.g., `langdetect`, `fasttext`) over external API calls if possible, to reduce latency and costs.

---

## 2. Technology Options Evaluated

The following technologies were evaluated for local, offline language detection suitable for a Python containerized environment:

1.  **Lingua (lingua-py)**: A high-accuracy library (Python bindings for Rust) specifically designed for short text and mixed languages.
2.  **FastText (Facebook)**: A widely used, efficient library for text classification and representation.
3.  **Langdetect**: A pure Python port of Google's language-detection library.
4.  **LLM-Based Detection**: Using the existing LLM (Zero-shot) to classify language before processing.
5.  **Azure AI Language / AWS Comprehend**: Cloud-based API solutions (Evaluated for baseline comparison only).

---

## 3. Detailed Technology Profiles

### Option 1: Lingua (lingua-py)

**Description:** Python bindings for a Rust-based language detection library. It focuses on high accuracy for short text (like chat queries) and mixed-language text.
**Pros:**
- **Accuracy on Short Text:** Significantly outperforms others on single words or short phrases (< 12 words), which is critical for chat interfaces.
- **Offline/Privateness:** No external API calls; runs locally.
- **Speed:** Rust implementation offers high performance (multi-threaded support available).
- **Control:** Allows restricting the set of languages to expected ones (e.g., European languages) to improve accuracy further.
**Cons:**
- **Memory Usage:** Can be higher (800MB-1GB) if all 75 languages are eager-loaded in high-accuracy mode, though "low accuracy" mode drops this to ~100MB.
- **Project Maturity:** Good, but fewer contributors than FastText.

### Option 2: FastText (Facebook)

**Description:** A library for efficient text classification and representation learning.
**Pros:**
- **Speed:** Extremely fast inference, suitable for massive scale.
- **Language Support:** Supports over 176 languages.
- **Community:** Massive adoption, battle-tested.
**Cons:**
- **Short Text Accuracy:** Struggles more with very short inputs (1-3 words) compared to Lingua.
- **Model Size:** Requires managing significant binary model files (`lid.176.bin` is ~126MB, `.ftz` is smaller).
- **Usability:** Requires manually downloading and managing model files in the Docker container.

### Option 3: Langdetect

**Description:** Pure Python port of Google's language-detection library.
**Pros:**
- **Simplicity:** Pure Python, easy `pip install`, no compilation or external model files to manage.
- **Lightweight:** Negligible disk footprint.
**Cons:**
- **Determinism:** Non-deterministic algorithm (can return different results for the same short text).
- **Performance:** Slower than C++/Rust implementations.
- **Accuracy:** Often fails on very short text.

### Option 4: LLM-Based Detection (Zero-shot)

**Description:** Asking the LLM (e.g., GPT-4o, Claude 3.5 Sonnet) effectively: "What language is this? {text}".
**Pros:**
- **Context Awareness:** Can understand nuance and intent even in mixed inputs.
- **No New Deps:** Uses existing infrastructure.
**Cons:**
- **Latency:** Adds a full LLM round-trip (500ms - 2s) to *every* interaction before processing begins.
- **Cost:** Increases token usage.
- **Overkill:** Using a sledgehammer to crack a nut.

---

## 4. Comparative Analysis

| Feature | Lingua (Rust/Py) | FastText (C++) | Langdetect (Python) | LLM (API) |
| :--- | :--- | :--- | :--- | :--- |
| **Short Text Accuracy** | **High** (Best in Class) | Medium | Low | Very High |
| **Latency** | < 1ms | < 1ms | ~10ms | > 500ms |
| **Memory Footprint** | ~100MB (Low Acc) - 1GB | ~150MB | < 10MB | N/A (Cloud) |
| **Offline capable** | Yes | Yes | Yes | No (usually) |
| **Setup Complexity** | Low (`pip install`) | Medium (Model mgmt) | Very Low | None |

### Weighted Analysis

**Decision Priorities:** 
1. **Accuracy on Short Queries:** (Critical) Users often type "Hola" or "Was kostet das?".
2. **Latency:** (High) Must not degrade chat responsiveness.
3. **Implementation Simplicity:** (Medium)

**Winner:** **Lingua** clearly stands out for the chatbot use case because its rule-based + statistical approach is specifically tuned for short, ambiguous text where standard n-gram models (FastText) might guess wrong or return low confidence.

---

## 5. Recommendations

{{recommendations}}

### Key Recommendation

**Primary Choice:** **Lingua (lingua-py)**

**Rationale:** The user's specific problem involves a chat interface. Chat inputs are often short, informal, and lack context. Lingua provides the highest accuracy for this specific data profile while maintaining negligible latency and running offline.

**Key Benefits:**
- **Solves the "Hola" problem:** Correctly identifies language even on single words.
- **Privacy:** Reasoning stays local.
- **Speed:** Doesn't block the user experience.

### Implementation Roadmap

1.  **Add Dependency:** Add `lingua-language-detector` to `requirements.txt`.
2.  **Create Service:** Implement a `LanguageDetectionService` class.
    - Initialize the `LanguageDetectorbuilder` with a focused set of Expected Languages (English, Spanish, German, French, Portuguese) to effectively reduce false positives.
3.  **Integrate in Chat Flow:**
    - In `chat_service.py`, call detection on the incoming user message.
    - Pass the detected language code (e.g., "Spanish") to the Prompt Construction logic.
4.  **Prompt Engineering:**
    - Update system prompts to include: `User Language: {detected_language}. You must answer in {detected_language}.`
    - (Advanced) If RAG documents are English-only: implementing a "Translate-Search-Answer" pattern may be needed.

### Risk Mitigation

- **Memory Usage:** We will use `LanguageDetectorBuilder.from_iso_codes_639_1` to load ONLY the languages we expect to support (e.g., EN, ES, DE, IT, FR, PT), rather than `from_all_languages()`. This drastically reduces memory footprint vs loading all 75 languages.
- **Fallback:** If confidence is Low (< 0.5), default to English.

---

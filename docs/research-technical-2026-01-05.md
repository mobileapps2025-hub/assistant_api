# Technical Research: Handling Unknown Terms in AI Agent

**Date:** 2026-01-05
**Author:** BMad Analyst (Mary)
**Context:** Spotplan (Brownfield)

## 1. Technical Question & Context
**Question:** How to robustly handle "unknown terms" (out-of-domain vocabulary) in the AI Agent to prevent generic or hallucinated answers? Specifically, mapping user terms (e.g., "event") to domain terms (e.g., "task") or triggering clarification.

**Context:**
- **Project:** Spotplan (Brownfield)
- **Stack:** Python, LangGraph, Weaviate (Hybrid), Cohere (Re-rank)
- **Issue:** "Event" -> Irrelevant Docs -> Hallucination.
- **Goal:** Detect unknown terms, map to known terms, or clarify.

## 2. Requirements

### Functional Requirements
1.  **Unknown Term Detection:** The system must identify when a user's query contains terms that are likely "out-of-domain" or "unknown" relative to the knowledge base.
2.  **Semantic Mapping:** The system should attempt to map unknown terms to known domain terms (e.g., "event" -> "task") using semantic similarity.
3.  **Clarification Loop:** If no clear mapping is found, the system must ask the user for clarification instead of generating a generic answer.
4.  **Fallback Mechanism:** A robust fallback response when retrieval confidence is low.

### Non-Functional Requirements
1.  **Latency:** Solution should add minimal latency (target < 1s overhead).
2.  **Accuracy:** High precision in detecting unknown terms to avoid false positives (annoying clarifications).
3.  **Maintainability:** No manual glossary maintenance; solution should be dynamic or automated.

### Constraints
1.  **Tech Stack:** Must integrate with Python, LangGraph, and Weaviate.
2.  **Data:** No pre-existing glossary of known/unknown terms.

## 3. Research & Evaluation of Options

### Option A: Domain-Aware Query Rewriting (Recommended)
**Concept:** Enhance the existing `rewrite_query` node to act as a "Domain Translator".
**Mechanism:**
1.  Inject a system prompt into the Rewriter: "You are an expert in Spotplan. Users often use generic terms. Map 'event' to 'task', 'post' to 'article', etc."
2.  When the grader rejects documents (because "event" wasn't found), the Rewriter sees the history and explicitly translates the term.
**Pros:**
-   Leverages existing workflow loop.
-   Fixes the root cause (vocabulary mismatch).
-   No new nodes required, just prompt engineering.
**Cons:**
-   Requires knowing the mappings (we can learn them or hardcode common ones).

### Option B: Low-Confidence Fallback to Clarification
**Concept:** Modify the workflow to ask for help if retrieval fails repeatedly.
**Mechanism:**
1.  In `decide_to_generate`, if `retry_count >= 3` AND `grade == "irrelevant"`, route to a new `clarification` node instead of `generate_answer`.
2.  `clarification` node asks: "I'm having trouble finding info on 'events'. Did you mean 'tasks' or something else?"
**Pros:**
-   Prevents hallucinations completely.
-   Engages user to fix the ambiguity.
**Cons:**
-   Requires a new node (`clarification`).
-   Can be frustrating if the system *should* know that event=task.

### Option C: Vector-Based Term Validation (Advanced)
**Concept:** Check query keywords against a "Domain Vocabulary" vector index.
**Mechanism:**
1.  Extract nouns from query.
2.  Search against a `Vocabulary` collection in Weaviate.
3.  If distance > threshold, flag as unknown.
**Pros:**
-   Very precise.
**Cons:**
-   High complexity (maintain vocabulary index).
-   Overkill for this stage.

### Recommendation
**Combine Option A and Option B.**
1.  **First Defense:** Improve `rewrite_query` to handle domain mapping (fixes "event" -> "task").
2.  **Second Defense:** If that fails (3 retries), do NOT generate a hallucination. Route to a `clarification` node.

## 4. Technical Design

### 4.1. Workflow Updates
We will modify the `StateGraph` in `app/graph/workflow.py`.

**Current Flow:**
`detect_language` -> `retrieve_documents` -> `grade_documents` -> (conditional) -> `generate_answer` OR `rewrite_query`

**New Flow:**
1.  **Rewriter Upgrade:** `rewrite_query` will now include domain mapping logic.
2.  **New Node:** `clarify_ambiguity` node added.
3.  **Conditional Logic:**
    -   If `grade == "relevant"` -> `generate_answer`
    -   If `grade == "irrelevant"` AND `retry_count < 3` -> `rewrite_query`
    -   If `grade == "irrelevant"` AND `retry_count >= 3` -> `clarify_ambiguity` (NEW PATH)

### 4.2. Component Details

#### A. `rewrite_query` (Enhanced)
-   **Input:** `state["query"]`, `state["documents"]` (irrelevant ones)
-   **Logic:**
    -   Analyze why documents were irrelevant.
    -   Check for potential vocabulary mismatches (e.g., "User asked for 'event', but I only see 'tasks' in the index").
    -   **Prompt Addition:** "If the user uses terms that don't exist in the domain (e.g., 'event', 'post'), rewrite the query using the correct domain terms (e.g., 'task', 'article')."

#### B. `clarify_ambiguity` (New Node)
-   **Input:** `state["query"]`
-   **Logic:**
    -   Generate a polite response acknowledging the difficulty.
    -   Ask the user to clarify specific terms.
    -   **Prompt:** "I searched for information about '{query}' but couldn't find anything relevant. Could you clarify what you mean by that term? Did you mean [closest_match]?"

### 4.3. Implementation Steps
1.  **Modify `app/graph/nodes.py`**:
    -   Update `rewrite_query` method.
    -   Add `clarify_ambiguity` method.
2.  **Modify `app/graph/workflow.py`**:
    -   Register `clarify_ambiguity` node.
    -   Update `decide_to_generate` conditional logic.
    -   Add edge from `clarify_ambiguity` to `END`.

### 4.4. Verification Plan
1.  **Test Case 1 (Mapping):** User asks "How to edit an event?". System rewrites to "How to edit a task?" -> Finds docs -> Answers correctly.
2.  **Test Case 2 (Clarification):** User asks "How to fly a spaceship?". System retries 3 times -> Routes to Clarification -> "I couldn't find info on spaceships. Can you clarify?"


# Story 2.0: Refactor to LangGraph Architecture

**Status:** In Progress

---

## User Story

As a **System Architect**,
I want **to refactor the linear ChatService into a LangGraph State Machine**,
So that **the agent can perform self-correction, cyclic reasoning, and support the future DSPy optimization loop**.

---

## Acceptance Criteria

**Given** a user query
**When** the agent processes it
**Then** it should execute a graph workflow: `Retrieve -> Grade -> Generate`

**And** if the "Grade" step fails (retrieved docs are irrelevant), it should loop back to `Rewrite Query`
**And** the state must be managed using a typed `AgentState` object
**And** the final response must match or exceed the quality of the current linear implementation.

---

## Implementation Details

### Tasks / Subtasks

- [x] (AC: #1) Add `langgraph` to `requirements.txt`.
- [x] (AC: #2) Create `app/core/state.py` to define `AgentState` (TypedDict).
- [x] (AC: #3) Create `app/graph/nodes.py` containing:
    - `retrieve_node`: Calls `VectorStoreService`.
    - `grade_documents_node`: LLM check for relevance.
    - `generate_node`: Calls GPT-4o for final answer.
    - `rewrite_query_node`: Reformulates query if grading fails.
- [x] (AC: #3) Create `app/graph/workflow.py` to define the graph edges and conditional logic.
- [x] (AC: #4) Update `ChatService` to execute the compiled graph instead of the linear logic.

### Technical Summary

This refactor replaces the `_handle_text_request` method in `ChatService` with a compiled LangGraph application. This is the structural prerequisite for the "Self-Improving" capabilities.
The graph now includes a self-correction loop: `Retrieve -> Grade -> (if irrelevant) Rewrite -> Retrieve`.

### Project Structure Notes

- **New Files:**
    - `app/core/state.py`
    - `app/graph/__init__.py`
    - `app/graph/nodes.py`
    - `app/graph/workflow.py`
- **Files to modify:**
    - `requirements.txt`
    - `app/services/chat_service.py`
- **Estimated effort:** 5 story points (2-3 days)

### Key Code References

- `app/services/chat_service.py` (Logic to be migrated)
- `app/services/vector_store.py` (Tool used by nodes)

---

## Context References

**Research Report:** [AI Agent Development Research Report (1).md](../AI Agent Development Research Report (1).md) - Section 3 "Agentic Architecture"

---

# Story 4.1: Integrate DSPy Module into LangGraph

**Status:** Draft

---

## User Story

As a **System Architect**,
I want **the LangGraph `generate_answer` node to use the compiled DSPy module**,
So that **the agent uses the optimized prompts learned from feedback instead of static hardcoded prompts**.

---

## Acceptance Criteria

**Given** a compiled DSPy program exists at `app/optimization/compiled_rag.json`
**When** the `generate_answer` node is executed
**Then** it should load and invoke the DSPy module
**And** the response should be generated using the optimized prompt.

**Given** no compiled program exists
**When** the `generate_answer` node is executed
**Then** it should fall back to the uncompiled DSPy module (or the previous logic).

---

## Implementation Details

### Tasks / Subtasks

- [ ] (AC: #1) Update `app/core/config.py` to configure the global DSPy LM (using `dspy.LM` or `dspy.OpenAI`).
- [ ] (AC: #2) Modify `app/graph/nodes.py`:
    - Initialize `RAGModule`.
    - Attempt to load `app/optimization/compiled_rag.json`.
    - Replace the `client.chat.completions.create` call with `self.rag_module(question=..., context=...)`.
- [ ] (AC: #3) Verify with a test script.

### Technical Summary

This is the integration point where the "Brain" (LangGraph) gets upgraded with "Skills" (DSPy).
We need to ensure that `dspy.settings` are configured correctly so the loaded module has access to the LLM.

### Key Code References

- `app/graph/nodes.py`
- `app/optimization/dspy_module.py`

---

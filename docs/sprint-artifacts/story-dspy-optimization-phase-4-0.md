# Story 4.0: DSPy Optimization Pipeline

**Status:** Done

---

## User Story

As a **AI Engineer**,
I want **to implement a DSPy optimization pipeline**,
So that **the agent can automatically learn from the collected feedback and improve its prompts over time**.

---

## Acceptance Criteria

**Given** a dataset of "Golden Q&A" pairs (derived from positive feedback)
**When** the optimization pipeline runs
**Then** it should use DSPy to optimize the `Generate` signature/prompt
**And** the optimized prompt should achieve a higher score on the validation set than the baseline.

---

## Implementation Details

### Tasks / Subtasks

- [x] (AC: #1) Add `dspy-ai` to `requirements.txt`.
- [x] (AC: #2) Create `app/optimization/dspy_module.py` defining the DSPy Signature and Module for the RAG task.
- [x] (AC: #3) Create `app/optimization/optimizer.py` to run the `BootstrapFewShot` or `MIPRO` optimizer.
- [x] (AC: #4) Create a script `train_agent.py` that:
    1. Loads "Golden" examples from the `Feedback` database (or a JSON file for now).
    2. Runs the optimizer.
    3. Saves the compiled DSPy program.

### Technical Summary

This story introduces the "Learning" capability. We will wrap the core RAG logic (specifically the generation step) in a DSPy Module.
Initially, we will use a static dataset for training to prove the concept.

### Project Structure Notes

- **New Directory:** `app/optimization/`
- **New Files:**
    - `app/optimization/__init__.py`
    - `app/optimization/dspy_module.py`
    - `app/optimization/optimizer.py`
    - `train_agent.py`

### Key Code References

- `app/graph/nodes.py` (Will eventually use the compiled DSPy module)

---

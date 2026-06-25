"""Layer 1 — Instruction File package.

Exposes the single entry point for composing MarieClaire's system prompts.
"""
from app.instructions.builder import Mode, get_system_prompt

__all__ = ["Mode", "get_system_prompt"]

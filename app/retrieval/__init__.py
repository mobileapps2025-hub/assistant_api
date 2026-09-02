"""Layer 5 — knowledge-base retrieval package."""
from app.retrieval.answerer import answer, build_context_sections, render_markers
from app.retrieval.contextualizer import build_vision_query, contextualize
from app.retrieval.pipeline import run
from app.retrieval.retriever import retrieve

__all__ = [
    "answer", "build_context_sections", "build_vision_query", "contextualize",
    "render_markers", "retrieve", "run",
]

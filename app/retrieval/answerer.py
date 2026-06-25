"""Layer 5 — answer generation from Ragie chunks, grounded via the rag.md instruction.

Image chunks (hi_res) are exposed to the model as visual aids whose markdown points at our
own image-proxy endpoint, so the answer can embed real screenshots without leaking the Ragie
key or hitting cross-origin issues.
"""
from typing import Any, Dict, List, Optional

from app.core.config import API_PUBLIC_URL, client
from app.core.logging import get_logger
from app.enforcement import enforce_answer
from app.instructions import get_system_prompt

logger = get_logger(__name__)

ANSWER_MODEL = "gpt-4o"

_NO_CONTEXT_ANSWER = (
    "I could not find information about that in the current MCL guides. I can help with "
    "Checklists, Tasks, Inspections, the Dashboard, Sync, and Roles & Permissions — could "
    "you rephrase or add a little detail?"
)


def _image_proxy_url(chunk: Any) -> Optional[str]:
    links = getattr(chunk, "links", None) or {}
    if isinstance(links, dict) and links.get("self_image"):
        return (
            f"{API_PUBLIC_URL}/api/ragie/image"
            f"?document_id={chunk.document_id}&chunk_id={chunk.id}"
        )
    return None


def _textual_context(chunks: List[Any]) -> str:
    return "\n".join(
        f"[Source: {getattr(c, 'document_name', 'unknown')}]: {c.text}" for c in chunks
    )


def _visual_aids(chunks: List[Any]) -> str:
    lines = []
    for chunk in chunks:
        url = _image_proxy_url(chunk)
        if url:
            source = getattr(chunk, "document_name", "screenshot")
            lines.append(f"![Screenshot from {source}]({url})")
    return "\n".join(lines)


def _build_user_prompt(query: str, chunks: List[Any], history_text: str) -> str:
    prompt = f"# TEXTUAL CONTEXT\n{_textual_context(chunks)}\n"
    visual = _visual_aids(chunks)
    if visual:
        prompt += f"\n# AVAILABLE VISUAL AIDS\n{visual}\n"
    if history_text:
        prompt += f"\n{history_text}\n"
    prompt += (
        f"\nUser Question: {query}\n\n"
        "Answer as MarieClaire (cite sources inline with [Source: filename]):"
    )
    return prompt


def answer(query: str, chunks: List[Any], *, language: Optional[str] = None,
           history_text: str = "", memory: Optional[str] = None) -> Dict[str, Any]:
    if not chunks:
        return {"answer": _NO_CONTEXT_ANSWER, "sources": []}

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": get_system_prompt("rag", language=language, memory=memory)},
            {"role": "user", "content": _build_user_prompt(query, chunks, history_text)},
        ],
        temperature=0,
        timeout=60,
    )
    content = response.choices[0].message.content.strip()
    allowed_sources = {getattr(c, "document_name", "") for c in chunks}
    allowed_image_urls = {url for url in (_image_proxy_url(c) for c in chunks) if url}
    content = enforce_answer(content, allowed_sources=allowed_sources, allowed_image_urls=allowed_image_urls)

    sources = sorted({getattr(c, "document_name", "unknown") for c in chunks})
    return {"answer": content, "sources": sources}

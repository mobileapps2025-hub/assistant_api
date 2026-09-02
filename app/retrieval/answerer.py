"""Layer 5 — answer generation from KB units, grounded via the rag.md instruction.

The model never writes an image URL. It writes markers — {{step:<procedure>.<n>}} after a
step it reproduced, or {{image:<id>}} for a standalone screenshot — and the renderer here
replaces them deterministically with the approved screenshot(s), or strips them if unknown.
"""
import re
from typing import Any, Dict, List, Optional

from app.core.config import API_PUBLIC_URL, client
from app.core.logging import get_logger
from app.enforcement import enforce_answer
from app.instructions import get_system_prompt

logger = get_logger(__name__)

ANSWER_MODEL = "gpt-4o"
STEP_MARKER = re.compile(r"\{\{\s*step\s*:\s*([^}.\s]+)\.(\d+)\s*\}\}")
IMAGE_MARKER = re.compile(r"\{\{\s*image\s*:\s*([^}\s]+)\s*\}\}")


def _image_markdown(image: Dict[str, Any]) -> str:
    return f"\n\n![{image.get('alt', 'Screenshot')}]({API_PUBLIC_URL}/images/{image['path']})\n"


def _textual_context(units: List[Any]) -> str:
    return "\n".join(f"[Source: {u.document_name}]: {u.text}" for u in units if u.kind == "text")


def _procedures_context(units: List[Any]) -> str:
    blocks = []
    for u in units:
        if u.kind != "procedure":
            continue
        lines = [f"[Source: {u.document_name}] Procedure `{u.id}` — {u.title}"]
        for n, step in enumerate(u.steps, start=1):
            marker = f"  {{{{step:{u.id}.{n}}}}}" if step.get("images") else ""
            lines.append(f"  {n}. {step['text']}{marker}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _images_context(units: List[Any]) -> str:
    lines = [f"- {{{{image:{u.images[0]['id']}}}}} — {u.text}" for u in units if u.kind == "image" and u.images]
    return "\n".join(lines)


def build_context_sections(units: List[Any]) -> str:
    """Render retrieved units for the model: prose, procedures carrying step markers, and
    standalone screenshots. The single place screenshot markers are offered to a model."""
    sections = ""
    if text := _textual_context(units):
        sections += f"# TEXTUAL CONTEXT\n{text}\n"
    if procedures := _procedures_context(units):
        sections += f"\n# PROCEDURES (step-by-step, with screenshot markers)\n{procedures}\n"
    if images := _images_context(units):
        sections += f"\n# STANDALONE SCREENSHOTS\n{images}\n"
    return sections


def _build_user_prompt(query: str, units: List[Any], history_text: str) -> str:
    prompt = build_context_sections(units) or "# TEXTUAL CONTEXT\n(nothing retrieved)\n"
    if history_text:
        prompt += f"\n{history_text}\n"
    prompt += (
        f"\nUser Question: {query}\n\n"
        "Answer as MarieClaire (cite sources inline with [Source: filename]):"
    )
    return prompt


def _step_images(units: List[Any]) -> Dict[tuple, List[Dict[str, Any]]]:
    lookup = {}
    for u in units:
        if u.kind == "procedure":
            for n, step in enumerate(u.steps, start=1):
                lookup[(u.id, n)] = step.get("images", [])
    return lookup


def _standalone_images(units: List[Any]) -> Dict[str, Dict[str, Any]]:
    return {img["id"]: img for u in units for img in u.images}


def render_markers(content: str, units: List[Any]) -> tuple[str, List[str]]:
    """Replace step/image markers with approved screenshots; strip anything unknown."""
    steps, images, urls = _step_images(units), _standalone_images(units), []

    def step_sub(match):
        found = steps.get((match.group(1), int(match.group(2))), [])
        urls.extend(f"{API_PUBLIC_URL}/images/{i['path']}" for i in found)
        return "".join(_image_markdown(i) for i in found)

    def image_sub(match):
        image = images.get(match.group(1))
        if not image:
            return ""
        urls.append(f"{API_PUBLIC_URL}/images/{image['path']}")
        return _image_markdown(image)

    content = STEP_MARKER.sub(step_sub, content)
    content = IMAGE_MARKER.sub(image_sub, content)
    return content, urls


def answer(query: str, units: List[Any], *, language: Optional[str] = None,
           device: Optional[str] = None, history_text: str = "", memory: Optional[str] = None) -> Dict[str, Any]:
    # Empty context still goes through the model so rag.md produces a refusal in the
    # user's language, rather than a hardcoded English string.
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": get_system_prompt("rag", language=language, device=device, memory=memory)},
            {"role": "user", "content": _build_user_prompt(query, units, history_text)},
        ],
        temperature=0,
        timeout=60,
    )
    content = response.choices[0].message.content.strip()
    content, image_urls = render_markers(content, units)
    allowed_sources = {u.document_name for u in units}
    content = enforce_answer(content, allowed_sources=allowed_sources, allowed_image_urls=set(image_urls))

    sources = sorted({u.document_name for u in units})
    return {"answer": content, "sources": sources}

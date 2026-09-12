"""The new web knowledge: real screens of the rebuilt web platform, walked from the live app.

Each screen is one JSON file under app/documents/new_app_map/. This turns those into KB units
tagged surface='web' — the only source a web how-to question is allowed to use. Shapes match
the rest of the index: a text overview, one image unit per screenshot (offered to the model as
a {{image:...}} marker), and procedure units whose steps carry their screenshot.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from app.kb_capabilities import EVERYONE, capability_from_requires
from app.kb_surfaces import WEB

MAP_DIR = Path(__file__).resolve().parent / "documents" / "new_app_map"
IMAGE_URL_PREFIX = "kb/webmap"


def _image_ref(screenshot: Dict[str, Any]) -> Dict[str, str]:
    return {"id": screenshot["id"], "path": f"{IMAGE_URL_PREFIX}/{screenshot['file']}",
            "alt": screenshot.get("alt", "Screenshot")}


def _screen_units(screen: Dict[str, Any]) -> List[Dict[str, Any]]:
    document_name = screen["document_name"]
    refs = {s["id"]: _image_ref(s) for s in screen.get("screenshots", [])}
    aliases = ", ".join(screen.get("aliases", []))
    # The overview stays visible to everyone: it names what the screen is and, for a restricted
    # screen, who may use it — so a user without access is pointed to their administrator instead
    # of getting silence. The steps and screenshots carry the real permission, so only a user who
    # has it gets the how-to.
    capability = capability_from_requires(screen.get("requires"), screen.get("capability"))

    overview_text = (
        f"Screen: {screen['screen']} (web platform). Menu: {screen['menu_label']}.\n"
        f"{screen['overview']}"
    )
    if screen.get("requires"):
        overview_text += (
            f"\nWho can do this: {screen['requires']}. If your role does not have this, you will not "
            "see the menu option; ask your company administrator to do it or to grant access.")
    if aliases:
        overview_text += f"\nAlso asked about as: {aliases}."

    units: List[Dict[str, Any]] = [{
        "kind": "text", "id": f"web_{_slug(screen['screen'])}_overview",
        "document_name": document_name, "text": overview_text, "images": [], "steps": [],
        "surface": WEB, "capability": EVERYONE,
    }]

    for shot in screen.get("screenshots", []):
        units.append({
            "kind": "image", "id": shot["id"], "document_name": document_name,
            "text": shot.get("description") or shot.get("alt", ""),
            "images": [refs[shot["id"]]], "steps": [], "surface": WEB, "capability": capability,
        })

    for proc in screen.get("procedures", []):
        steps = [{"text": s["text"], "images": [refs[s["image"]]] if s.get("image") in refs else []}
                 for s in proc["steps"]]
        text = f"Procedure: {proc['title']}\n" + "\n".join(f"{n}. {s['text']}" for n, s in enumerate(steps, 1))
        units.append({
            "kind": "procedure", "id": proc["id"], "document_name": document_name,
            "text": text, "title": proc["title"], "steps": steps,
            "images": [img for s in steps for img in s["images"]], "surface": WEB, "capability": capability,
        })

    return units


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def build_web_map_units() -> List[Dict[str, Any]]:
    if not MAP_DIR.exists():
        return []
    units: List[Dict[str, Any]] = []
    for path in sorted(MAP_DIR.glob("*.json")):
        units.extend(_screen_units(json.loads(path.read_text(encoding="utf-8"))))
    return units

"""Stage 2 — the new web map: real web screens as searchable web units, with a relevance floor."""
import numpy as np

import app.retrieval.retriever as r
from app.kb_capabilities import EVERYONE
from app.kb_surfaces import WEB
from app.web_map import build_web_map_units


def test_web_map_units_are_all_tagged_web():
    units = build_web_map_units()
    assert units, "no web-map screens found"
    assert all(u["surface"] == WEB for u in units)


def test_web_map_has_the_three_seed_screens():
    docs = {u["document_name"] for u in build_web_map_units()}
    assert "MCL Web Guide - Tasks" in docs
    assert "MCL Web Guide - Overview" in docs
    assert "MCL Web Guide - Photo management" in docs


def test_web_map_screens_carry_screenshots_with_servable_paths():
    images = [img for u in build_web_map_units() for img in u["images"]]
    assert images, "no screenshots attached"
    assert all(img["path"].startswith("kb/webmap/") for img in images)


def test_required_permission_is_stated_in_the_overview():
    from app.web_map import _screen_units
    overview = next(u for u in _screen_units({
        "screen": "Markets", "menu_label": "Markets", "route": "/x",
        "document_name": "d", "overview": "The markets page.", "requires": "a Company administrator",
    }) if u["kind"] == "text")
    assert "Who can do this: a Company administrator" in overview["text"]
    assert "ask your company administrator" in overview["text"].lower()

    plain = next(u for u in _screen_units({
        "screen": "Tasks", "menu_label": "Tasks", "route": "/t",
        "document_name": "d", "overview": "The tasks page.",
    }) if u["kind"] == "text")
    assert "Who can do this" not in plain["text"]


def test_restricted_screen_keeps_overview_open_but_steps_behind_the_capability():
    from app.web_map import _screen_units
    units = _screen_units({
        "screen": "Company checklists", "menu_label": "Checklists", "route": "/x",
        "document_name": "d", "overview": "The checklists page.",
        "requires": "a Checklist editor role (Company administrator, ...)",
        "screenshots": [{"id": "web-cl", "file": "web-cl.jpg", "alt": "a", "description": "b"}],
        "procedures": [{"id": "web_make_checklist", "title": "Create a checklist",
                        "steps": [{"text": "Do it.", "image": "web-cl"}]}],
    })
    overview = next(u for u in units if u["kind"] == "text")
    proc = next(u for u in units if u["kind"] == "procedure")
    image = next(u for u in units if u["kind"] == "image")
    assert overview["capability"] == EVERYONE
    assert proc["capability"] == "checklists.edit"
    assert image["capability"] == "checklists.edit"


def test_open_screen_tags_everything_everyone():
    from app.web_map import _screen_units
    units = _screen_units({
        "screen": "Tasks", "menu_label": "Tasks", "route": "/t", "document_name": "d",
        "overview": "The tasks page.",
        "procedures": [{"id": "web_new_task", "title": "New task", "steps": [{"text": "Go."}]}],
    })
    assert all(u["capability"] == EVERYONE for u in units)


def test_retrieve_hides_units_the_user_lacks_the_capability_for(monkeypatch):
    units = [
        r.Unit(kind="text", id="overview", document_name="d", text="x", surface=WEB, capability=EVERYONE),
        r.Unit(kind="procedure", id="steps", document_name="d", text="y", surface=WEB, capability="checklists.edit"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
    monkeypatch.setattr(r, "_load_index", lambda: (units, embeddings))
    monkeypatch.setattr(r, "_embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))

    editor = r.retrieve("q", surfaces=frozenset({WEB}), capabilities=frozenset({"everyone", "checklists.edit"}))
    assert {u.id for u in editor} == {"overview", "steps"}

    plain = r.retrieve("q", surfaces=frozenset({WEB}), capabilities=frozenset({"everyone"}))
    assert {u.id for u in plain} == {"overview"}

    legacy = r.retrieve("q", surfaces=frozenset({WEB}), capabilities=frozenset())
    assert {u.id for u in legacy} == {"overview"}


def test_create_task_procedure_exists_with_its_dialog_image():
    proc = next(u for u in build_web_map_units() if u["id"] == "web_create_task")
    assert proc["kind"] == "procedure"
    assert any(img["id"] == "web-tasks-new" for img in proc["images"])


def test_floor_gates_on_the_best_match_then_returns_the_page(monkeypatch):
    """The floor decides whether a relevant page exists; once the best match clears it, the page's
    own weaker units (its steps) still come along, and when nothing clears it, nothing returns."""
    units = [
        r.Unit(kind="text", id="strong", document_name="d", text="x", surface=WEB),
        r.Unit(kind="procedure", id="weak", document_name="d", text="y", surface=WEB),
    ]
    embeddings = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    monkeypatch.setattr(r, "_load_index", lambda: (units, embeddings))
    monkeypatch.setattr(r, "_embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))

    # best match (1.0) clears the bar -> both units returned, including the below-bar step (0.6)
    kept = r.retrieve("q", surfaces=frozenset({WEB}), min_score=0.8)
    assert {u.id for u in kept} == {"strong", "weak"}

    # nothing clears the bar -> nothing returned (honest "I don't have it")
    assert r.retrieve("q", surfaces=frozenset({WEB}), min_score=1.01) == []

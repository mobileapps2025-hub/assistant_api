"""Stage 2 — the new web map: real web screens as searchable web units, with a relevance floor."""
import numpy as np

import app.retrieval.retriever as r
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


def test_create_task_procedure_exists_with_its_dialog_image():
    proc = next(u for u in build_web_map_units() if u["id"] == "web_create_task")
    assert proc["kind"] == "procedure"
    assert any(img["id"] == "web-tasks-new" for img in proc["images"])


def test_relevance_floor_drops_weak_matches(monkeypatch):
    units = [
        r.Unit(kind="text", id="strong", document_name="d", text="x", surface=WEB),
        r.Unit(kind="text", id="weak", document_name="d", text="y", surface=WEB),
    ]
    embeddings = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    monkeypatch.setattr(r, "_load_index", lambda: (units, embeddings))
    monkeypatch.setattr(r, "_embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))

    hits = r.retrieve("q", surfaces=frozenset({WEB}), min_score=0.8)

    assert [u.id for u in hits] == ["strong"]   # 1.0 kept, 0.6 dropped

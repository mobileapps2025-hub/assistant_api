"""Stage 0 — knowledge split by surface.

Old web documents must never be searched (they describe the rebuilt web app and would give
confident wrong answers). App and shared documents stay live. The index that ships must carry
a surface on every unit.
"""
import json
from pathlib import Path

from app.kb_surfaces import (APP, BOTH, SEARCHABLE_BY_DEFAULT, WEB, WEB_LEGACY,
                             WEB_SEARCH, classify)

INDEX = Path(__file__).resolve().parent.parent / "app" / "kb_index" / "index.json"


def test_old_web_documents_are_legacy():
    assert classify("MCL Visual Guide Tasks.pdf") == WEB_LEGACY
    assert classify("Mcl Dashboard Faq Revised.pdf") == WEB_LEGACY
    assert classify("Creating Checklists EN v5 05.22.2020_compressed.pdf") == WEB_LEGACY
    assert classify("Task Creation Permissions in MCL") == WEB_LEGACY


def test_mobile_app_documents_stay_app():
    assert classify("Visual Guide Tasks MCL APP.pdf") == APP
    assert classify("Mcl Mobile App Faq Revised.pdf") == APP
    assert classify("Synchronization Overview") == APP


def test_shared_documents_are_both():
    assert classify("Visual Guide Notifications APP & WEB.pdf") == BOTH
    assert classify("MCL_User_Guide_QA_EN.pdf") == BOTH


def test_legacy_web_is_not_searchable_but_the_new_map_is():
    assert WEB_LEGACY not in SEARCHABLE_BY_DEFAULT
    assert WEB_LEGACY not in WEB_SEARCH
    assert WEB in WEB_SEARCH
    assert APP in SEARCHABLE_BY_DEFAULT and BOTH in SEARCHABLE_BY_DEFAULT


def test_shipped_index_has_a_surface_on_every_unit():
    units = json.loads(INDEX.read_text(encoding="utf-8"))["units"]
    assert units, "index is empty"
    assert all(u.get("surface") for u in units)
    assert not any(u["surface"] in SEARCHABLE_BY_DEFAULT
                   for u in units if u["surface"] == WEB_LEGACY)


def test_a_web_screen_question_finds_nothing_in_the_default_view(monkeypatch):
    import numpy as np
    import app.retrieval.retriever as r

    units = [
        r.Unit(kind="text", id="w1", document_name="Mcl Dashboard Faq Revised.pdf",
               text="Open the Checklist Wizard from the Dashboard.", surface=WEB_LEGACY),
        r.Unit(kind="text", id="a1", document_name="Mcl Mobile App Faq Revised.pdf",
               text="Sync the mobile app from the menu.", surface=APP),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(r, "_load_index", lambda: (units, embeddings))
    monkeypatch.setattr(r, "_embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))

    hits = r.retrieve("how do I create a checklist")

    assert all(u.surface != WEB_LEGACY for u in hits)
    assert "w1" not in [u.id for u in hits]

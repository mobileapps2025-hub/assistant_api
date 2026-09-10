"""Stage 1 — decide web vs app before searching, and route to the matching knowledge."""
import app.routing.surface as surface
from app.kb_surfaces import APP, APP_SEARCH, WEB, WEB_SEARCH
from app.models import AuthContext, Device, PlatformTurn


def _platform_turn() -> PlatformTurn:
    return PlatformTurn(id="t1", token="g", operations_url="https://localhost/ops")


def test_platform_caller_defaults_to_web():
    actor = AuthContext(platform_turn=_platform_turn())
    assert surface.base_surface(actor, None) == WEB


def test_mobile_device_defaults_to_app():
    assert surface.base_surface(None, Device(platform="iOS", form_factor="phone")) == APP
    assert surface.base_surface(None, Device(platform="Android")) == APP


def test_web_device_defaults_to_web():
    assert surface.base_surface(None, Device(platform="Web", form_factor="desktop")) == WEB


def test_unknown_caller_defaults_to_app():
    assert surface.base_surface(None, None) == APP


def test_search_surfaces_map_to_the_right_index_view():
    assert surface.search_surfaces(WEB) == WEB_SEARCH
    assert surface.search_surfaces(APP) == APP_SEARCH


def test_empty_query_keeps_the_caller_surface_without_a_model_call():
    assert surface.classify_surface("", [], WEB) == WEB
    assert surface.classify_surface("   ", [], APP) == APP


def test_classifier_keeps_base_on_model_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no model")
    monkeypatch.setattr(surface.client.chat.completions, "create", boom)
    assert surface.classify_surface("how do I create a checklist", [], WEB) == WEB


def _fake_model(payload):
    class _Msg:
        content = payload

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    return lambda *a, **k: _Resp()


def test_classifier_can_switch_to_the_other_surface(monkeypatch):
    monkeypatch.setattr(surface.client.chat.completions, "create", _fake_model('{"surface": "app"}'))
    assert surface.classify_surface("how do I do this on my phone", [], WEB) == APP


def test_a_web_question_routed_to_web_finds_no_app_content():
    import numpy as np
    import app.retrieval.retriever as r

    units = [
        r.Unit(kind="text", id="a1", document_name="Mcl Mobile App Faq Revised.pdf",
               text="Open the MCL app and tap Tasks.", surface=APP),
    ]
    r._index = (units, np.array([[1.0, 0.0]], dtype=np.float32))
    old_embed = r._embed_query
    r._embed_query = lambda q: np.array([1.0, 0.0], dtype=np.float32)
    try:
        assert r.retrieve("how do I create a task", surfaces=WEB_SEARCH) == []
        assert len(r.retrieve("how do I create a task", surfaces=APP_SEARCH)) == 1
    finally:
        r._embed_query = old_embed
        r._index = None

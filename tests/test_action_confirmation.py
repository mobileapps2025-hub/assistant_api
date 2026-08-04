"""Layer 4 — danger-review / ACTION confirmation flow."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import tools
from app.enforcement import check_tool_call
from app.enforcement import pending
from app.services.chat_service import ChatService


def _auth(user_id="u1"):
    return SimpleNamespace(access_token="t", user_id=user_id, company_id="c", email="e@x.com")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _clear_pending():
    pending._PENDING.clear()
    yield
    pending._PENDING.clear()


def test_delete_task_is_destructive_and_executable():
    spec = tools.get_spec("delete_task")
    assert spec.risk == "destructive"
    assert check_tool_call("delete_task").allowed is True  # allowed to run — but only after confirm


def _tool_response(name, args_json):
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = name
    tc.function.arguments = args_json
    resp = MagicMock()
    resp.choices[0].message.tool_calls = [tc]
    resp.choices[0].message.content = None
    return resp


def _text_response(text):
    resp = MagicMock()
    resp.choices[0].message.tool_calls = None
    resp.choices[0].message.content = text
    return resp


def test_fc_asks_for_missing_detail_instead_of_assuming():
    """A vague create request: the model asks for the name (no tool call) and that question is
    returned to the user — it must not fall through to RAG or invent a task."""
    svc = ChatService(None, None)
    llm = MagicMock()
    llm.chat.completions.create.side_effect = [_text_response("Sure — what should the task be called?")]
    with patch("app.services.chat_service.client", llm):
        out = _run(svc._handle_function_calling(
            [{"role": "user", "content": "create a task"}],
            {"content": "create a task"}, _auth()))
    assert out is not None                                   # did NOT fall through to RAG
    assert out["response"] == "Sure — what should the task be called?"
    assert "requires_confirmation" not in out               # nothing staged yet


def test_fc_empty_step0_falls_through_to_rag():
    """No tool call and nothing said on step 0 stays a RAG fall-through (misroute safety net)."""
    svc = ChatService(None, None)
    llm = MagicMock()
    llm.chat.completions.create.side_effect = [_text_response("")]
    with patch("app.services.chat_service.client", llm):
        out = _run(svc._handle_function_calling(
            [{"role": "user", "content": "how do checklists work"}],
            {"content": "how do checklists work"}, _auth()))
    assert out is None


def test_fc_loop_reads_then_pauses_on_write():
    """Referencing a task by name: the loop runs the read tool inline, then chains to the
    write tool and pauses for confirmation (instead of stopping after the read)."""
    pending._PENDING.clear()
    svc = ChatService(None, None)
    fake_client = SimpleNamespace(get_task_todos=AsyncMock(return_value=[{"tdo_id": "9"}]))
    llm = MagicMock()
    llm.chat.completions.create.side_effect = [
        _tool_response("get_task_todos", "{}"),
        _tool_response("add_task_note", '{"todo_id":"9","note":"urgent","confirmation":"Add the note \\"urgent\\" to the task \\"Freeze meat\\"."}'),
    ]
    with patch("app.services.chat_service.client", llm), \
         patch("app.services.chat_service.MCLServiceClient", return_value=fake_client):
        out = _run(svc._handle_function_calling(
            [{"role": "user", "content": "add a note to task Freeze meat"}],
            {"content": "add a note to task Freeze meat"}, _auth()))
    fake_client.get_task_todos.assert_awaited_once()          # the read ran inline
    assert out["requires_confirmation"] is True               # then paused on the write
    assert out["confirmation"]["action_summary"] == 'Add the note "urgent" to the task "Freeze meat".'


def test_confirmation_summary_is_the_model_authored_string():
    # The card text is whatever the model wrote in `confirmation` — in the user's language, no
    # per-tool code, no raw id.
    for name in ("add_task", "edit_task", "add_task_note", "delete_task"):
        summary = tools.get_spec(name).summarize({"todo_id": "f9e05e53-guid", "confirmation": "Lösche die Aufgabe „QA“."})
        assert summary == "Lösche die Aufgabe „QA“."
        assert "f9e05e53" not in summary


def test_confirmation_falls_back_generically_when_model_omits_it():
    # Defensive only (the field is required). Falls back to the tool description, never a crash.
    summary = tools.get_spec("delete_task").summarize({"todo_id": "g"})
    assert summary and "g" != summary  # some human text, not the raw id


def test_write_tools_are_write_risk_and_need_confirmation():
    for name in ("add_task", "edit_task", "add_task_note"):
        spec = tools.get_spec(name)
        assert spec.risk == "write" and spec.exposed and spec.executable


def test_add_task_handler_builds_todo_body_dropping_nulls():
    mcl = SimpleNamespace(add_task=AsyncMock(return_value=None))
    _run(tools._add_task(mcl, _auth(), {"description": "Clean freezer", "due_date": None, "market_id": "m1", "assigned_user_id": None}))
    _, company_id, user_id, todo = mcl.add_task.call_args.args
    assert company_id == "c" and user_id == "u1"
    assert todo == {"tdo_description": "Clean freezer", "mkt_id": "m1",
                    "tty_id": tools.DEFAULT_TASK_TYPE_ID}  # nulls dropped, task type added


def test_pending_store_roundtrip_and_user_scoping():
    cid = pending.create_pending("delete_task", {"todo_id": "9"}, "u1", "Delete task 9", "destructive")
    assert pending.take_pending(cid, "u2") is None          # wrong user
    got = pending.take_pending(cid, "u1")
    assert got["args"] == {"todo_id": "9"}
    assert pending.take_pending(cid, "u1") is None           # one-shot


def test_expired_pending_is_not_returned(monkeypatch):
    cid = pending.create_pending("delete_task", {}, "u1", "s", "destructive")
    monkeypatch.setattr(pending.time, "time", lambda: pending._PENDING[cid]["created_at"] + pending.TTL_SECONDS + 1)
    assert pending.take_pending(cid, "u1") is None


def test_reject_makes_no_change():
    svc = ChatService(None, None)
    cid = pending.create_pending("delete_task", {"todo_id": "9"}, "u1", "Delete task 9", "destructive")
    out = _run(svc.execute_confirmed_action(cid, "reject", _auth()))
    assert "cancelled" in out["response"].lower()
    assert pending.take_pending(cid, "u1") is not None       # still pending, not consumed by reject


def test_approve_runs_the_handler():
    svc = ChatService(None, None)
    cid = pending.create_pending("delete_task", {"todo_id": "9"}, "u1", "Delete task 9", "destructive")
    fake_client = SimpleNamespace(delete_task=AsyncMock(return_value=None))
    with patch("app.services.chat_service.MCLServiceClient", return_value=fake_client):
        out = _run(svc.execute_confirmed_action(cid, "approve", _auth()))
    fake_client.delete_task.assert_awaited_once_with("t", "c", "9")
    assert out["response"].startswith("✓ Done")


def test_approve_with_expired_or_wrong_user_does_nothing():
    svc = ChatService(None, None)
    cid = pending.create_pending("delete_task", {"todo_id": "9"}, "u1", "s", "destructive")
    fake_client = SimpleNamespace(delete_task=AsyncMock())
    with patch("app.services.chat_service.MCLServiceClient", return_value=fake_client):
        out = _run(svc.execute_confirmed_action(cid, "approve", _auth(user_id="OTHER")))
    fake_client.delete_task.assert_not_awaited()
    assert "expired" in out["response"].lower() or "wasn't found" in out["response"].lower()

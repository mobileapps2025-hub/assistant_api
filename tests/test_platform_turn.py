"""M1 — MarieClaire accepts a turn from the trusted caller (MCL.Api) instead of a user token.

The browser never reaches this endpoint. MCL.Api forwards the message with the actor it
derived from the validated session, authenticated by a shared service secret.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core import dependencies
from app.main import app

SECRET = "test-service-secret"
HEADERS = {"Authorization": f"Bearer {SECRET}"}

TURN = {
    "actor": {"userId": "u-1", "companyId": "c-1", "roleIds": ["3"], "language": "es", "platform": "web"},
    "message": "Hola",
    "history": [],
    "context": None,
    "turn": {"id": "t-1", "token": "turn-token", "operationsUrl": "https://mcl.example/api/v1/assistant/operations"},
}


@pytest.fixture
def secret_configured(monkeypatch):
    monkeypatch.setattr("app.platform.auth.ASSISTANT_SERVICE_SECRET", SECRET)


@pytest.fixture
def fake_chat_service():
    service = AsyncMock()
    service.process_chat_request.return_value = {"response": "¡Hola! ¿En qué te ayudo?", "success": True}
    app.dependency_overrides[dependencies.get_chat_service] = lambda: service
    yield service
    app.dependency_overrides.pop(dependencies.get_chat_service, None)


def test_turn_without_secret_is_rejected(secret_configured, fake_chat_service):
    response = TestClient(app).post("/api/platform/turn", json=TURN)
    assert response.status_code == 401
    fake_chat_service.process_chat_request.assert_not_called()


def test_turn_with_wrong_secret_is_rejected(secret_configured, fake_chat_service):
    response = TestClient(app).post("/api/platform/turn", json=TURN, headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_turn_is_closed_when_no_secret_is_configured(monkeypatch, fake_chat_service):
    monkeypatch.setattr("app.platform.auth.ASSISTANT_SERVICE_SECRET", "")
    response = TestClient(app).post("/api/platform/turn", json=TURN, headers=HEADERS)
    assert response.status_code == 503


def test_turn_returns_reply_and_turn_id(secret_configured, fake_chat_service):
    response = TestClient(app).post("/api/platform/turn", json=TURN, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "¡Hola! ¿En qué te ayudo?"
    assert body["turnId"] == "t-1"
    assert body["proposal"] is None


def test_actor_becomes_the_auth_context_and_platform_the_device(secret_configured, fake_chat_service):
    TestClient(app).post("/api/platform/turn", json=TURN, headers=HEADERS)
    kwargs = fake_chat_service.process_chat_request.call_args.kwargs
    auth = kwargs["auth_context"]
    assert auth.user_id == "u-1"
    assert auth.company_id == "c-1"
    assert auth.access_token is None                      # never a user credential
    assert auth.platform_turn.token == "turn-token"
    assert auth.platform_turn.operations_url.startswith("https://mcl.example")
    assert kwargs["device"].platform == "web"


def test_capabilities_from_the_actor_reach_the_auth_context(secret_configured, fake_chat_service):
    turn = {**TURN, "actor": {**TURN["actor"], "capabilities": ["everyone", "checklists.edit"]}}
    TestClient(app).post("/api/platform/turn", json=turn, headers=HEADERS)
    auth = fake_chat_service.process_chat_request.call_args.kwargs["auth_context"]
    assert auth.capabilities == ["everyone", "checklists.edit"]


def test_missing_capabilities_default_to_empty(secret_configured, fake_chat_service):
    TestClient(app).post("/api/platform/turn", json=TURN, headers=HEADERS)
    auth = fake_chat_service.process_chat_request.call_args.kwargs["auth_context"]
    assert auth.capabilities == []


def test_history_and_message_become_the_conversation(secret_configured, fake_chat_service):
    turn = {**TURN, "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
    TestClient(app).post("/api/platform/turn", json=turn, headers=HEADERS)
    messages = fake_chat_service.process_chat_request.call_args.args[0]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == "Hola"


def test_turn_rejects_empty_message(secret_configured, fake_chat_service):
    response = TestClient(app).post("/api/platform/turn", json={**TURN, "message": "  "}, headers=HEADERS)
    assert response.status_code == 422


def _platform_actor():
    from app.models import AuthContext, PlatformTurn
    return AuthContext(user_id="u-1", company_id="c-1",
                       platform_turn=PlatformTurn(id="t-9", token="turn-secret", operations_url="https://mcl.example/ops"))


def _tool_call(name, args_json):
    from unittest.mock import MagicMock
    tc = MagicMock(); tc.id = "call_1"; tc.function.name = name; tc.function.arguments = args_json
    resp = MagicMock(); resp.choices[0].message.tool_calls = [tc]; resp.choices[0].message.content = None
    return resp


def _text(text):
    from unittest.mock import MagicMock
    resp = MagicMock(); resp.choices[0].message.tool_calls = None; resp.choices[0].message.content = text
    return resp


def _run(coro):
    import asyncio
    return asyncio.new_event_loop().run_until_complete(coro)


def test_platform_actor_is_signed_in_and_gets_the_platform_tools_not_the_legacy_ones():
    from app.models import AuthContext
    from app.platform.tools import PLATFORM_TOOLS
    from app.services.chat_service import _is_authenticated, _live_tools_for

    assert _is_authenticated(_platform_actor()) is True
    assert _live_tools_for(_platform_actor()) is PLATFORM_TOOLS
    assert _live_tools_for(AuthContext(access_token="bearer", user_id="u-1", company_id="c-1")) is not PLATFORM_TOOLS


def test_platform_actor_gets_no_plan_tool_and_the_surface_note():
    from unittest.mock import MagicMock, patch
    from app.platform.tools import PLATFORM_SURFACE_NOTE, LIST_TASKS
    from app.services.chat_service import ChatService, ACTION_PLAN_TOOL_NAME

    service = ChatService(MagicMock(), MagicMock())
    messages = [{"role": "user", "content": "hola"}]
    with patch("app.services.chat_service.client") as mock_client:
        mock_client.chat.completions.create.return_value = _text("¡Hola!")
        _run(service._handle_agent_request(messages, messages[0], "s1", _platform_actor(), "", "Spanish", ""))
        kwargs = mock_client.chat.completions.create.call_args.kwargs

    names = [tool["function"]["name"] for tool in kwargs["tools"]]
    assert ACTION_PLAN_TOOL_NAME not in names and LIST_TASKS in names
    assert any(m["role"] == "system" and m["content"] == PLATFORM_SURFACE_NOTE for m in kwargs["messages"])


def test_list_my_tasks_goes_to_mcl_api_with_the_turn_pass():
    from unittest.mock import MagicMock, patch
    import json
    from app.services.chat_service import ChatService

    service = ChatService(MagicMock(), MagicMock())
    messages = [{"role": "user", "content": "what tasks do I have?"}]
    seen = {}

    async def fake_post(self, url, json=None, headers=None):
        seen.update(url=url, body=json, headers=headers)
        resp = MagicMock(); resp.status_code = 200
        resp.json.return_value = {"ok": True, "data": [{"id": "t1", "description": "Check fridge"}]}
        return resp

    with patch("app.services.chat_service.client") as mock_client,          patch("httpx.AsyncClient.post", new=fake_post):
        mock_client.chat.completions.create.side_effect = [
            _tool_call("list_my_tasks", '{"status": null, "dueBefore": null}'),
            _text("You have one task: Check fridge."),
        ]
        result = _run(service._handle_agent_request(messages, messages[0], "s1", _platform_actor(), "", "English", ""))
        fed_back = json.loads(mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]["content"])

    assert seen["url"] == "https://mcl.example/ops"
    assert seen["headers"] == {"Authorization": "Bearer turn-secret"}
    assert seen["body"] == {"operation": "tasks.list", "args": {}}
    assert fed_back["ok"] is True and fed_back["data"][0]["description"] == "Check fridge"
    assert result["response"] == "You have one task: Check fridge."


def test_refused_operation_is_explained_to_the_model_not_invented():
    from unittest.mock import MagicMock, patch
    import json
    from app.services.chat_service import ChatService

    service = ChatService(MagicMock(), MagicMock())
    messages = [{"role": "user", "content": "show task 7"}]

    async def fake_post(self, url, json=None, headers=None):
        resp = MagicMock(); resp.status_code = 403
        resp.json.return_value = {"ok": False, "code": "forbidden"}
        return resp

    with patch("app.services.chat_service.client") as mock_client,          patch("httpx.AsyncClient.post", new=fake_post):
        mock_client.chat.completions.create.side_effect = [
            _tool_call("get_task", '{"id": "7"}'),
            _text("You are not allowed to see that task."),
        ]
        _run(service._handle_agent_request(messages, messages[0], "s1", _platform_actor(), "", "English", ""))
        fed_back = json.loads(mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]["content"])

    assert fed_back == {"ok": False, "code": "forbidden", "instruction": fed_back["instruction"]}
    assert "not allowed" in fed_back["instruction"]


def test_proposing_a_task_returns_a_proposal_and_executes_nothing():
    from unittest.mock import MagicMock, patch
    from app.services.chat_service import ChatService

    service = ChatService(MagicMock(), MagicMock())
    messages = [{"role": "user", "content": "create a task to clean the freezer"}]
    calls = []

    async def fake_post(self, url, json=None, headers=None):
        calls.append(json["operation"])
        resp = MagicMock(); resp.status_code = 200
        resp.json.return_value = {"ok": True, "data": {"tasks": [], "markets": [{"id": "m-only", "name": "Only"}]}}
        return resp

    with patch("app.services.chat_service.client") as mock_client,          patch("httpx.AsyncClient.post", new=fake_post):
        mock_client.chat.completions.create.return_value = _tool_call(
            "propose_task_creation",
            '{"description": "Clean the freezer", "dueDate": null, "note": null, "marketId": null, "summary": "Create the task Clean the freezer with no due date."}')
        result = _run(service._handle_agent_request(messages, messages[0], "s1", _platform_actor(), "", "English", ""))

    assert calls == ["tasks.list"]                      # only a read: nothing is created here
    assert result["proposal"] == {
        "operation": "tasks.create",
        "args": {"description": "Clean the freezer", "dueDate": None, "note": None, "marketId": None},
        "summary": "Create the task Clean the freezer with no due date.",
    }
    assert result["response"] == result["proposal"]["summary"]


def test_router_surfaces_the_proposal(secret_configured, fake_chat_service):
    fake_chat_service.process_chat_request.return_value = {
        "response": "Create the task X.", "success": True,
        "proposal": {"operation": "tasks.create", "args": {"description": "X", "dueDate": None, "note": None}, "summary": "Create the task X."},
    }
    body = TestClient(app).post("/api/platform/turn", json=TURN, headers=HEADERS).json()
    assert body["proposal"]["operation"] == "tasks.create"
    assert body["proposal"]["summary"] == "Create the task X."


def test_tls_is_verified_everywhere_except_loopback():
    from app.platform.tools import verify_tls
    assert verify_tls("https://assistantapi.example.net/api/v1/assistant/operations") is True
    assert verify_tls("https://localhost:7206/api/v1/assistant/operations") is False
    assert verify_tls("https://127.0.0.1:7206/ops") is False


def test_market_written_as_name_or_position_becomes_the_real_id():
    from app.platform.tools import resolve_market, build_proposal
    markets = [{"id": "m-aaa", "name": "Elli-Markt Surenheide"}, {"id": "m-bbb", "name": "Elli-Markt Stromberg"}]
    assert resolve_market("m-bbb", markets) == "m-bbb"
    assert resolve_market("elli-markt stromberg", markets) == "m-bbb"
    assert resolve_market("2", markets) == "m-bbb"
    assert resolve_market("7", markets) == "7"            # nothing to match: MCL.Api will refuse it honestly
    assert resolve_market(None, markets) is None
    proposal = build_proposal("propose_task_creation", {"description": "x", "summary": "Create x.", "marketId": "1"}, markets)
    assert proposal["args"]["marketId"] == "m-aaa"


def test_proposal_with_an_unknown_market_is_sent_back_to_the_model_not_the_user():
    from app.platform.tools import build_proposal, proposal_problem
    markets = [{"id": "m-aaa", "name": "Nord"}, {"id": "m-bbb", "name": "Sud"}]
    T = "propose_task_creation"
    args = {"description": "x", "summary": "Create x.", "marketId": "7"}
    problem = proposal_problem(T, args, build_proposal(T, args, markets), markets)
    assert problem and "Nord (id m-aaa)" in problem

    args_ok = {"description": "x", "summary": "Create x.", "marketId": "Sud"}
    assert proposal_problem(T, args_ok, build_proposal(T, args_ok, markets), markets) is None

    args_none = {"description": "x", "summary": "Create x.", "marketId": None}
    assert "several markets" in proposal_problem(T, args_none, build_proposal(T, args_none, markets), markets)
    assert proposal_problem(T, args_none, build_proposal(T, args_none, markets[:1]), markets[:1]) is None


def test_proposing_without_a_list_this_turn_fetches_the_markets_first():
    from unittest.mock import MagicMock, patch
    from app.services.chat_service import ChatService

    service = ChatService(MagicMock(), MagicMock())
    messages = [{"role": "user", "content": "create a task in Stromberg"}]
    calls = []

    async def fake_post(self, url, json=None, headers=None):
        calls.append(json["operation"])
        resp = MagicMock(); resp.status_code = 200
        resp.json.return_value = {"ok": True, "data": {"tasks": [], "markets": [
            {"id": "m-1", "name": "Surenheide"}, {"id": "m-2", "name": "Stromberg"}]}}
        return resp

    with patch("app.services.chat_service.client") as mock_client, \
         patch("httpx.AsyncClient.post", new=fake_post):
        mock_client.chat.completions.create.return_value = _tool_call(
            "propose_task_creation",
            '{"description": "Count stock", "dueDate": null, "note": null, "marketId": "2", "summary": "Create Count stock in Stromberg."}')
        result = _run(service._handle_agent_request(messages, messages[0], "s1", _platform_actor(), "", "English", ""))

    assert calls == ["tasks.list"]
    assert result["proposal"]["args"]["marketId"] == "m-2"


def test_every_read_tool_maps_to_an_operation_and_every_write_to_a_proposal():
    from app.platform.tools import PLATFORM_TOOLS, OPERATION_FOR_TOOL, PROPOSAL_FOR_TOOL
    names = [t["function"]["name"] for t in PLATFORM_TOOLS]
    assert set(names) == set(OPERATION_FOR_TOOL) | set(PROPOSAL_FOR_TOOL)
    assert {"my_profile", "list_markets", "list_departments", "list_checklists", "list_questions",
            "list_assignable_users"} <= set(OPERATION_FOR_TOOL)
    assert {"propose_task_creation": "tasks.create", "propose_task_update": "tasks.update",
            "propose_task_note": "tasks.comment", "propose_task_deletion": "tasks.delete"}.items() <= PROPOSAL_FOR_TOOL.items()
    for tool in PLATFORM_TOOLS:
        fn = tool["function"]
        assert fn["strict"] is True and set(fn["parameters"]["required"]) == set(fn["parameters"]["properties"])


def test_task_deletion_and_update_proposals_need_a_task_id_and_a_change():
    from app.platform.tools import build_proposal, proposal_problem
    assert build_proposal("propose_task_deletion", {"taskId": "t-1", "summary": "Delete task 5 'Fridge'."}) == {
        "operation": "tasks.delete", "args": {"taskId": "t-1"}, "summary": "Delete task 5 'Fridge'."}
    assert build_proposal("propose_task_deletion", {"taskId": "", "summary": "Delete."}) is None
    assert "Look the task up first" in proposal_problem("propose_task_deletion", {}, None, [])

    update = build_proposal("propose_task_update", {"taskId": "t-1", "description": None, "dueDate": "2026-10-01", "summary": "Move it."})
    assert update["args"] == {"taskId": "t-1", "description": None, "dueDate": "2026-10-01"}
    assert build_proposal("propose_task_update", {"taskId": "t-1", "description": None, "dueDate": None, "summary": "x"}) is None
    assert build_proposal("propose_task_note", {"taskId": "t-1", "note": " urgent ", "summary": "Add a note."})["args"]["note"] == "urgent"


def test_deletion_proposal_flows_through_the_loop_without_executing():
    from unittest.mock import MagicMock, patch
    from app.services.chat_service import ChatService
    service = ChatService(MagicMock(), MagicMock())
    messages = [{"role": "user", "content": "delete task 5"}]
    with patch("app.services.chat_service.client") as mock_client, patch("httpx.AsyncClient.post") as never:
        mock_client.chat.completions.create.return_value = _tool_call(
            "propose_task_deletion", '{"taskId": "t-5", "summary": "Delete task 5 (Fridge check)."}')
        result = _run(service._handle_agent_request(messages, messages[0], "s1", _platform_actor(), "", "English", ""))
    never.assert_not_called()
    assert result["proposal"] == {"operation": "tasks.delete", "args": {"taskId": "t-5"}, "summary": "Delete task 5 (Fridge check)."}


CLOSURE = {
    "actor": TURN["actor"],
    "history": [{"role": "user", "content": "create a task"}, {"role": "assistant", "content": "Create the task X."}],
    "proposal": {"operation": "tasks.create", "args": {"description": "X"}, "summary": "Create the task X."},
    "outcome": {"status": "executed", "code": "executed", "taskId": "t-1", "taskNumber": 33823},
}


def test_closure_requires_the_service_secret(secret_configured):
    assert TestClient(app).post("/api/platform/closure", json=CLOSURE).status_code == 401


def test_closure_returns_marieclaires_closing_line(secret_configured):
    from unittest.mock import MagicMock, patch
    with patch("app.platform.closure.client") as mock_client:
        reply = MagicMock(); reply.choices[0].message.content = "Listo: la tarea 33823 quedó creada."
        mock_client.chat.completions.create.return_value = reply
        body = TestClient(app).post("/api/platform/closure", json=CLOSURE, headers=HEADERS).json()
        sent = mock_client.chat.completions.create.call_args.kwargs["messages"]

    assert body["message"] == "Listo: la tarea 33823 quedó creada."
    assert "33823" in sent[-1]["content"] and "ACTION RESULT" in sent[-1]["content"]
    assert any("SPANISH" in m["content"] for m in sent if m["role"] == "system")


def test_closure_failure_yields_no_message_not_an_error(secret_configured):
    from unittest.mock import patch
    with patch("app.platform.closure.client") as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        response = TestClient(app).post("/api/platform/closure", json=CLOSURE, headers=HEADERS)
    assert response.status_code == 200 and response.json()["message"] is None


def test_all_company_proposals_exist_and_need_an_id_or_a_name():
    from app.platform.tools import PROPOSAL_FOR_TOOL, build_proposal
    assert {"markets.create", "markets.update", "markets.delete", "departments.create", "departments.update",
            "departments.delete", "users.update", "users.remove", "checklists.rename", "checklists.set_active",
            "checklists.set_archived"} <= set(PROPOSAL_FOR_TOOL.values())
    assert "users.create" not in PROPOSAL_FOR_TOOL.values()          # passwords never pass through the chat

    assert build_proposal("propose_market_creation", {"name": "Neu", "city": None, "address": None, "postalCode": None,
                                                      "departments": "Frische; Kasse", "summary": "Create market Neu."})["args"]["departments"] == "Frische; Kasse"
    assert build_proposal("propose_market_creation", {"name": "", "summary": "x"}) is None
    assert build_proposal("propose_user_update", {"userId": "u1", "locked": "true", "summary": "Lock Ana."})["args"]["locked"] == "true"
    assert build_proposal("propose_user_update", {"userId": "u1", "summary": "nothing changes"}) is None
    assert build_proposal("propose_checklist_archiving", {"checklistId": "c1", "archived": "true", "summary": "Archive it."})["operation"] == "checklists.set_archived"
    assert build_proposal("propose_department_deletion", {"departmentId": "", "summary": "Delete."}) is None

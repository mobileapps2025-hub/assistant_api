"""Unit tests for the tool registry (app/tools) — schema exposure + arg-passing."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app import tools


def _auth():
    return SimpleNamespace(access_token="t", user_id="u", company_id="c", email="e@x.com")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_exposed_schemas_are_wellformed():
    schemas = tools.tool_schemas()
    assert schemas and all(s["type"] == "function" and s["function"]["name"] for s in schemas)


def test_checklists_tool_exposes_date_range_args():
    spec = tools.get_spec("get_user_checklists")
    props = spec.parameters["properties"]
    assert set(props) == {"date_from", "date_to"}


def test_checklists_handler_defaults_a_year_window_when_no_args():
    mcl = SimpleNamespace(get_checklists_by_date=AsyncMock(return_value=[]))
    _run(tools._user_checklists(mcl, _auth(), {}))
    _, _, date_from, date_to = mcl.get_checklists_by_date.call_args.args
    assert date_from < date_to  # a real window was computed


def test_checklists_handler_honors_supplied_args():
    mcl = SimpleNamespace(get_checklists_by_date=AsyncMock(return_value=[]))
    _run(tools._user_checklists(mcl, _auth(), {"date_from": "2026-01-01T00:00:00", "date_to": "2026-02-01T00:00:00"}))
    _, _, date_from, date_to = mcl.get_checklists_by_date.call_args.args
    assert (date_from, date_to) == ("2026-01-01T00:00:00", "2026-02-01T00:00:00")


def test_no_arg_handler_tolerates_args_dict():
    # count is derived from the Task/ToDos list, so it always matches what the user sees
    mcl = SimpleNamespace(get_task_todos=AsyncMock(return_value=[{"tdo_id": "1"}, {"tdo_id": "2"}, {"tdo_id": "3"}]))
    assert _run(tools._open_task_count(mcl, _auth(), {"ignored": 1})) == {"open_task_count": 3}


def test_open_task_count_excludes_completed_when_flagged():
    mcl = SimpleNamespace(get_task_todos=AsyncMock(return_value=[
        {"tdo_id": "1"}, {"tdo_id": "2", "completed": True}, {"tdo_id": "3", "completed": False},
    ]))
    assert _run(tools._open_task_count(mcl, _auth(), None)) == {"open_task_count": 2}


def test_open_task_count_matches_list_length_without_status_field():
    todos = [{"tdo_id": str(i)} for i in range(5)]
    mcl = SimpleNamespace(get_task_todos=AsyncMock(return_value=todos))
    assert _run(tools._open_task_count(mcl, _auth(), None))["open_task_count"] == len(todos)

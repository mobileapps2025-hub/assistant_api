"""Live data and actions for a platform actor — through MCL.Api, never around it.

Reads go to MCL.Api's operations endpoint with the one-turn pass it issued; MCL.Api applies
the user's permissions and runs its own use-case code. Writes never execute here: MarieClaire
returns a proposal, the app shows it, the user confirms, MCL.Api performs it.
"""
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from app.core.flow import flow
from app.core.logging import get_logger
from app.models import PlatformTurn

logger = get_logger(__name__)

OPERATIONS_TIMEOUT_S = 15

# ---- reads: tool name -> MCL.Api operation -------------------------------------------------

LIST_TASKS = "list_my_tasks"

_READS: List[Dict[str, Any]] = [
    {"tool": LIST_TASKS, "operation": "tasks.list",
     "description": ("List the signed-in user's MCL tasks (live). Use for 'what tasks do I have', 'anything "
                     "overdue', 'my open tasks'. Also returns the user's markets and whether they may create tasks."),
     "params": {"status": {"type": ["string", "null"], "description": "Optional filter: open, done, overdue."},
                "dueBefore": {"type": ["string", "null"], "description": "Optional ISO date; only tasks due before it."}}},
    {"tool": "get_task", "operation": "tasks.get",
     "description": "One of the user's MCL tasks by id (live), with its comments.",
     "params": {"id": {"type": "string"}}},
    {"tool": "my_profile", "operation": "me.profile",
     "description": "Who the signed-in user is: name, company, role, language.", "params": {}},
    {"tool": "list_markets", "operation": "markets.list",
     "description": "The markets/stores of the user's company (live).", "params": {}},
    {"tool": "list_departments", "operation": "departments.list",
     "description": "The departments of the user's company (live).", "params": {}},
    {"tool": "list_department_markets", "operation": "departments.markets",
     "description": "Which markets belong to which department in the user's company (live).", "params": {}},
    {"tool": "list_checklists", "operation": "checklists.list",
     "description": "The checklists configured for the user's company (live): name, periodicity, departments, active.",
     "params": {}},
    {"tool": "list_questions", "operation": "questions.list",
     "description": "The checklist questions configured for the user's company (live).", "params": {}},
    {"tool": "list_assignable_users", "operation": "users.list",
     "description": "The people the user may assign tasks to (live), with their markets.", "params": {}},
]

# ---- writes: tool name -> proposal operation ----------------------------------------------

PROPOSE_TASK = "propose_task_creation"

_WRITES: List[Dict[str, Any]] = [
    {"tool": PROPOSE_TASK, "operation": "tasks.create",
     "description": ("Propose creating a new MCL task. Does NOT create anything: the app shows the proposal "
                     "and the user confirms. Call list_my_tasks first if you haven't: its 'markets' list has "
                     "the user's markets. With several markets, ask which one, then pass its id or exact name."),
     "params": {"description": {"type": "string", "description": "What the task is."},
                "dueDate": {"type": ["string", "null"], "description": "ISO date (YYYY-MM-DD) or null."},
                "note": {"type": ["string", "null"], "description": "Optional first note/comment."},
                "marketId": {"type": ["string", "null"],
                             "description": ("The market's id or exact name from list_my_tasks 'markets'. Null ONLY when "
                                             "that list has a single market. Never leave it null with several markets.")}}},
    {"tool": "propose_task_update", "operation": "tasks.update",
     "description": ("Propose changing one of the user's tasks: its description and/or due date. Use get_task or "
                     "list_my_tasks first to know the task's id and current values."),
     "params": {"taskId": {"type": "string", "description": "The task's id (not its number)."},
                "description": {"type": ["string", "null"], "description": "New description, or null to keep it."},
                "dueDate": {"type": ["string", "null"], "description": "New ISO due date, 'none' to remove it, or null to keep it."}}},
    {"tool": "propose_task_note", "operation": "tasks.comment",
     "description": "Propose adding a note/comment to one of the user's tasks.",
     "params": {"taskId": {"type": "string", "description": "The task's id (not its number)."},
                "note": {"type": "string", "description": "The note text."}}},
    {"tool": "propose_task_deletion", "operation": "tasks.delete",
     "description": ("Propose deleting one of the user's tasks. Irreversible once confirmed, so name the task "
                     "clearly in the summary (number and description)."),
     "params": {"taskId": {"type": "string", "description": "The task's id (not its number)."}}},
]

_SUMMARY_PARAM = {
    "type": "string",
    "description": ("One sentence, in the user's language, saying exactly what will happen — this is what "
                    "the user reads before confirming."),
}


def _schema(spec: Dict[str, Any], with_summary: bool) -> Dict[str, Any]:
    params = dict(spec["params"])
    if with_summary:
        params["summary"] = _SUMMARY_PARAM
    return {
        "type": "function",
        "function": {
            "name": spec["tool"],
            "description": spec["description"],
            "parameters": {"type": "object", "properties": params, "required": list(params),
                           "additionalProperties": False},
            "strict": True,
        },
    }


PLATFORM_TOOLS: List[Dict[str, Any]] = [_schema(s, False) for s in _READS] + [_schema(s, True) for s in _WRITES]

OPERATION_FOR_TOOL = {s["tool"]: s["operation"] for s in _READS}
PROPOSAL_FOR_TOOL = {s["tool"]: s["operation"] for s in _WRITES}

PLATFORM_SURFACE_NOTE = (
    "# THIS SURFACE\n"
    "The user is signed in to the MCL web app; you already know who they are. Their live data comes "
    "from the list_* / get_* / my_profile tools, which MCL itself answers with the user's own "
    "permissions. You never create, change or delete anything yourself: for any change to a task, "
    "call the matching propose_* tool and the app shows the user a card to confirm. A proposal needs "
    "the task's id — get it from list_my_tasks or get_task, never from the user's memory of a number. "
    "Not available from here: creating or editing checklists, markets, departments or users, the sync "
    "snapshot, 'services to download' and notification e-mails — say so plainly and explain how to do "
    "it in MCL instead. Never ask the user to connect or sign in."
)


def is_operation_tool(name: str) -> bool:
    return name in OPERATION_FOR_TOOL


def is_proposal_tool(name: str) -> bool:
    return name in PROPOSAL_FOR_TOOL


async def run_operation(turn: PlatformTurn, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Ask MCL.Api for live data. Returns a JSON-ready dict the model can read either way."""
    operation = OPERATION_FOR_TOOL[tool_name]
    payload = {"operation": operation, "args": {k: v for k, v in args.items() if v is not None}}
    flow(f"🔁 MarieClaire → MCL.Api operations: {operation} (turn {turn.id})")
    try:
        async with httpx.AsyncClient(timeout=OPERATIONS_TIMEOUT_S, verify=verify_tls(turn.operations_url)) as http:
            response = await http.post(
                turn.operations_url, json=payload,
                headers={"Authorization": f"Bearer {turn.token}"},
            )
    except httpx.HTTPError as err:
        logger.warning(f"[PLATFORM] operations call failed: {err}")
        flow("⚠ MCL.Api operations unreachable")
        return {"ok": False, "code": "unreachable",
                "instruction": "Tell the user you couldn't reach their MCL data right now. Do not invent data."}

    body = _json_or_empty(response)
    if response.status_code >= 400 or not body.get("ok", False):
        code = body.get("code") or f"http_{response.status_code}"
        flow(f"⚠ MCL.Api operations refused {operation}: {code}")
        return {"ok": False, "code": code, "instruction": _refusal_instruction(code)}

    data = body.get("data")
    flow(f"✅ MCL.Api operations {operation}: {_describe(data)}")
    return {"ok": True, "operation": operation, "data": data}


def verify_tls(url: str) -> bool:
    """A local MCL.Api runs on the .NET development certificate, which Python does not trust.
    Loopback cannot be intercepted from the network, so only there is verification relaxed."""
    host = (urlparse(url).hostname or "").lower()
    return host not in ("localhost", "127.0.0.1", "::1")


def markets_in(operation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = operation_result.get("data") if operation_result.get("ok") else None
    markets = data.get("markets") if isinstance(data, dict) else None
    return [m for m in markets if isinstance(m, dict) and m.get("id")] if isinstance(markets, list) else []


def resolve_market(value: Any, markets: List[Dict[str, Any]]) -> Optional[str]:
    """The model may write the market's id, its name, or its position in the list it was shown.
    Turn any of those into the id MCL.Api needs; leave it alone if there is nothing to match."""
    text = str(value or "").strip()
    if not text:
        return None
    for market in markets:
        if text.lower() in (str(market["id"]).lower(), str(market.get("name", "")).strip().lower()):
            return str(market["id"])
    if text.isdigit() and 1 <= int(text) <= len(markets):
        return str(markets[int(text) - 1]["id"])
    return text


def build_proposal(tool_name: str, args: Dict[str, Any], markets: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    operation = PROPOSAL_FOR_TOOL[tool_name]
    summary = str(args.get("summary") or "").strip()
    if not summary:
        return None
    clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in args.items() if k != "summary"}
    clean = {k: (v if v not in ("", None) else None) for k, v in clean.items()}
    if operation == "tasks.create":
        if not clean.get("description"):
            return None
        clean["marketId"] = resolve_market(clean.get("marketId"), markets or [])
    elif not clean.get("taskId"):
        return None
    elif operation == "tasks.comment" and not clean.get("note"):
        return None
    elif operation == "tasks.update" and not (clean.get("description") or clean.get("dueDate")):
        return None
    return {"operation": operation, "args": clean, "summary": summary}


def proposal_problem(tool_name: str, args: Dict[str, Any], proposal: Optional[Dict[str, Any]],
                     markets: List[Dict[str, Any]]) -> Optional[str]:
    """Why a proposal must not go to the user yet — or None when it is sound."""
    if proposal is None:
        return _missing_field_instruction(PROPOSAL_FOR_TOOL[tool_name])
    if proposal["operation"] != "tasks.create":
        return None
    market = proposal["args"].get("marketId")
    names = "; ".join(f"{m.get('name', m['id'])} (id {m['id']})" for m in markets)
    if market and markets and market not in {str(m["id"]) for m in markets}:
        return f"'{args.get('marketId')}' is not one of the user's markets. Ask which of these they mean and pass its id: {names}"
    if not market and len(markets) > 1:
        return f"The user has several markets and none was chosen. Ask which one and pass its id: {names}"
    return None


def _missing_field_instruction(operation: str) -> str:
    return {
        "tasks.create": "A description and a one-sentence summary are required. Ask the user for the missing part.",
        "tasks.update": "A task id, at least one change (description or dueDate) and a summary are required. Look the task up first.",
        "tasks.comment": "A task id, the note text and a summary are required. Look the task up first.",
        "tasks.delete": "A task id and a summary naming the task are required. Look the task up first.",
    }.get(operation, "The proposal is incomplete; gather the missing details first.")


def _json_or_empty(response: httpx.Response) -> Dict[str, Any]:
    try:
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}


def _refusal_instruction(code: str) -> str:
    if code in ("expired_grant", "invalid_grant"):
        return "The live-data pass for this turn expired. Tell the user to send the message again."
    if code == "forbidden":
        return "The user is not allowed to see this in MCL. Say so plainly; do not guess the content."
    if code == "not_found":
        return "MCL has no such record for this user. Say so; do not guess."
    if code == "unknown_operation":
        return "This information is not available from the new MCL web app. Say so plainly."
    return "MCL could not answer this request. Tell the user briefly and do not invent data."


def _describe(data: Any) -> str:
    if isinstance(data, list):
        return f"{len(data)} item(s)"
    if isinstance(data, dict):
        return "1 record"
    return "empty" if data is None else type(data).__name__

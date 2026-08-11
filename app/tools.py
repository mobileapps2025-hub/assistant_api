"""MCL tool registry.

One `ToolSpec` per MCL endpoint we might ever expose. Two independent flags gate each tool:

- ``exposed``    — is its schema handed to the model for selection?
- ``executable`` — is it allowed to actually run? (enforcement checks this)

Only functional, verified, read-safe tools are exposed+executable today. Writes, notifications,
auth, broken, and unverified endpoints are registered (so the catalog is complete and the future
policy/danger-review layer has something to flip) but stay gated off. Adding a spec never grants
execution on its own — ``executable`` must be true AND enforcement re-checks it per call.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Handlers receive (mcl_client, auth_context, args) where args is the model-supplied argument
# dict (empty for no-arg tools). Session values (token, user_id, company_id) come from auth.
Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Any]]

# ponytail: the standard MCL task type required by AddTaskMCL; make it an arg if other types are ever needed.
DEFAULT_TASK_TYPE_ID = "AE209949-8B2F-43A2-AC9E-81BCDFAFE899"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    group: str                       # account | checklist | notification | taskapp
    type: str                        # read | write | notify | auth | util
    status: str = "working"          # working | broken | noop | unverified
    risk: str = "safe"               # safe (auto-run) | write | destructive (need confirmation)
    exposed: bool = False
    executable: bool = False
    handler: Optional[Handler] = None
    summary: Optional[Callable[[Dict[str, Any]], str]] = None
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    )

    def summarize(self, args: Dict[str, Any]) -> str:
        # Write tools carry a model-authored `confirmation` string (clear, in the user's
        # language) — that's what the confirmation card shows. No per-tool code needed.
        if self.summary:
            return self.summary(args)
        return args.get("confirmation") or self.description

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": True,
            },
        }


# --- Handlers for the live (executable) tools -------------------------------
# Each is async (mcl_client, auth_context) -> JSON-serialisable data. All args come from the
# authenticated session, so the tools stay no-arg from the model's point of view.

async def _user_info(mcl, auth, args=None) -> Dict[str, Any]:
    info = await mcl.get_user_info(auth.access_token)
    return {
        "full_name": info.get("fullName"),
        "email": info.get("email"),
        "company_id": info.get("companyId"),
        "company_name": info.get("companyName"),
        "role_id": info.get("roleId"),
        "can_create_task": info.get("createTask"),
    }


async def _user_markets(mcl, auth, args=None) -> Any:
    return await mcl.get_markets_by_username(auth.access_token, auth.email)


async def _user_checklists(mcl, auth, args=None) -> Any:
    args = args or {}
    now = datetime.utcnow()
    date_from = args.get("date_from") or (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
    date_to = args.get("date_to") or (now + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
    return await mcl.get_checklists_by_date(auth.access_token, auth.user_id, date_from, date_to)


# Completion flags Task/ToDos might carry. If none is present we treat the item as open, so the
# count always equals the list we display. Verify the real field name against a live token.
_COMPLETION_KEYS = ("completed", "isCompleted", "isDone", "done", "closed", "tdo_completed")


def _is_open(todo: Any) -> bool:
    if not isinstance(todo, dict):
        return True
    for key in _COMPLETION_KEYS:
        if key in todo:
            return not bool(todo[key])
    return True


async def _open_task_count(mcl, auth, args=None) -> Any:
    # Single source of truth: derive the count from the same Task/ToDos list the user sees, so the
    # number can never contradict the list (the upstream GetOpenTaskNumber didn't track create/delete).
    todos = await mcl.get_task_todos(auth.access_token, auth.company_id, auth.user_id)
    todos = todos if isinstance(todos, list) else []
    return {"open_task_count": sum(1 for t in todos if _is_open(t))}


async def _company_questions(mcl, auth, args=None) -> Any:
    return await mcl.get_company_questions(auth.access_token, auth.company_id)


async def _company_departments(mcl, auth, args=None) -> Any:
    return await mcl.get_company_departments(auth.access_token, auth.company_id)


async def _company_departments_markets(mcl, auth, args=None) -> Any:
    return await mcl.get_company_departments_markets(auth.access_token, auth.company_id)


async def _services_to_download(mcl, auth, args=None) -> Any:
    return await mcl.get_services_to_download(auth.access_token, auth.company_id)


async def _synchronization(mcl, auth, args=None) -> Any:
    return await mcl.get_synchronization(auth.access_token, auth.company_id, auth.user_id)


async def _company_emails(mcl, auth, args=None) -> Any:
    return await mcl.get_company_emails(auth.access_token, auth.company_id)


async def _task_users(mcl, auth, args=None) -> Any:
    return await mcl.get_task_users(auth.access_token, auth.company_id)


async def _task_todos(mcl, auth, args=None) -> Any:
    return await mcl.get_task_todos(auth.access_token, auth.company_id, auth.user_id)


async def _delete_task(mcl, auth, args=None) -> Any:
    args = args or {}
    return await mcl.delete_task(auth.access_token, auth.company_id, args.get("todo_id"))


async def _add_task(mcl, auth, args=None) -> Any:
    args = args or {}
    todo = {
        "tdo_description": args.get("description"),
        "tdo_due_date": args.get("due_date"),
        "mkt_id": args.get("market_id"),
        "tdo_user": args.get("assigned_user_id"),
    }
    todo = {k: v for k, v in todo.items() if v is not None}
    todo["tty_id"] = DEFAULT_TASK_TYPE_ID
    if "tdo_user" not in todo and "mkt_id" not in todo:
        todo["tdo_user"] = auth.user_id   # no target given -> assign to the requester
    return await mcl.add_task(auth.access_token, auth.company_id, auth.user_id, todo)


async def _edit_task(mcl, auth, args=None) -> Any:
    args = args or {}
    return await mcl.edit_task(
        auth.access_token, auth.company_id, args.get("todo_id"),
        description=args.get("description"), due_date=args.get("due_date"),
    )


async def _add_task_note(mcl, auth, args=None) -> Any:
    args = args or {}
    return await mcl.add_task_note(
        auth.access_token, auth.company_id, auth.user_id, args.get("todo_id"), args.get("note"),
    )


def _live(name, description, handler, group="checklist", status="working", parameters=None) -> ToolSpec:
    extra = {"parameters": parameters} if parameters else {}
    return ToolSpec(name=name, description=description, group=group, type="read",
                    status=status, exposed=True, executable=True, handler=handler, **extra)


def _params(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


# Optional args must be nullable to satisfy strict function-calling (all keys stay required).
_DATE_RANGE_PARAMS = _params(
    {
        "date_from": {"type": ["string", "null"], "description": "ISO start datetime, or null for the default (last 12 months)"},
        "date_to": {"type": ["string", "null"], "description": "ISO end datetime, or null for the default (next 12 months)"},
    },
    required=["date_from", "date_to"],
)


def _write(name, description, handler, risk, parameters, summary=None, group="checklist") -> ToolSpec:
    return ToolSpec(name=name, description=description, group=group, type="write",
                    status="working", risk=risk, exposed=True, executable=True,
                    handler=handler, summary=summary, parameters=parameters)


def _catalog(name, group, type_, status="working") -> ToolSpec:
    """A registered-but-gated endpoint (no schema-quality description or handler yet)."""
    return ToolSpec(name=name, description=f"[{group}/{type_}] {name} — not yet enabled", group=group, type=type_, status=status)


# Every write tool carries this: the model writes the exact, user-language sentence the
# confirmation card shows. One field for all actions — new actions need no summary code.
_CONFIRMATION_PARAM = {
    "type": "string",
    "description": "A short, clear sentence IN THE USER'S LANGUAGE describing exactly what you "
                   "will do, shown to the user to approve — e.g. 'Delete the task \"QA Fridge Check\".'",
}


TOOL_REGISTRY: List[ToolSpec] = [
    # --- LIVE: personal reads (the user's own data) ---
    _live("get_user_info",
          "Get the current user's profile: full name, email, company name, and role. Use when "
          "the user asks about their own account, who they are, their company, or their role.",
          _user_info, group="account"),
    _live("get_user_markets",
          "Get the markets assigned to the current user. Use when the user asks about their "
          "markets, which are assigned to them, or how many they have.",
          _user_markets),
    _live("get_user_checklists",
          "Get the checklists available to the current user, optionally within a date range. Use "
          "when the user asks about their checklists, which are assigned or pending, or audits/"
          "inspections tied to their account. Pass date_from/date_to to narrow the window.",
          _user_checklists, parameters=_DATE_RANGE_PARAMS),
    _live("get_open_task_count",
          "Get how many open (pending) tasks are assigned to the current user. Use when the user "
          "asks how many tasks they have open, pending, or still to do.",
          _open_task_count),

    # --- LIVE: company config reads (verified 200) ---
    _live("get_company_questions",
          "List the checklist questions configured for the user's company. Use when the user asks "
          "what questions exist, or about the company's checklist question set.",
          _company_questions),
    _live("get_company_departments",
          "List the departments configured for the user's company. Use when the user asks what "
          "departments their company has.",
          _company_departments),
    _live("get_company_departments_markets",
          "List the user's company departments mapped to their markets/stores. Use when the user "
          "asks which markets belong to which department.",
          _company_departments_markets),
    _live("get_services_to_download",
          "List the data services/datasets the app syncs for the user's company. Use when the user "
          "asks what data or services their company has configured to sync.",
          _services_to_download),
    _live("get_company_emails",
          "List the notification email recipients configured for the user's company.",
          _company_emails),
    _live("get_synchronization",
          "Get the full sync snapshot for the user (company config + their data). Use as a fallback "
          "when a more specific tool doesn't cover what the user asked.",
          _synchronization),
    _live("get_task_users",
          "List the users available for task assignment in the user's company.",
          _task_users, group="taskapp", status="unverified"),
    _live("get_task_todos",
          "List the current user's MCL tasks, including tasks created via the assistant. Use when "
          "the user asks to see their tasks or to find a task's id.",
          _task_todos, group="taskapp"),

    # --- LIVE write (needs confirmation via the danger gate) ---
    _write("add_task",
           "Create a new MCL task. Requires a description; optionally a due date, a target market, "
           "or an assigned user.",
           _add_task, risk="write",
           parameters=_params({
               "description": {"type": "string", "description": "What the task is"},
               "due_date": {"type": ["string", "null"], "description": "ISO due datetime, or null"},
               "market_id": {"type": ["string", "null"], "description": "Target market id, or null"},
               "assigned_user_id": {"type": ["string", "null"], "description": "User id to assign, or null"},
               "confirmation": _CONFIRMATION_PARAM,
           }, required=["description", "due_date", "market_id", "assigned_user_id", "confirmation"]),
           group="taskapp"),
    _write("edit_task",
           "Edit one of the user's tasks — change its description and/or due date.",
           _edit_task, risk="write",
           parameters=_params({
               "todo_id": {"type": "string", "description": "The id of the task to edit"},
               "description": {"type": ["string", "null"], "description": "New description, or null to leave unchanged"},
               "due_date": {"type": ["string", "null"], "description": "New ISO due datetime, or null to leave unchanged"},
               "confirmation": _CONFIRMATION_PARAM,
           }, required=["todo_id", "description", "due_date", "confirmation"]),
           group="taskapp"),
    _write("add_task_note",
           "Add a note/comment to one of the user's tasks.",
           _add_task_note, risk="write",
           parameters=_params({
               "todo_id": {"type": "string", "description": "The id of the task"},
               "note": {"type": "string", "description": "The note text to add"},
               "confirmation": _CONFIRMATION_PARAM,
           }, required=["todo_id", "note", "confirmation"]),
           group="taskapp"),
    _write("delete_task",
           "Delete one of the user's tasks by its id. This is destructive and cannot be undone — "
           "only call it when the user clearly wants a specific task deleted.",
           _delete_task, risk="destructive",
           parameters=_params({
               "todo_id": {"type": "string", "description": "The id of the task to delete"},
               "confirmation": _CONFIRMATION_PARAM,
           }, required=["todo_id", "confirmation"]),
           group="taskapp"),

    # --- REGISTERED but GATED (broken / need arguments — wired when fixed) ---
    _catalog("get_company_checklists", "checklist", "read", status="unverified"),
    _catalog("get_company_markets", "checklist", "read", status="broken"),
    # GetTasksMCL inner-joins Markets -> only returns market-assigned tasks; use get_task_todos
    # for a user's tasks. Re-enable when markets are wired.
    _catalog("get_tasks_mcl", "checklist", "read"),
]

_BY_NAME = {spec.name: spec for spec in TOOL_REGISTRY}


def get_spec(name: str) -> Optional[ToolSpec]:
    return _BY_NAME.get(name)


def is_executable(name: str) -> bool:
    spec = _BY_NAME.get(name)
    return bool(spec and spec.executable)


def exposed_specs() -> List[ToolSpec]:
    return [spec for spec in TOOL_REGISTRY if spec.exposed]


def tool_schemas() -> List[Dict[str, Any]]:
    return [spec.schema() for spec in exposed_specs()]


# Backwards-compatible name used by the router (capability-aware prompt) and function-calling.
MCL_USER_TOOLS = tool_schemas()

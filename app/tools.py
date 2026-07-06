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


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    group: str                       # account | checklist | notification | taskapp
    type: str                        # read | write | notify | auth | util
    status: str = "working"          # working | broken | noop | unverified
    exposed: bool = False
    executable: bool = False
    handler: Optional[Handler] = None
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    )

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


async def _open_task_count(mcl, auth, args=None) -> Any:
    return await mcl.get_open_task_count(auth.access_token, auth.user_id)


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


def _catalog(name, group, type_, status="working") -> ToolSpec:
    """A registered-but-gated endpoint (no schema-quality description or handler yet)."""
    return ToolSpec(name=name, description=f"[{group}/{type_}] {name} — not yet enabled", group=group, type=type_, status=status)


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
          "List the to-dos/tasks for the current user in the Task app.",
          _task_todos, group="taskapp", status="unverified"),

    # --- REGISTERED but GATED (broken / need arguments — wired when fixed) ---
    _catalog("get_company_checklists", "checklist", "read", status="unverified"),
    _catalog("get_company_markets", "checklist", "read", status="broken"),
    _catalog("get_tasks_mcl", "checklist", "read"),              # takes a TaskFilter body
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

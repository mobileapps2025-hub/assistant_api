def _no_args_tool(name: str, description: str) -> dict:
    """Build an OpenAI function-tool spec that takes no arguments.

    All MCL user tools operate on the authenticated session (token + user_id
    + company_id come from the request's auth context), so none of them need
    model-supplied arguments.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


MCL_USER_TOOLS = [
    _no_args_tool(
        "get_user_info",
        "Get the current user's profile: full name, email, company name, and "
        "role. Use this when the user asks about their own account, who they "
        "are, their company, or their role/permissions.",
    ),
    _no_args_tool(
        "get_user_markets",
        "Get the markets assigned to the current user. Use this when the user "
        "asks about their markets, which markets are assigned to them, or how "
        "many markets they have.",
    ),
    _no_args_tool(
        "get_user_checklists",
        "Get the checklists available to the current user. Use this when the "
        "user asks about their checklists, which checklists are assigned or "
        "pending, or about audits/inspections tied to their account.",
    ),
    _no_args_tool(
        "get_open_task_count",
        "Get how many open (pending) tasks are assigned to the current user. "
        "Use this when the user asks how many tasks they have open, pending, "
        "or still to do.",
    ),
]

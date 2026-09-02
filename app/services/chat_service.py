import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from zoneinfo import ZoneInfo
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.core.config import client, ENABLE_MCL_IMAGE_VALIDATION
from app.core.logging import get_logger
from app.core.flow import flow
from app.models import AuthContext, Device
from app.tools import MCL_USER_TOOLS, get_spec, exposed_specs
from app.clients.mcl_service_client import MCLServiceClient
from app.services.memory_service import MemoryService
from app.services.gap_service import record_gap
from app.instructions import get_system_prompt, set_request_date
from app.routing import classify_route, detect_language
from app.retrieval import (
    run as run_retrieval, retrieve, build_vision_query, contextualize,
    build_context_sections, render_markers,
)
from app.enforcement import check_tool_call, enforce_answer
from app.enforcement.actions import recall_action_context, record_action
from app.enforcement.pending import create_pending, peek_pending, take_pending
from app.core.localize import localize

logger = get_logger(__name__)

MAX_TOOL_STEPS = 5   # bound the read->act tool loop per request
KNOWLEDGE_TOOL_NAME = "search_mcl_documentation"
ACTION_PLAN_TOOL_NAME = "confirm_mcl_action_plan"
MISSING_INFO_TOOL_NAME = "report_missing_information"

KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": KNOWLEDGE_TOOL_NAME,
        "description": (
            "Search the MCL documentation for general product how-to, troubleshooting, "
            "screen, feature, platform, sync, Dashboard, Mobile App, or Checklist Wizard "
            "questions. Do not use for questions about the assistant's own behavior, "
            "recent actions, available tools, or defaults already described by tool schemas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A standalone English documentation search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

MISSING_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": MISSING_INFO_TOOL_NAME,
        "description": (
            "Report that the MCL documentation you searched does not contain enough information "
            "to answer the user's product question. Call this instead of guessing, right before "
            "telling the user you do not have those details yet. Only for genuine documentation "
            "gaps about how MCL behaves — never for small talk, for questions outside MCL, for "
            "questions about the user's own records, or for a question you should simply ask the "
            "user to clarify."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The user's question as a standalone English sentence, with the "
                        "conversation's context filled in (never a bare 'and how do I do that?')."
                    ),
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_PLAN_STEP_FIELDS = (
    "todo_id", "description", "due_date", "market_id", "assigned_user_id", "note"
)

ACTION_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": ACTION_PLAN_TOOL_NAME,
        "description": (
            "Stage a single confirmation for multiple MCL write/destructive actions. "
            "Use this instead of calling an individual write tool when the user asks for "
            "more than one change, such as deleting the last 2 tasks. The user must approve "
            "the full plan before any step executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "A concise user-language summary listing every action in the plan."
                    ),
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered write/destructive actions to execute after approval.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "enum": ["add_task", "edit_task", "add_task_note", "delete_task"],
                                "description": "The MCL write/destructive tool for this step.",
                            },
                            "summary": {
                                "type": "string",
                                "description": "A clear user-language sentence for this step.",
                            },
                            "todo_id": {"type": ["string", "null"]},
                            "description": {"type": ["string", "null"]},
                            "due_date": {"type": ["string", "null"]},
                            "market_id": {"type": ["string", "null"]},
                            "assigned_user_id": {"type": ["string", "null"]},
                            "note": {"type": ["string", "null"]},
                        },
                        "required": [
                            "tool", "summary", "todo_id", "description", "due_date",
                            "market_id", "assigned_user_id", "note",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["summary", "steps"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

POST_ACTION_MODEL = "gpt-4o"
POST_ACTION_TOOL_STEPS = 3

_POST_ACTION_SYSTEM_PROMPT = """You are MarieClaire after a user approved or rejected a
confirmed MCL action. Give the user useful closure.

Rules:
- Do not perform or suggest another write action as already done.
- You may call SAFE READ tools only when a fresh lookup clearly improves the feedback.
- If the action succeeded, say what changed and infer the most useful next detail from the
  recent conversation. If the user was working from a list, refreshing or summarizing the
  remaining relevant items is often useful. If they were not, keep it brief and offer the
  best next step.
- If the action failed, explain that it did not complete and give the most useful recovery
  step. If a safe lookup can clarify whether the item still exists, use it.
- If a multi-step plan partially failed, clearly say which steps succeeded and which failed.
- Be concise and write in the user's language."""


def _latest_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message
    return None


def _image_urls(message: Dict[str, Any]) -> List[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    urls = []
    for item in content:
        if item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url")
            if url:
                urls.append(url)
    return urls


def _is_authenticated(auth_context: Optional[AuthContext]) -> bool:
    return bool(auth_context and auth_context.access_token)


def _model_content(message: Dict[str, Any]) -> Any:
    """Pass content to the model preserving an attached image; flatten text-only lists."""
    content = message.get("content", "")
    if isinstance(content, list):
        if any(item.get("type") == "image_url" for item in content):
            return content
        return " ".join(item.get("text", "") for item in content if item.get("type") == "text")
    return content or ""


def _preview(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        text = " ".join(i.get("text", "") for i in content if i.get("type") == "text").strip()
        has_image = any(i.get("type") == "image_url" for i in content)
    else:
        text, has_image = str(content or "").strip(), False
    return f"{'[image] ' if has_image else ''}\"{text[:50]}\""


def _no_user_message_response() -> Dict[str, Any]:
    return {"response": "No user message found.", "success": False, "has_vision": False}


def _needs_session_response(language: str = "English") -> Dict[str, Any]:
    return {
        "response": localize(
            "To answer questions about your own MCL data (your profile, "
            "markets, checklists or tasks) I need your MCL session. "
            "Please open the assistant from the MCL app, or connect a token first.",
            language,
        ),
        "success": True,
        "has_vision": False,
    }


def _format_today(timezone: Optional[str]) -> Optional[str]:
    """Today's date in the user's browser timezone, e.g. "Wednesday, 2026-08-04".

    Returns None on a missing or unrecognised zone so the prompt falls back to the server date.
    """
    if not timezone:
        return None
    try:
        return datetime.now(ZoneInfo(timezone)).strftime("%A, %Y-%m-%d")
    except Exception:
        logger.warning(f"[DATE] unrecognised timezone '{timezone}' — using server date")
        return None


def _format_device(device: Optional[Device]) -> str:
    if not device:
        return ""
    label = " ".join(p for p in (device.platform, device.form_factor) if p)
    if device.app_version:
        label = f"{label} (app v{device.app_version})".strip()
    return label.strip()


def _recall_memory(auth_context: Optional[AuthContext], query: str = "") -> str:
    user_id = auth_context.user_id if auth_context else None
    try:
        return MemoryService(user_id).recall_context(query)
    except Exception:
        return ""


def _knowledge_context(chunks: List[Any]) -> str:
    return build_context_sections(chunks) if chunks else ""


def _tool_call_message(tool_call: Any) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }],
    }


def _tool_result_message(tool_call: Any, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call.id, "content": content}


def _safe_read_tool_schemas() -> List[Dict[str, Any]]:
    return [
        spec.schema()
        for spec in exposed_specs()
        if spec.risk == "safe" and spec.executable and spec.handler is not None
    ]


def _write_tool_schemas() -> List[Dict[str, Any]]:
    return [
        spec.schema()
        for spec in exposed_specs()
        if spec.risk != "safe" and spec.executable and spec.handler is not None
    ]


def _plan_step_args(step: Dict[str, Any]) -> Dict[str, Any]:
    args = {field: step.get(field) for field in _PLAN_STEP_FIELDS if step.get(field) is not None}
    args["confirmation"] = step.get("summary") or ""
    return args


def _missing_required_args(spec: Any, args: Dict[str, Any]) -> List[str]:
    missing = []
    for key in spec.parameters.get("required", []):
        if key == "confirmation":
            continue
        schema = spec.parameters.get("properties", {}).get(key, {})
        nullable = isinstance(schema.get("type"), list) and "null" in schema["type"]
        if not nullable and args.get(key) in (None, ""):
            missing.append(key)
    return missing


def _normalize_action_plan(raw_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    steps = raw_args.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return None

    normalized_steps = []
    highest_risk = "write"
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return None
        tool_name = step.get("tool")
        spec = get_spec(tool_name)
        if spec is None or spec.handler is None or spec.risk == "safe":
            return None
        args = _plan_step_args(step)
        if _missing_required_args(spec, args):
            return None
        if spec.risk == "destructive":
            highest_risk = "destructive"
        normalized_steps.append({
            "tool": tool_name,
            "args": args,
            "summary": step.get("summary") or spec.summarize(args),
            "index": index,
            "risk": spec.risk,
        })

    summary = (raw_args.get("summary") or "").strip()
    if not summary:
        summary = "Approve this multi-step MCL action plan:\n" + "\n".join(
            f"{step['index']}. {step['summary']}" for step in normalized_steps
        )
    return {"steps": normalized_steps, "risk": highest_risk, "summary": summary}


def _fallback_action_response(pending: Dict[str, Any], status: str, language: str) -> str:
    summary = pending.get("summary", "")
    results = (pending.get("args") or {}).get("results") or []
    if results:
        lines = []
        for result in results:
            mark = "✓" if result.get("status") == "success" else "⚠"
            line = f"{mark} {result.get('summary') or result.get('tool')}"
            if result.get("status") != "success" and result.get("error"):
                line += f" ({result['error']})"
            lines.append(line)
        return "\n".join(lines)
    if status == "success":
        return f"✓ {summary}"
    return localize("The action failed to complete. Please try again.", language)


def _conversation_snapshot(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    snapshot = []
    for message in messages[-8:]:
        if message.get("role") not in ("user", "assistant"):
            continue
        content = _model_content(message)
        if content:
            snapshot.append({"role": message["role"], "content": content})
    return snapshot


class ChatService:
    def __init__(
        self,
        vision_service: VisionService,
        image_validator_service: ImageValidatorService,
    ):
        self.vision_service = vision_service
        self.image_validator = image_validator_service

    async def process_chat_request(
        self,
        messages: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        auth_context: Optional[AuthContext] = None,
        device: Optional[Device] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        latest_user_message = _latest_user_message(messages)
        if not latest_user_message:
            return _no_user_message_response()

        set_request_date(_format_today(timezone))
        device_context = _format_device(device)
        flow(f"📥 /api/chat · {_preview(latest_user_message)}")
        recall_query = _model_content(latest_user_message)
        if isinstance(recall_query, list):
            recall_query = " ".join(i.get("text", "") for i in recall_query if i.get("type") == "text")
        memory_context = _recall_memory(auth_context, recall_query)
        flow(f"🧠 memory: {'recalled' if memory_context else 'none'}")

        language = detect_language(messages)
        flow(f"🗣 language → {language}")
        if device_context:
            flow(f"📱 device → {device_context}")

        decision = classify_route(messages, tools_catalog=MCL_USER_TOOLS)
        logger.info(
            f"[PREFLIGHT] label={decision.route} authed={_is_authenticated(auth_context)} "
            f"reason={decision.reason[:80]}"
        )
        flow(f"🧭 preflight → {decision.route}  ({decision.reason[:50]})")

        result = await self._dispatch_route(
            decision.route, messages, latest_user_message, session_id, auth_context,
            memory_context, language, device_context
        )
        flow(f"✅ response ready ({decision.route})")
        return result

    async def execute_confirmed_action(
        self, confirmation_id: str, decision: str, auth_context: Optional[AuthContext],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        user_id = auth_context.user_id if auth_context else None
        if decision != "approve":
            lang = (peek_pending(confirmation_id, user_id) or {}).get("language", "English")
            return {"response": localize("Okay — cancelled. No changes were made.", lang), "success": True, "has_vision": False}

        pending = take_pending(confirmation_id, user_id)
        if not pending:
            return {
                "response": "That confirmation has expired or wasn't found. Please ask again.",
                "success": True, "has_vision": False,
            }

        lang = pending.get("language", "English")
        if pending["tool"] == ACTION_PLAN_TOOL_NAME:
            return await self._execute_confirmed_plan(pending, auth_context, session_id=session_id)

        spec = get_spec(pending["tool"])
        if spec is None or spec.handler is None or not check_tool_call(pending["tool"], auth_context).allowed:
            return {"response": localize("I'm not able to perform that action.", lang), "success": True, "has_vision": False}

        flow(f"✅ confirmed → executing {pending['tool']}")
        try:
            await spec.handler(MCLServiceClient(), auth_context, pending["args"])
            record_action(
                session_id, user_id, pending["tool"], pending["args"], pending["summary"]
            )
            response = await self._post_action_feedback(
                pending, "success", auth_context, session_id=session_id
            )
            return {"response": response, "success": True, "has_vision": False}
        except Exception as e:
            logger.error(f"[ACTION] confirmed execution failed: {e}")
            response = await self._post_action_feedback(
                pending, "failed", auth_context, session_id=session_id, error=str(e)
            )
            return {"response": response, "success": True, "has_vision": False}

    async def _execute_confirmed_plan(
        self,
        pending: Dict[str, Any],
        auth_context: Optional[AuthContext],
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        lang = pending.get("language", "English")
        user_id = auth_context.user_id if auth_context else None
        if not _is_authenticated(auth_context):
            return _needs_session_response(lang)

        steps = (pending.get("args") or {}).get("steps") or []
        if not steps:
            return {"response": localize("I'm not able to perform that action.", lang), "success": True, "has_vision": False}

        flow(f"✅ confirmed → executing {len(steps)} planned action(s)")
        mcl = MCLServiceClient()
        results = []
        for step in steps:
            tool_name = step.get("tool")
            spec = get_spec(tool_name)
            summary = step.get("summary") or tool_name
            args = step.get("args") or {}
            if spec is None or spec.handler is None or spec.risk == "safe":
                results.append({"tool": tool_name, "summary": summary, "status": "failed", "error": "tool unavailable"})
                continue
            if not check_tool_call(tool_name, auth_context).allowed:
                results.append({"tool": tool_name, "summary": summary, "status": "failed", "error": "tool blocked"})
                continue
            try:
                await spec.handler(mcl, auth_context, args)
                record_action(session_id, user_id, tool_name, args, summary)
                results.append({"tool": tool_name, "summary": summary, "status": "success"})
            except Exception as step_err:
                logger.error(f"[ACTION_PLAN] step '{tool_name}' failed: {step_err}")
                results.append({
                    "tool": tool_name,
                    "summary": summary,
                    "status": "failed",
                    "error": str(step_err),
                })

        succeeded = sum(1 for result in results if result["status"] == "success")
        status = "success" if succeeded == len(results) else "failed" if succeeded == 0 else "partial_failed"
        pending_with_results = {
            **pending,
            "args": {**(pending.get("args") or {}), "results": results},
        }
        response = await self._post_action_feedback(
            pending_with_results, status, auth_context, session_id=session_id
        )
        return {"response": response, "success": True, "has_vision": False}

    async def _post_action_feedback(
        self,
        pending: Dict[str, Any],
        status: str,
        auth_context: Optional[AuthContext],
        *,
        session_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:
        """Generate context-aware feedback after an approved action resolves.

        The feedback agent can only call safe read tools. If the confirmation came from an
        older path without stored conversation context, keep the previous concise response.
        """
        lang = pending.get("language", "English")
        summary = pending.get("summary", "")
        messages = pending.get("messages") or []
        fallback = _fallback_action_response(pending, status, lang)
        if not messages:
            return fallback

        safe_tools = _safe_read_tool_schemas() if _is_authenticated(auth_context) else []
        action_context = recall_action_context(
            session_id, auth_context.user_id if auth_context else None
        )
        event = {
            "status": status,
            "tool": pending.get("tool"),
            "summary": summary,
            "args": pending.get("args") or {},
            "error": error,
        }
        api_messages = [
            {"role": "system", "content": get_system_prompt("agent", language=lang, tools_catalog=safe_tools)},
            {"role": "system", "content": _POST_ACTION_SYSTEM_PROMPT},
        ]
        if action_context:
            api_messages.append({"role": "system", "content": action_context})
        api_messages.extend(messages)
        api_messages.append({
            "role": "user",
            "content": "# ACTION RESULT\n" + json.dumps(event, ensure_ascii=False),
        })

        try:
            for _ in range(POST_ACTION_TOOL_STEPS):
                response = client.chat.completions.create(
                    model=POST_ACTION_MODEL,
                    messages=api_messages,
                    tools=safe_tools,
                    tool_choice="auto" if safe_tools else "none",
                    parallel_tool_calls=False,
                    temperature=0,
                    timeout=30,
                )
                choice = response.choices[0]
                if not choice.message.tool_calls:
                    content = (choice.message.content or "").strip()
                    return content or fallback

                tool_call = choice.message.tool_calls[0]
                function_name = tool_call.function.name
                spec = get_spec(function_name)
                if spec is None or spec.handler is None or spec.risk != "safe":
                    tool_content = json.dumps({"error": f"The '{function_name}' read tool is not available."})
                else:
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")
                    except (ValueError, TypeError):
                        tool_args = {}
                    try:
                        data = await spec.handler(MCLServiceClient(), auth_context, tool_args)
                        tool_content = json.dumps(data, ensure_ascii=False)
                    except Exception as tool_err:
                        logger.error(f"[POST_ACTION] read tool '{function_name}' failed: {tool_err}")
                        tool_content = json.dumps(
                            {"error": f"The '{function_name}' read tool failed and returned no data."}
                        )
                api_messages.append(_tool_call_message(tool_call))
                api_messages.append(_tool_result_message(tool_call, tool_content))
            return fallback
        except Exception as post_err:
            logger.error(f"[POST_ACTION] feedback generation failed: {post_err}")
            return fallback

    async def _dispatch_route(
        self,
        route: str,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        session_id: Optional[str],
        auth_context: Optional[AuthContext],
        memory_context: str,
        language: str,
        device_context: str,
    ) -> Dict[str, Any]:
        if route == "PERSONAL" and not _is_authenticated(auth_context):
            flow("⛔ no MCL session → ask to connect")
            return _needs_session_response(language)

        if _image_urls(latest_user_message) and route != "PERSONAL":
            return await self._handle_text_request(
                messages, latest_user_message, session_id=session_id,
                memory_context=memory_context, language=language, device_context=device_context
            )

        return await self._handle_agent_request(
            messages, latest_user_message, session_id, auth_context,
            memory_context, language, device_context
        )

    async def _handle_agent_request(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        session_id: Optional[str],
        auth_context: Optional[AuthContext],
        memory_context: str,
        language: str,
        device_context: str,
    ) -> Dict[str, Any]:
        """Manager agent: answer directly, search docs, or use live MCL tools.

        This is the hybrid architecture: preflight still gives observability and an auth
        pre-check, but normal text turns are not locked into a single downstream path.
        The capable model can call the documentation search tool and authenticated MCL
        tools in one bounded loop.
        """
        flow("🤖 AGENT → manager loop")
        mcl_tools = MCL_USER_TOOLS if _is_authenticated(auth_context) else []
        agent_tools = [KNOWLEDGE_TOOL, MISSING_INFO_TOOL, ACTION_PLAN_TOOL, *mcl_tools]
        action_context = recall_action_context(
            session_id, auth_context.user_id if auth_context else None
        )

        system_prompt = get_system_prompt(
            "agent",
            language=language or None,
            device=device_context or None,
            tools_catalog=mcl_tools or None,
            memory=memory_context or None,
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        if action_context:
            api_messages.append({"role": "system", "content": action_context})
        api_messages.extend(
            {"role": m.get("role", "user"), "content": _model_content(m)} for m in messages
        )

        allowed_sources: Set[str] = set()
        retrieved_units: List[Any] = []
        used_knowledge = False
        contextualized_gap = ""

        try:
            for step in range(MAX_TOOL_STEPS):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=api_messages,
                    tools=agent_tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=0,
                    timeout=45,
                )
                choice = response.choices[0]

                if not choice.message.tool_calls:
                    content = (choice.message.content or "").strip()
                    if used_knowledge:
                        content, image_urls = render_markers(content, retrieved_units)
                        flow(f"🖼 rendered {len(image_urls)} screenshot(s)")
                        content = enforce_answer(
                            content,
                            allowed_sources=allowed_sources,
                            allowed_image_urls=set(image_urls),
                        )
                    if content:
                        return {"response": content, "success": True, "has_vision": False}
                    return {
                        "response": localize(
                            "I couldn't complete that in a few steps — could you rephrase or be more specific?",
                            language,
                        ),
                        "success": True,
                        "has_vision": False,
                    }

                tool_call = choice.message.tool_calls[0]
                function_name = tool_call.function.name
                flow(f"🔧 agent tool requested: {function_name}")

                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except (ValueError, TypeError):
                    tool_args = {}

                if function_name == KNOWLEDGE_TOOL_NAME:
                    query = str(tool_args.get("query") or "").strip()
                    contextualized = contextualize(query, messages) if query else ""
                    contextualized_gap = contextualized or contextualized_gap
                    chunks = retrieve(contextualized) if contextualized else []
                    used_knowledge = True
                    retrieved_units.extend(chunks)
                    allowed_sources.update(
                        getattr(c, "document_name", "") for c in chunks if getattr(c, "document_name", "")
                    )
                    tool_content = json.dumps({
                        "query": contextualized,
                        "answer_status": "answered" if chunks else "missing_information",
                        "found": bool(chunks),
                        "sources": sorted(allowed_sources),
                        "context": _knowledge_context(chunks),
                        "instruction": (
                            "Answer documentation facts only from this context and cite each "
                            "documentation claim with [Source: filename]. If this context does "
                            f"not cover the question, call {MISSING_INFO_TOOL_NAME} and then tell "
                            "the user you do not have those details yet — never guess."
                        ),
                    }, ensure_ascii=False)
                    api_messages.append(_tool_call_message(tool_call))
                    api_messages.append(_tool_result_message(tool_call, tool_content))
                    continue

                if function_name == MISSING_INFO_TOOL_NAME:
                    gap_question = str(tool_args.get("question") or "").strip() or contextualized_gap
                    recorded = await record_gap(gap_question, language)
                    flow(f"📝 documentation gap {'recorded' if recorded else 'not recorded'}: {gap_question[:60]}")
                    api_messages.append(_tool_call_message(tool_call))
                    api_messages.append(_tool_result_message(tool_call, json.dumps({
                        "recorded": recorded,
                        "instruction": (
                            "Now tell the user, in their language, that you did not find this in "
                            "the available documentation and that their question has been logged "
                            "to be reviewed and possibly included in a future update. Do not "
                            "promise it will be answered. Do not guess an answer."
                            if recorded else
                            "Tell the user, in their language, that you do not have those details "
                            "yet. Do not mention logging and do not guess an answer."
                        ),
                    })))
                    continue

                if function_name == ACTION_PLAN_TOOL_NAME:
                    if not _is_authenticated(auth_context):
                        return _needs_session_response(language)
                    plan = _normalize_action_plan(tool_args)
                    if not plan:
                        return {
                            "response": localize(
                                "I need a clear multi-step action plan before I can ask for approval. "
                                "Could you rephrase or specify the exact items?",
                                language,
                            ),
                            "success": True,
                            "has_vision": False,
                        }
                    cid = create_pending(
                        ACTION_PLAN_TOOL_NAME,
                        {"steps": plan["steps"]},
                        auth_context.user_id,
                        plan["summary"],
                        plan["risk"],
                        language,
                        messages=_conversation_snapshot(messages),
                    )
                    flow(f"⚠ confirmation required: {ACTION_PLAN_TOOL_NAME} ({plan['risk']})")
                    return {
                        "response": f"⚠ {plan['summary']}",
                        "success": True,
                        "has_vision": False,
                        "requires_confirmation": True,
                        "confirmation": {
                            "id": cid,
                            "risk": plan["risk"],
                            "action_summary": plan["summary"],
                        },
                    }

                if not _is_authenticated(auth_context):
                    return _needs_session_response(language)

                if not check_tool_call(function_name, auth_context).allowed:
                    flow("🛡 enforcement: tool BLOCKED (deny-by-default)")
                    return {"response": "I'm not able to do that.", "success": True, "has_vision": False}
                flow("🛡 enforcement: tool allowed")

                spec = get_spec(function_name)
                if spec is None or spec.handler is None:
                    tool_content = json.dumps(
                        {"error": f"The '{function_name}' tool is not available."}
                    )
                    api_messages.append(_tool_call_message(tool_call))
                    api_messages.append(_tool_result_message(tool_call, tool_content))
                    continue

                if spec.risk != "safe":
                    summary = spec.summarize(tool_args)
                    cid = create_pending(
                        function_name, tool_args, auth_context.user_id, summary, spec.risk,
                        language, messages=_conversation_snapshot(messages)
                    )
                    flow(f"⚠ confirmation required: {function_name} ({spec.risk})")
                    return {
                        "response": f"⚠ {summary}",
                        "success": True,
                        "has_vision": False,
                        "requires_confirmation": True,
                        "confirmation": {"id": cid, "risk": spec.risk, "action_summary": summary},
                    }

                try:
                    data = await spec.handler(MCLServiceClient(), auth_context, tool_args)
                    tool_content = json.dumps(data, ensure_ascii=False)
                except Exception as tool_err:
                    logger.error(f"[AGENT] tool '{function_name}' failed: {tool_err}")
                    flow(f"⚠ tool {function_name} failed — feeding error back to the model")
                    tool_content = json.dumps(
                        {"error": f"The '{function_name}' tool failed (service error) and returned no data."}
                    )
                api_messages.append(_tool_call_message(tool_call))
                api_messages.append(_tool_result_message(tool_call, tool_content))

            return {
                "response": localize(
                    "I couldn't complete that in a few steps — could you rephrase or be more specific?",
                    language,
                ),
                "success": True,
                "has_vision": False,
            }
        except Exception as e:
            logger.error(f"[AGENT] Manager loop error: {e}")
            return {
                "response": localize(
                    "I ran into an issue while handling that. Please try again in a moment.",
                    language,
                ),
                "success": True,
                "has_vision": False,
            }

    async def _handle_personal_request(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        session_id: Optional[str],
        auth_context: Optional[AuthContext],
        memory_context: str,
        language: str,
        device_context: str,
    ) -> Dict[str, Any]:
        flow("👤 PERSONAL (user's own data)")
        if not _is_authenticated(auth_context):
            flow("⛔ no MCL session → ask to connect")
            return _needs_session_response(language)
        tool_result = await self._handle_function_calling(
            messages, latest_user_message, auth_context, memory_context, language, device_context
        )
        if tool_result is not None:
            return tool_result
        return await self._handle_text_request(
            messages, latest_user_message, session_id=session_id,
            memory_context=memory_context, language=language, device_context=device_context
        )

    async def _answer_over_image(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        image_urls: List[str],
        memory_context: str = "",
        language: str = "",
        device_context: str = "",
    ) -> Dict[str, Any]:
        logger.info(f"Answering over {len(image_urls)} image(s)")

        if ENABLE_MCL_IMAGE_VALIDATION:
            for url in image_urls:
                validation = self.image_validator.validate_image(url)
                if not validation["is_mcl"] and validation["confidence"] >= 0.7:
                    return {
                        "response": validation["suggestion"],
                        "success": True,
                        "has_vision": True,
                        "metadata": {"validation_failed": True},
                    }

        search_query = build_vision_query(messages)
        if not search_query:
            raw = latest_user_message.get("content", "")
            if isinstance(raw, list):
                raw = " ".join(item.get("text", "") for item in raw if item.get("type") == "text")
            search_query = raw.strip()
        flow(f"🔎 vision query → '{search_query[:50]}'")

        chunks = retrieve(search_query) if search_query else []
        flow(f"📄 retrieved {len(chunks)} chunk(s) from Ragie")

        api_messages = [
            {"role": "system", "content": get_system_prompt(
                "vision", language=language or None, device=device_context or None,
                memory=memory_context or None
            )}
        ]
        if chunks:
            context = "\n".join(
                f"[Source: {getattr(c, 'document_name', 'unknown')}]: {c.text}" for c in chunks
            )
            api_messages.append({"role": "system", "content": f"# TEXTUAL CONTEXT\n{context}"})
        api_messages.extend(messages)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=api_messages,
                max_tokens=1500,
                temperature=0,
                timeout=30,
            )
            content = response.choices[0].message.content or ""
            if chunks:
                content = enforce_answer(
                    content,
                    allowed_sources={getattr(c, "document_name", "") for c in chunks},
                    allowed_image_urls=set(),
                )
            return {"response": content, "success": True, "has_vision": True}
        except Exception as e:
            logger.error(f"Error in vision processing: {e}")
            return {
                "response": "I encountered an error processing the image.",
                "success": False,
                "has_vision": True,
                "error": str(e),
            }

    async def _handle_function_calling(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        auth_context: AuthContext,
        memory_context: str = "",
        language: str = "",
        device_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Run the GPT tool loop for an authenticated, personal-data or action request.

        Called after the intent is PERSONAL. The model drives (tool_choice="auto"): it
        fetches fresh data for lookups, or — for a change — gathers missing details from the
        user first (returning its question as the reply) and only calls the write tool once
        ready, which pauses for confirmation. Returns a graceful message on error; returns
        None only when the model neither calls a tool nor says anything (defensive fall-through
        to RAG, e.g. a misroute).
        """
        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            user_query = " ".join([item["text"] for item in user_query if item.get("type") == "text"])

        system_prompt = get_system_prompt(
            "tools", language=language or None, device=device_context or None,
            tools_catalog=MCL_USER_TOOLS, memory=memory_context or None
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend({"role": m.get("role"), "content": _model_content(m)} for m in messages)

        try:
            # Multi-step agent loop: the model drives — it fetches (reads), asks the user for
            # missing details before a change, or calls another tool to chain (resolve a task
            # name -> id, then act). tool_choice="auto" lets it reply with a question instead of
            # acting; the instruction file keeps reads honest ("a data question means a fresh
            # lookup"). A write/destructive tool pauses the loop for confirmation.
            for step in range(MAX_TOOL_STEPS):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=api_messages,
                    tools=MCL_USER_TOOLS,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=0,
                    timeout=30,
                )
                choice = response.choices[0]

                if not choice.message.tool_calls:
                    content = (choice.message.content or "").strip()
                    if content:
                        # The model asked the user for details, answered a mid-flow question,
                        # or handled a general question — return it as the reply.
                        return {"response": content, "success": True, "has_vision": False}
                    # Neither a tool call nor any text — a rare stuck state. Fall through to the
                    # legacy language-aware RAG fallback rather than emit a canned line that
                    # would always be English regardless of the user's language.
                    logger.info("[FC] No tool call and no content — falling through to RAG")
                    return None

                tool_call = choice.message.tool_calls[0]
                function_name = tool_call.function.name
                logger.info(f"[FC] Tool called: {function_name}")
                flow(f"🔧 tool requested: {function_name}")

                if not check_tool_call(function_name, auth_context).allowed:
                    flow("🛡 enforcement: tool BLOCKED (deny-by-default)")
                    return {"response": "I'm not able to do that.", "success": True, "has_vision": False}
                flow("🛡 enforcement: tool allowed")

                spec = get_spec(function_name)
                if spec is None or spec.handler is None:
                    logger.warning(f"[FC] Unknown/handlerless tool '{function_name}' — falling through to RAG")
                    return None

                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except (ValueError, TypeError):
                    tool_args = {}

                if spec.risk != "safe":
                    summary = spec.summarize(tool_args)   # model-authored, in the user's language
                    cid = create_pending(function_name, tool_args, auth_context.user_id, summary, spec.risk, language)
                    flow(f"⚠ confirmation required: {function_name} ({spec.risk})")
                    return {
                        "response": f"⚠ {summary}",
                        "success": True,
                        "has_vision": False,
                        "requires_confirmation": True,
                        "confirmation": {"id": cid, "risk": spec.risk, "action_summary": summary},
                    }

                # A single tool failing (e.g. an upstream 500) must not abort the whole flow —
                # feed the error back as the tool result so the model can recover (retry another
                # way, proceed without that data, or tell the user) instead of dying.
                try:
                    data = await spec.handler(MCLServiceClient(), auth_context, tool_args)
                    tool_content = json.dumps(data, ensure_ascii=False)
                except Exception as tool_err:
                    logger.error(f"[FC] tool '{function_name}' failed: {tool_err}")
                    flow(f"⚠ tool {function_name} failed — feeding error back to the model")
                    tool_content = json.dumps(
                        {"error": f"The '{function_name}' tool failed (service error) and returned no data."}
                    )
                api_messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tool_call.id, "type": "function",
                                    "function": {"name": function_name, "arguments": tool_call.function.arguments}}],
                })
                api_messages.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "content": tool_content,
                })

            logger.info("[FC] Max tool steps reached")
            return {
                "response": "I couldn't complete that in a few steps — could you rephrase or be more specific?",
                "success": True, "has_vision": False,
            }

        except Exception as e:
            logger.error(f"[FC] Function calling error: {e}")
            # This is a personal-data query; don't degrade to a RAG
            # "no information found" — return a clear, on-topic message.
            return {
                "response": localize(
                    "I couldn't retrieve your information from MCL right now. "
                    "Please try again in a moment.",
                    language,
                ),
                "success": True,
                "has_vision": False,
            }

    async def _handle_chat(
        self, messages: List[Dict[str, Any]], memory_context: str = "", language: str = "",
        device_context: str = ""
    ) -> Dict[str, Any]:
        flow("💬 CHAT → direct reply")
        system_prompt = get_system_prompt(
            "chat",
            language=language or None,
            device=device_context or None,
            memory=memory_context or None,
            tools_catalog=MCL_USER_TOOLS,
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend({"role": m.get("role", "user"), "content": _model_content(m)} for m in messages)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                max_tokens=300,
                temperature=0.7,
                timeout=15
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"[CHAT] Response: {content[:80]}...")
            return {"response": content, "success": True, "has_vision": False}
        except Exception as e:
            logger.error(f"[CHAT] Error: {e}")
            return {"response": "I'm here! How can I help you with MCL?", "success": True, "has_vision": False}

    async def _handle_text_request(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        session_id: Optional[str] = None,
        memory_context: str = "",
        language: str = "",
        device_context: str = "",
    ) -> Dict[str, Any]:
        image_urls = _image_urls(latest_user_message)
        if image_urls:
            flow("📚 KNOWLEDGE → image path")
            return await self._answer_over_image(
                messages, latest_user_message, image_urls, memory_context, language, device_context
            )

        flow("📚 KNOWLEDGE → text path")
        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            user_query = " ".join(item["text"] for item in user_query if item.get("type") == "text")
        user_query = user_query.strip()

        logger.info(f"[KNOWLEDGE] query='{user_query[:60]}'")

        try:
            result = run_retrieval(
                user_query, messages, language=language or None,
                device=device_context or None, memory=memory_context or None
            )
            return {"response": result["answer"], "success": True, "has_vision": False}
        except Exception as e:
            logger.error(f"[KNOWLEDGE] retrieval failed: {e}")
            return {
                "response": "I encountered an error processing your request.",
                "success": False,
                "has_vision": False,
                "error": str(e),
            }

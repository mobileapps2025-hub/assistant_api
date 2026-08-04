import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.core.config import client, ENABLE_MCL_IMAGE_VALIDATION
from app.core.logging import get_logger
from app.core.flow import flow
from app.models import AuthContext, Device
from app.tools import MCL_USER_TOOLS, get_spec
from app.clients.mcl_service_client import MCLServiceClient
from app.services.memory_service import MemoryService
from app.instructions import get_system_prompt, set_request_date
from app.routing import classify_route, detect_language
from app.retrieval import run as run_retrieval, retrieve, build_vision_query
from app.enforcement import check_tool_call, enforce_answer
from app.enforcement.pending import create_pending, peek_pending, take_pending
from app.core.localize import localize

logger = get_logger(__name__)

MAX_TOOL_STEPS = 5   # bound the read->act tool loop per request


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
            f"[ROUTE] route={decision.route} authed={_is_authenticated(auth_context)} "
            f"reason={decision.reason[:80]}"
        )
        flow(f"🧭 router → {decision.route}  ({decision.reason[:50]})")

        result = await self._dispatch_route(
            decision.route, messages, latest_user_message, session_id, auth_context,
            memory_context, language, device_context
        )
        flow(f"✅ response ready ({decision.route})")
        return result

    async def execute_confirmed_action(
        self, confirmation_id: str, decision: str, auth_context: Optional[AuthContext]
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
        spec = get_spec(pending["tool"])
        if spec is None or spec.handler is None or not check_tool_call(pending["tool"], auth_context).allowed:
            return {"response": localize("I'm not able to perform that action.", lang), "success": True, "has_vision": False}

        flow(f"✅ confirmed → executing {pending['tool']}")
        try:
            await spec.handler(MCLServiceClient(), auth_context, pending["args"])
            # pending['summary'] is the model-authored confirmation, already in the user's language.
            return {"response": f"✓ {pending['summary']}", "success": True, "has_vision": False}
        except Exception as e:
            logger.error(f"[ACTION] confirmed execution failed: {e}")
            return {"response": localize("The action failed to complete. Please try again.", lang), "success": True, "has_vision": False}

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
        if route == "PERSONAL":
            return await self._handle_personal_request(
                messages, latest_user_message, session_id, auth_context,
                memory_context, language, device_context
            )
        if route == "CHAT":
            return await self._handle_chat(messages, memory_context, language, device_context)
        return await self._handle_text_request(
            messages, latest_user_message, session_id=session_id,
            memory_context=memory_context, language=language, device_context=device_context
        )

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
                    # model-driven RAG path (language-aware) rather than emit a canned line that
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

                data = await spec.handler(MCLServiceClient(), auth_context, tool_args)
                api_messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tool_call.id, "type": "function",
                                    "function": {"name": function_name, "arguments": tool_call.function.arguments}}],
                })
                api_messages.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "content": json.dumps(data, ensure_ascii=False),
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

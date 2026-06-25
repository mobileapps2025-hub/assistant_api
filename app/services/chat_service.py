import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.core.config import client, ENABLE_MCL_IMAGE_VALIDATION
from app.core.logging import get_logger
from app.core.flow import flow
from app.models import AuthContext
from app.tools import MCL_USER_TOOLS
from app.clients.mcl_service_client import MCLServiceClient
from app.services.memory_service import MemoryService
from app.instructions import get_system_prompt
from app.routing import classify_route
from app.retrieval import run as run_retrieval, retrieve, build_vision_query
from app.enforcement import check_tool_call, enforce_answer

logger = get_logger(__name__)


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


def _needs_session_response() -> Dict[str, Any]:
    return {
        "response": (
            "To answer questions about your own MCL data (your profile, "
            "markets, checklists or tasks) I need your MCL session. "
            "Please open the assistant from the MCL app, or connect a token first."
        ),
        "success": True,
        "has_vision": False,
    }


def _recall_memory(auth_context: Optional[AuthContext]) -> str:
    user_id = auth_context.user_id if auth_context else None
    try:
        return MemoryService(user_id).recall_context()
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
        auth_context: Optional[AuthContext] = None
    ) -> Dict[str, Any]:
        latest_user_message = _latest_user_message(messages)
        if not latest_user_message:
            return _no_user_message_response()

        flow(f"📥 /api/chat · {_preview(latest_user_message)}")
        memory_context = _recall_memory(auth_context)
        flow(f"🧠 memory: {'recalled' if memory_context else 'none'}")

        decision = classify_route(messages)
        logger.info(
            f"[ROUTE] route={decision.route} authed={_is_authenticated(auth_context)} "
            f"reason={decision.reason[:80]}"
        )
        flow(f"🧭 router → {decision.route}  ({decision.reason[:50]})")

        result = await self._dispatch_route(
            decision.route, messages, latest_user_message, session_id, auth_context, memory_context
        )
        flow(f"✅ response ready ({decision.route})")
        return result

    async def _dispatch_route(
        self,
        route: str,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        session_id: Optional[str],
        auth_context: Optional[AuthContext],
        memory_context: str,
    ) -> Dict[str, Any]:
        if route == "PERSONAL":
            return await self._handle_personal_request(
                messages, latest_user_message, session_id, auth_context, memory_context
            )
        if route == "CHAT":
            return await self._handle_chat(messages, memory_context)
        return await self._handle_text_request(
            messages, latest_user_message, session_id=session_id, memory_context=memory_context
        )

    async def _handle_personal_request(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        session_id: Optional[str],
        auth_context: Optional[AuthContext],
        memory_context: str,
    ) -> Dict[str, Any]:
        flow("👤 PERSONAL (user's own data)")
        if not _is_authenticated(auth_context):
            flow("⛔ no MCL session → ask to connect")
            return _needs_session_response()
        tool_result = await self._handle_function_calling(
            messages, latest_user_message, auth_context, memory_context
        )
        if tool_result is not None:
            return tool_result
        return await self._handle_text_request(
            messages, latest_user_message, session_id=session_id, memory_context=memory_context
        )

    async def _answer_over_image(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        image_urls: List[str],
        memory_context: str = "",
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
            {"role": "system", "content": get_system_prompt("vision", memory=memory_context or None)}
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
        memory_context: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Run GPT function-calling for an authenticated, personal-data query.

        Called only after the intent has been classified as PERSONAL, so a tool
        call is forced (tool_choice="required") — the model must fetch fresh data
        instead of answering from conversation history. Returns a graceful
        message on error; returns None only in the defensive no-tool-call case.
        """
        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            user_query = " ".join([item["text"] for item in user_query if item.get("type") == "text"])

        system_prompt = get_system_prompt("tools", tools_catalog=MCL_USER_TOOLS, memory=memory_context or None)

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend({"role": m.get("role"), "content": _model_content(m)} for m in messages)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=api_messages,
                tools=MCL_USER_TOOLS,
                tool_choice="required",
                temperature=0,
                timeout=30
            )

            choice = response.choices[0]
            if not choice.message.tool_calls:
                logger.info("[FC] No tool calls despite required — falling through to RAG")
                return None

            tool_call = choice.message.tool_calls[0]
            function_name = tool_call.function.name
            logger.info(f"[FC] Tool called: {function_name}")
            flow(f"🔧 tool requested: {function_name}")

            if not check_tool_call(function_name, auth_context).allowed:
                flow("🛡 enforcement: tool BLOCKED (deny-by-default)")
                return {
                    "response": "I'm not able to do that.",
                    "success": True,
                    "has_vision": False,
                }
            flow("🛡 enforcement: tool allowed")

            mcl = MCLServiceClient()
            if function_name == "get_user_info":
                info = await mcl.get_user_info(auth_context.access_token)
                data = {
                    "full_name": info.get("fullName"),
                    "email": info.get("email"),
                    "company_id": info.get("companyId"),
                    "company_name": info.get("companyName"),
                    "role_id": info.get("roleId"),
                    "can_create_task": info.get("createTask"),
                }
            elif function_name == "get_user_markets":
                data = await mcl.get_markets_by_username(
                    auth_context.access_token,
                    auth_context.email,
                )
            elif function_name == "get_user_checklists":
                # CheckListDate needs a window; use a wide range around today.
                now = datetime.utcnow()
                date_from = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
                date_to = (now + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
                data = await mcl.get_checklists_by_date(
                    auth_context.access_token,
                    auth_context.user_id,
                    date_from,
                    date_to,
                )
            elif function_name == "get_open_task_count":
                data = await mcl.get_open_task_count(
                    auth_context.access_token,
                    auth_context.user_id,
                )
            else:
                logger.warning(f"[FC] Unknown tool '{function_name}' — falling through to RAG")
                return None

            tool_result = json.dumps(data, ensure_ascii=False)

            api_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            })
            api_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=api_messages,
                temperature=0,
                timeout=30
            )
            return {
                "response": final_response.choices[0].message.content,
                "success": True,
                "has_vision": False
            }

        except Exception as e:
            logger.error(f"[FC] Function calling error: {e}")
            # This is a personal-data query; don't degrade to a RAG
            # "no information found" — return a clear, on-topic message.
            return {
                "response": (
                    "I couldn't retrieve your information from MCL right now. "
                    "Please try again in a moment."
                ),
                "success": True,
                "has_vision": False,
            }

    async def _handle_chat(self, messages: List[Dict[str, Any]], memory_context: str = "") -> Dict[str, Any]:
        flow("💬 CHAT → direct reply")
        system_prompt = get_system_prompt(
            "chat",
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
        memory_context: str = ""
    ) -> Dict[str, Any]:
        image_urls = _image_urls(latest_user_message)
        if image_urls:
            flow("📚 KNOWLEDGE → image path")
            return await self._answer_over_image(messages, latest_user_message, image_urls, memory_context)

        flow("📚 KNOWLEDGE → text path")
        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            user_query = " ".join(item["text"] for item in user_query if item.get("type") == "text")
        user_query = user_query.strip()

        logger.info(f"[KNOWLEDGE] query='{user_query[:60]}'")

        try:
            result = run_retrieval(user_query, messages, memory=memory_context or None)
            return {"response": result["answer"], "success": True, "has_vision": False}
        except Exception as e:
            logger.error(f"[KNOWLEDGE] retrieval failed: {e}")
            return {
                "response": "I encountered an error processing your request.",
                "success": False,
                "has_vision": False,
                "error": str(e),
            }

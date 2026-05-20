import logging
import time
import os
import re
import json
import cohere
from typing import List, Dict, Any, Optional
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.services.language_service import LanguageService
from app.core.context import ContextAnalysis
from app.core.config import client, ENABLE_MCL_IMAGE_VALIDATION, COHERE_API_KEY
from app.core.logging import get_logger
from app.graph.nodes import AgentNodes
from app.graph.workflow import create_workflow
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.database import CuratedQA
from app.core.config import AsyncSessionLocal
from sqlalchemy import select
from app.models import AuthContext
from app.tools import MCL_USER_TOOLS
from app.clients.mcl_service_client import MCLServiceClient
from app.services.memory_service import MemoryService

logger = get_logger(__name__)

class ChatService:
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        vision_service: VisionService,
        image_validator_service: ImageValidatorService,
        language_service: LanguageService # Injected
    ):
        self.vector_store = vector_store_service
        self.vision_service = vision_service
        self.image_validator = image_validator_service
        self.language_service = language_service 
        
        # Initialize Cohere client if key is available
        self.cohere_client = None
        if COHERE_API_KEY:
            try:
                self.cohere_client = cohere.Client(COHERE_API_KEY)
                logger.info("Cohere client initialized for re-ranking.")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere client: {e}")
        else:
            logger.warning("COHERE_API_KEY not found. Re-ranking will be disabled.")

        # Initialize Graph
        self.nodes = AgentNodes(self.vector_store, self.language_service, self.cohere_client)
        self.workflow = create_workflow(self.nodes)

    async def process_chat_request(
        self,
        messages: List[Dict[str, Any]],
        situational_context: Optional[ContextAnalysis] = None,
        session_id: Optional[str] = None,
        auth_context: Optional[AuthContext] = None
    ) -> Dict[str, Any]:
        """
        Process a chat request, handling both text and vision.
        """
        # 1. Check for images
        has_images = False
        image_urls = []
        latest_user_message = None
        
        for msg in reversed(messages):
            if msg.get("role") == "user":
                latest_user_message = msg
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image_url":
                            has_images = True
                            url = item.get("image_url", {}).get("url")
                            if url:
                                image_urls.append(url)
                break
        
        if not latest_user_message:
             return {
                "response": "No user message found.",
                "success": False,
                "has_vision": False
            }

        # 2. Handle Vision
        if has_images:
            return await self._handle_vision_request(
                messages, 
                image_urls, 
                situational_context
            )
        
        # 3. Try function-calling for authenticated user-specific queries
        if auth_context and auth_context.access_token:
            fc_result = await self._handle_function_calling(
                messages, latest_user_message, auth_context
            )
            if fc_result is not None:
                return fc_result
        
        # 4. Classify intent: chat (small talk) vs MCL query (needs RAG)
        intent = await self._classify_intent(latest_user_message)
        if intent == "CHAT":
            return await self._handle_chat(messages)

        # 5. Handle Text (RAG)
        return await self._handle_text_request(
            messages,
            latest_user_message,
            situational_context,
            session_id=session_id
        )

    async def _handle_vision_request(
        self,
        messages: List[Dict[str, Any]],
        image_urls: List[str],
        situational_context: Optional[ContextAnalysis]
    ) -> Dict[str, Any]:
        logger.info(f"Processing vision request with {len(image_urls)} images")
        
        # Validate images if enabled
        if ENABLE_MCL_IMAGE_VALIDATION:
            for url in image_urls:
                validation = self.image_validator.validate_image(url)
                if not validation["is_mcl"] and validation["confidence"] >= 0.7:
                    return {
                        "response": validation["suggestion"],
                        "success": True,
                        "has_vision": True,
                        "metadata": {"validation_failed": True}
                    }

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500,
                temperature=0,
                timeout=30
            )
            return {
                "response": response.choices[0].message.content,
                "success": True,
                "has_vision": True
            }
        except Exception as e:
            logger.error(f"Error in vision processing: {e}")
            return {
                "response": "I encountered an error processing the image.",
                "success": False,
                "has_vision": True,
                "error": str(e)
            }

    async def _handle_function_calling(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        auth_context: AuthContext
    ) -> Optional[Dict[str, Any]]:
        """
        Try GPT function-calling for authenticated user-specific queries.
        Returns None if no tool was called (fall through to RAG).
        """
        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            user_query = " ".join([item["text"] for item in user_query if item.get("type") == "text"])

        system_prompt = """You are MarieClaire, the MCL Support Specialist.
You have access to tools that can look up user-specific information from the MCL system.
Use these tools when the user asks about their own data (e.g., markets, assignments).
If the user is asking a general MCL question that doesn't require personal data,
do NOT call any tools — let the knowledge base handle it.
Always respond in the same language as the user."""

        api_messages = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
            api_messages.append({"role": role, "content": content or ""})

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=api_messages,
                tools=MCL_USER_TOOLS,
                tool_choice="auto",
                temperature=0,
                timeout=30
            )

            choice = response.choices[0]
            if not choice.message.tool_calls:
                logger.info("[FC] No tool calls — falling through to RAG")
                return None

            tool_call = choice.message.tool_calls[0]
            function_name = tool_call.function.name
            logger.info(f"[FC] Tool called: {function_name}")

            if function_name == "get_user_markets":
                mcl = MCLServiceClient()
                markets = await mcl.get_user_markets(
                    auth_context.access_token,
                    auth_context.company_id,
                    auth_context.user_id
                )
                tool_result = json.dumps(markets, ensure_ascii=False)

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

            return None

        except Exception as e:
            logger.error(f"[FC] Function calling error: {e}")
            return None

    async def _get_curated_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch curated Q&A pairs to use as fallback context.
        Performs simple keyword filtering to avoid polluting context with irrelevant facts.
        """
        if not AsyncSessionLocal:
            return []
            
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(CuratedQA).where(CuratedQA.active == True))
                rows = result.scalars().all()
                
                docs = []
                # Simple keyword extraction (lowercase, split by space)
                # We use a set for O(1) lookups
                query_words = set(query.lower().split())
                
                # Expanded stop words list to prevent false positives
                stop_words = {
                    "the", "is", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", 
                    "what", "how", "why", "when", "where", "who", "which",
                    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their",
                    "do", "does", "did", "can", "could", "will", "would", "should", "have", "has", "had", "be", "am", "are", "was", "were"
                }
                keywords = query_words - stop_words
                
                for row in rows:
                    # Tokenize the curated question to ensure we match whole words, not substrings
                    question_words = set(row.question.lower().split())
                    
                    is_relevant = False
                    if not keywords:
                        # If query has no keywords (e.g. "Hello"), don't inject anything
                        is_relevant = False
                    elif not keywords.isdisjoint(question_words):
                        # Check for intersection between query keywords and question words
                        is_relevant = True
                    
                    if is_relevant:
                        docs.append({
                            "text": f"Question: {row.question}\nAnswer: {row.answer}",
                            "source": "Learned Knowledge",
                            "header_path": "Curated QA",
                            "chunk_index": 0,
                            "score": 1.0 # High confidence
                        })
                return docs
        except Exception as e:
            logger.error(f"Failed to fetch curated knowledge: {e}")
            return []

    async def _classify_intent(self, latest_user_message: Dict[str, Any]) -> str:
        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            user_query = " ".join([item["text"] for item in user_query if item.get("type") == "text"])
        user_query = user_query.strip()
        if not user_query:
            return "CHAT"

        # Fast-path heuristics for very short messages
        short = user_query.lower().rstrip("!?.")
        if short in ("hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "bye", "yes", "no", "test", "testing", "good", "nice", "cool", "great", "lol", "haha"):
            return "CHAT"
        if len(user_query) < 4 and short not in ("mcl", "app", "ios", "bug", "faq"):
            return "CHAT"

        system_prompt = """Classify this message. Reply with exactly one word.

CHAT — the user is just talking: greeting, thanks, small talk, testing the bot, casual conversation, "how are you", "what can you do", or just playing around. Also CHAT if the user is asking about YOU (the bot) or making conversation.

MCL_QUERY — the user wants factual MCL help: how to use a feature, troubleshooting, platform differences, dashboard questions, checklists, tasks, inspections, sync issues.

Output ONLY one word: CHAT or MCL_QUERY"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=10,
                temperature=0,
                timeout=10
            )
            result = response.choices[0].message.content.strip().upper()
            logger.info(f"[CLASSIFY] '{user_query[:60]}' → {result}")
            return "CHAT" if "CHAT" in result else "MCL_QUERY"
        except Exception as e:
            logger.warning(f"[CLASSIFY] Failed, defaulting to MCL_QUERY: {e}")
            return "MCL_QUERY"

    async def _handle_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_prompt = """You are MarieClaire, the MCL Support Specialist. You help users with the MCL (Mobile Checklist) application for retail operations.

You are having a casual conversation right now. Be warm, friendly, and concise.

Your personality:
- Professional but approachable — like a helpful colleague
- Concise — 1 to 3 short sentences unless the user asks for more
- Use plain language, no markdown walls
- If the user is testing you, play along warmly
- If they ask what you can do, briefly mention MCL help: Dashboard, Checklists, Tasks, Inspections, Sync troubleshooting
- If they share personal info, show interest but don't overreact
- Stay on brand as MarieClaire from MCL

MEMORY: You have a persistent memory system. When the user shares personal info (name, preferences, projects), it IS saved for future conversations — tell them "I'll remember that!" or "Got it, saved!". When the user closes the chat or starts a new one, important details are automatically preserved. If they ask "what do you remember", tell them you can recall previous conversations. NEVER say you can't remember or save things.

NEVER fabricate MCL features or steps when just chatting. If they want MCL help, offer to switch modes."""

        api_messages = [{"role": "system", "content": system_prompt}]

        # Inject memory context if available
        try:
            memory_service = MemoryService()
            context = memory_service.recall_context()
            if context:
                api_messages.insert(0, {"role": "system", "content": context})
        except Exception:
            pass

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
            api_messages.append({"role": role, "content": content})

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
        situational_context: Optional[ContextAnalysis],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.monotonic()

        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            # Extract text from list content
            user_query = " ".join([item["text"] for item in user_query if item.get("type") == "text"])

        logger.info(f"Processing text request via Graph: {user_query[:50]}...")
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                 content = " ".join([item["text"] for item in content if item.get("type") == "text"])
            
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        # Fetch Curated Knowledge (Fallback/Augmentation)
        # Filtered by query keywords to prevent context pollution
        curated_docs = await self._get_curated_knowledge(user_query)

        # Invoke Graph
        thread_id = session_id or "anonymous"
        logger.info(f"[SESSION] thread_id={thread_id}")
        inputs = {
            # Full state reset on every invocation — prevents MemorySaver checkpoint
            # from bleeding stale values (e.g. language="de") into a new request.
            "messages": lc_messages,
            "query": user_query,
            "language": "",            # detect_language will overwrite this immediately
            "documents": curated_docs,
            "answer": None,
            "error": None,
            "grade": None,
            "retry_count": 0,
            "contextualized_query": None,
        }
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(
            f"[TRACE] ainvoke inputs: "
            f"query='{user_query[:60]}' | "
            f"language='{inputs['language']}' | "
            f"messages_count={len(lc_messages)} | "
            f"thread_id={thread_id}"
        )

        try:
            result = await self.workflow.ainvoke(inputs, config=config)

            duration_ms = int((time.monotonic() - start_time) * 1000)
            graph_path = _infer_graph_path(result)
            logger.info(
                f"[TRACE] ainvoke result: "
                f"language='{result.get('language')}' | "
                f"grade='{result.get('grade')}' | "
                f"retry_count={result.get('retry_count')} | "
                f"answer_preview='{str(result.get('answer', ''))[:80]}'"
            )
            logger.info(
                f"[CHAT_SUMMARY] query='{user_query[:60]}' "
                f"curated_docs={len(curated_docs)} "
                f"graph_path={graph_path} "
                f"duration_ms={duration_ms}"
            )

            return {
                "response": _rewrite_image_urls(result.get("answer", "No response generated.")),
                "success": True,
                "has_vision": False
            }
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(f"Error in graph execution after {duration_ms}ms: {e}")
            return {
                "response": "I encountered an error processing your request.",
                "success": False,
                "has_vision": False,
                "error": str(e)
            }


def _infer_graph_path(result: dict) -> str:
    """Derive which graph branch was taken from the final state."""
    retry_count = result.get("retry_count", 0)
    grade = result.get("grade", "")
    if grade == "irrelevant" and retry_count == 0:
        return "retrieve→grade→clarify"
    if retry_count >= 1:
        return "retrieve→grade→rewrite→retrieve→generate"
    return "retrieve→grade→generate"


def _rewrite_image_urls(text: str) -> str:
    """Rewrite relative image URLs to absolute backend URLs so they load correctly."""
    import re
    base_url = os.getenv(
        "API_PUBLIC_URL",
        os.getenv("WEBSITE_HOSTNAME", "assistantapi-ctgmb3aad8gvcybg.westeurope-01.azurewebsites.net")
    )
    if not base_url.startswith("http"):
        base_url = f"https://{base_url}"
    return re.sub(r"\]\(images/", f"]({base_url}/images/", text)

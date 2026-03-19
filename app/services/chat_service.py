import logging
import time
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
        situational_context: Optional[ContextAnalysis] = None
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
        
        # 3. Handle Text (RAG)
        return await self._handle_text_request(
            messages, 
            latest_user_message, 
            situational_context
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
                temperature=0.7,
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

    async def _handle_text_request(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        situational_context: Optional[ContextAnalysis]
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
        inputs = {
            "messages": lc_messages,
            "query": user_query,
            "documents": curated_docs, # Pre-load documents with curated knowledge
            "retry_count": 0
        }
        
        try:
            result = await self.workflow.ainvoke(inputs)

            duration_ms = int((time.monotonic() - start_time) * 1000)
            graph_path = _infer_graph_path(result)
            logger.info(
                f"[CHAT_SUMMARY] query='{user_query[:60]}' "
                f"curated_docs={len(curated_docs)} "
                f"graph_path={graph_path} "
                f"duration_ms={duration_ms}"
            )

            return {
                "response": result.get("answer", "No response generated."),
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

import logging
from typing import List, Dict, Any, Optional
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.core.context import ContextAnalysis
from app.core.config import client, ENABLE_MCL_IMAGE_VALIDATION
from app.core.logging import get_logger

logger = get_logger(__name__)

class ChatService:
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        vision_service: VisionService,
        image_validator_service: ImageValidatorService
    ):
        self.vector_store = vector_store_service
        self.vision_service = vision_service
        self.image_validator = image_validator_service

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

        # Prepare messages for Vision API (GPT-4o)
        # We can use the client directly or VisionService. 
        # VisionService.analyze_image_base64 takes a single image and prompt.
        # Here we have a conversation history and potentially multiple images.
        # It's better to use the OpenAI client directly here for full conversation support,
        # or extend VisionService. For now, I'll use the client directly as in legacy_services.
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
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

    async def _handle_text_request(
        self,
        messages: List[Dict[str, Any]],
        latest_user_message: Dict[str, Any],
        situational_context: Optional[ContextAnalysis]
    ) -> Dict[str, Any]:
        
        user_query = latest_user_message.get("content", "")
        if isinstance(user_query, list):
            # Extract text from list content
            user_query = " ".join([item["text"] for item in user_query if item.get("type") == "text"])
            
        logger.info(f"Processing text request: {user_query[:50]}...")
        
        # Detect language
        detected_lang = self._detect_language(user_query)
        
        # RAG Search
        relevant_chunks = self._find_relevant_chunks(user_query, detected_lang)
        
        # Build Context
        context_text = self._build_context_text(relevant_chunks)
        
        # Generate Response
        system_prompt = self._build_system_prompt(detected_lang, situational_context, context_text)
        
        # Prepare messages (replace system prompt or prepend)
        # We construct a new list of messages
        final_messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=final_messages,
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Append sources
            if relevant_chunks:
                sources_header = {
                    'de': "\n\n📚 **Quellen:**\n",
                    'en': "\n\n📚 **Sources:**\n"
                }.get(detected_lang, "\n\n📚 **Sources:**\n")
                
                unique_sources = sorted(list(set([f"{c['document_name']} (Chunk {c['chunk_index']+1})" for c in relevant_chunks])))
                content += sources_header + "\n".join([f"• {s}" for s in unique_sources])

            return {
                "response": content,
                "success": True,
                "has_vision": False
            }
            
        except Exception as e:
            logger.error(f"Error generating text response: {e}")
            return {
                "response": "I encountered an error generating the response.",
                "success": False,
                "has_vision": False,
                "error": str(e)
            }

    def _detect_language(self, text: str) -> str:
        # Simple heuristic + fallback to 'en'
        if not text: return 'en'
        text_lower = text.lower()
        if any(char in text_lower for char in ['ä', 'ö', 'ü', 'ß']): return 'de'
        if any(word in text_lower for word in ['ich', 'kannst', 'mir', 'checkliste']): return 'de'
        return 'en'

    def _find_relevant_chunks(self, query: str, lang: str) -> List[Dict[str, Any]]:
        # 1. Translate if needed
        search_query = query
        if lang != 'en':
            # Simple translation mock or use GPT (omitted for brevity, assuming English docs mostly or simple match)
            # Ideally call a translation helper.
            pass 

        # 2. Semantic Search via VectorStoreService
        # This uses the vector store's search capability
        semantic_results = self.vector_store.search(search_query, limit=10)
        
        # 3. Keyword Search (Hybrid) - Optional but good
        # Access chunks directly from vector_store
        keyword_results = []
        if self.vector_store.chunks:
            query_terms = search_query.lower().split()
            for chunk in self.vector_store.chunks:
                score = 0
                content_lower = chunk['content'].lower()
                for term in query_terms:
                    if term in content_lower:
                        score += 1
                if score > 0:
                    c = chunk.copy()
                    c['keyword_score'] = score
                    keyword_results.append(c)
            keyword_results.sort(key=lambda x: x['keyword_score'], reverse=True)
            keyword_results = keyword_results[:10]

        # Combine results (Simple deduplication)
        seen = set()
        final_results = []
        
        for r in semantic_results:
            if r['chunk_id'] not in seen:
                final_results.append(r)
                seen.add(r['chunk_id'])
        
        for r in keyword_results:
            if r['chunk_id'] not in seen:
                final_results.append(r)
                seen.add(r['chunk_id'])
                
        return final_results[:10]

    def _build_context_text(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks: return ""
        parts = []
        for c in chunks:
            parts.append(f"[From {c['document_name']}]:\n{c['content']}")
        return "\n\n---\n\n".join(parts)

    def _build_system_prompt(self, lang: str, context: Optional[ContextAnalysis], context_text: str) -> str:
        # Simplified prompt construction
        role = "You are 'MCL Assistant', an expert AI assistant for the MCL (Mobile Checklist) application."
        
        if lang == 'de':
            return f"""{role}
⚠️ CRITICAL: Respond in GERMAN (Deutsch).

Context:
{context_text}

Answer the user's question based on the context provided.
"""
        else:
            return f"""{role}

Context:
{context_text}

Answer the user's question based on the context provided.
"""

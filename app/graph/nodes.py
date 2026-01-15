import logging
import os
from typing import Dict, Any, List, Optional
from app.core.state import AgentState
from app.services.vector_store import VectorStoreService
from app.core.config import client
from app.core.logging import get_logger
from app.optimization.dspy_module import RAGModule
from app.services.vector_store import VectorStoreService
from app.services.language_service import LanguageService
import cohere

logger = get_logger(__name__)

class AgentNodes:
    def __init__(
        self, 
        vector_store: VectorStoreService, 
        language_service: LanguageService,
        cohere_client: Optional[cohere.Client] = None
    ):
        self.vector_store = vector_store
        self.language_service = language_service
        self.cohere_client = cohere_client
        
        # Initialize DSPy Module
        self.rag_module = RAGModule()
        compiled_path = "app/optimization/compiled_rag.json"
        if os.path.exists(compiled_path):
            try:
                self.rag_module.load(compiled_path)
                logger.info(f"Loaded compiled DSPy module from {compiled_path}")
            except Exception as e:
                logger.error(f"Failed to load compiled DSPy module: {e}")
        else:
            logger.info("No compiled DSPy module found. Using default.")

    def detect_language(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        lang = self.language_service.detect_language(query)
        logger.info(f"Detected language for query '{query[:20]}...': {lang}")
        return {"language": lang}

    async def retrieve_documents(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        existing_docs = state.get("documents", []) # Preserve existing docs (e.g. curated knowledge)
        
        logger.info(f"Retrieving documents for query: '{query}'")
        
        try:
            # 1. Hybrid Search
            initial_results = self.vector_store.hybrid_search(query, limit=25, alpha=0.5)
            logger.info(f"Hybrid search found {len(initial_results)} documents.")
            
            # 2. Re-ranking
            final_results = initial_results
            if self.cohere_client and initial_results:
                documents = [r.get("text", "") for r in initial_results]
                try:
                    response = self.cohere_client.rerank(
                        model="rerank-english-v3.0",
                        query=query,
                        documents=documents,
                        top_n=10
                    )
                    reranked = []
                    for hit in response.results:
                        if hit.relevance_score > 0.7:
                            original = initial_results[hit.index]
                            original["rerank_score"] = hit.relevance_score
                            reranked.append(original)
                    final_results = reranked
                    logger.info(f"Re-ranking kept {len(final_results)} documents (score > 0.7).")
                except Exception as e:
                    logger.error(f"Re-ranking failed: {e}")
                    final_results = initial_results[:10]
            else:
                final_results = initial_results[:10]
                logger.info(f"Re-ranking skipped or disabled. Using top {len(final_results)} results.")
            
            # Merge with existing docs (Curated Knowledge should take precedence or be included)
            # We prepend existing docs so they appear first in context
            merged_results = existing_docs + final_results
            logger.info(f"Total documents passed to generation: {len(merged_results)} (Curated: {len(existing_docs)}, Retrieved: {len(final_results)})")
                
            return {"documents": merged_results}
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Even if retrieval fails, return existing docs (curated knowledge)
            return {"documents": existing_docs, "error": str(e)}

    async def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        query = state["query"]
        documents = state.get("documents", [])
        
        if not documents:
            return {"grade": "irrelevant"}

        # Concatenate document content for grading
        context_text = "\n".join([d.get("text", "") for d in documents[:3]]) # Check top 3

        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        user_prompt = f"""Retrieved document: \n\n {context_text} \n\n User question: {query}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            grade = response.choices[0].message.content.strip().lower()
            if "yes" in grade:
                return {"grade": "relevant"}
            else:
                return {"grade": "irrelevant"}
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            return {"grade": "relevant"} # Fallback to relevant to avoid loop if error

    async def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Transform the query to produce a better question.
        """
        query = state["query"]
        retry_count = state.get("retry_count", 0)
        
        system_prompt = """You are MarieClaire, an expert AI assistant for the MCL mobile app and dashboard.
        Your task is to rewrite user questions to be optimized for vectorstore retrieval within the MCL knowledge base.
        
        Focus on standard MCL terminology (Tasks, Checklists, Inspections, Sync, Filters).
        Do NOT hallucinate features.
        Do NOT assume specific mappings for unknown terms like "event" unless semantically obvious in the MCL context.
        """
        
        user_prompt = f"Initial question: {query}. Formulate an improved question."

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            better_question = response.choices[0].message.content.strip()
            return {"query": better_question, "retry_count": retry_count + 1}
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return {"retry_count": retry_count + 1} # Return original query but increment count

    async def clarify_ambiguity(self, state: AgentState) -> Dict[str, Any]:
        """
        Ask the user for clarification when retrieval fails.
        """
        query = state["query"]
        lang = state.get("language", "en")
        
        system_prompt = """You are MarieClaire, the MCL Support Specialist.
        The user has asked a question that is not found in the MCL documentation or is unrelated to the MCL app.
        
        Your goal is to politely state that you cannot find the information and ask for clarification, strictly following the Source-Based Truth guideline.
        
        Guideline:
        "I cannot find information regarding that specific topic in the current MCL guides. I can help you only with MCL related information. If this is related to the app, could you please clarify what are you asking for?"
        
        Adapt the language to {lang} if necessary, but keep the meaning identical.
        """
        
        user_prompt = f"User Question: {query}. Generate the clarification response."

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            clarification = response.choices[0].message.content.strip()
            return {"answer": clarification}
        except Exception as e:
            logger.error(f"Clarification generation failed: {e}")
            return {"answer": "I cannot find information regarding that specific topic in the current MCL guides. I can help you only with MCL related information."}

    async def generate_answer(self, state: AgentState) -> Dict[str, Any]:
        lang = state.get("language", "en")
        documents = state.get("documents", [])
        query = state.get("query", "")
        
        # Prepare context list
        context_text = ""
        if documents:
            for c in documents:
                source = c.get('source', 'Unknown')
                header_path = c.get('header_path', 'Root')
                text = c.get('text', '')
                context_text += f"\n[Source: {source} | Section: {header_path}]: {text}\n"
        
        # STRICT GROUNDING CHECK
        if not context_text:
            logger.warning(f"No context available for query: '{query}'. Returning fallback response.")
            # Fallback to clarification node logic if we somehow got here without docs
            return {"answer": "I cannot find information regarding that specific topic in the current MCL guides. I can help you only with MCL related information."}

        system_prompt = f"""# Role & Persona
You are MarieClaire, the MCL Support Specialist, an expert AI assistant dedicated to helping users navigate the MCL ecosystem. Your knowledge base covers the **MCL Mobile App (iOS/Android, Phone/Tablet)**, the **MCL Dashboard**, and the **Checklist Wizard**.

Your goal is to provide clear, step-by-step instructions to troubleshoot issues, guide users through features, and explain role-based permissions.

# CRITICAL INSTRUCTION: LANGUAGE
The user is speaking in **{lang.upper()}**. You **MUST** answer in **{lang.upper()}**.
Translating technical terms:
- Use the standard MCL terminology if it exists in {lang.upper()}.
- If specific terms like "Dashboard" or "Checklist" are commonly used in English even in {lang.upper()} business context, keep them or provide the {lang.upper()} equivalent in parentheses.

# Core Guidelines

1.  **Source-Based Truth:** Answer **only** using the provided context. If a user asks a question not covered by the documentation (e.g., pricing, API integration not mentioned), politely state (in {lang.upper()}): *"I cannot find information regarding that specific topic in the current MCL guides. I can help you only with MCL related information. If this is related to the app, could you please clarify what are you asking for?"*
2.  **Platform Disambiguation:**
    * Many features (e.g., "Creating a Task," "Filtering") exist in both the **Mobile App** and the **Dashboard**.
    * **Always** determine which platform the user is asking about. If it is ambiguous, ask for clarification.
3.  **Device Specifics (Crucial):**
    * **Mobile vs. Tablet:** Watch for UI differences.
    * **iOS vs. Android:** Note specific functional differences.
4.  **Formatting:**
    * Use **Bold** for UI elements.
    * Use Bullet points for lists and numbered lists for sequential steps.
    * Use > Blockquotes for important warnings.

# Handling Specific Scenarios

## 1. Troubleshooting & "Missing" Items
Check "Common Culprits": Sync Status, Filters, Permissions, Connectivity.

## 2. Terminology Handling
* **N.Z. / N.A.:** Treat "N.Z." (German context) and "N.A." (English context) as synonymous ("Not Applicable").

## 3. Creating & Editing Content
* **Checklists:** Differentiate between "Routine" and "Special" inspections.
* **Tasks:** Differentiate between creating a task *inside* a checklist vs. the *Task Menu*.

# Tone
* Professional, helpful, and concise.
* Empathetic to technical frustrations.
"""

        user_prompt = f"""Context Information:
{context_text}

User Question: {query}

Answer as MarieClaire:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            
            # Append sources
            if documents:
                sources_header = {
                    'de': "\n\n📚 **Quellen:**\n",
                    'en': "\n\n📚 **Sources:**\n"
                }.get(lang, "\n\n📚 **Sources:**\n")
                
                unique_sources = sorted(list(set([f"{c.get('document_name', 'Doc')} (Chunk {c.get('chunk_index', 0)+1})" for c in documents])))
                content += sources_header + "\n".join([f"• {s}" for s in unique_sources])
                
            return {"answer": content}
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"answer": "Error generating response.", "error": str(e)}

import logging
from typing import Dict, Any, List, Optional
from app.core.state import AgentState
from app.services.vector_store import VectorStoreService
from app.core.config import client, RERANK_TOP_N, RERANK_THRESHOLD, MAX_CONTEXT_CHARS
from app.core.logging import get_logger
from app.services.language_service import LanguageService
from langchain_core.messages import HumanMessage, AIMessage
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

    def _trim_to_budget(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim document list so total text stays within MAX_CONTEXT_CHARS."""
        trimmed = []
        total = 0
        for doc in documents:
            text_len = len(doc.get("text", ""))
            if total + text_len > MAX_CONTEXT_CHARS:
                if not trimmed:
                    trimmed.append(doc)  # Always include at least one doc
                break
            trimmed.append(doc)
            total += text_len
        if len(trimmed) < len(documents):
            logger.info(f"Context trimmed: {len(documents)} → {len(trimmed)} chunks ({total} chars)")
        return trimmed

    def detect_language(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        lang = self.language_service.detect_language(query)
        logger.info(f"Detected language for query '{query[:20]}...': {lang}")
        return {"language": lang}

    async def retrieve_documents(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        retry_count = state.get("retry_count", 0)
        # On retry after rewrite, start fresh — don't prepend old irrelevant docs
        # (they would end up at positions [:3] and cause the grader to mark as irrelevant again)
        existing_docs = [] if retry_count > 0 else state.get("documents", [])

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
                        top_n=RERANK_TOP_N
                    )
                    reranked = []
                    for hit in response.results:
                        if hit.relevance_score > RERANK_THRESHOLD:
                            original = initial_results[hit.index]
                            original["rerank_score"] = hit.relevance_score
                            reranked.append(original)
                    final_results = reranked
                    logger.info(f"Re-ranking kept {len(final_results)} documents (score > {RERANK_THRESHOLD}).")
                except Exception as e:
                    logger.error(f"Re-ranking failed: {e}")
                    final_results = initial_results[:RERANK_TOP_N]
            else:
                final_results = initial_results[:RERANK_TOP_N]
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
        context_text = "\n".join([d.get("text", "") for d in documents[:5]]) # Check top 5

        system_prompt = """You are a grader assessing whether a retrieved document is relevant to a user question about the MCL mobile checklist app.
Give a binary score: 'yes' if the document contains information that could help answer the question (including out-of-scope guidance, redirects, or statements that a feature is not supported), 'no' if it does not.

Examples:
Document: "To create a new checklist, tap the + button in the top-right corner of the MCL mobile app..."
Question: "How do I add a checklist?"
Score: yes

Document: "The Dashboard shows all active tasks assigned to users in your organisation..."
Question: "What is the price of MCL?"
Score: no

Document: "Sync issues can occur when the device has no internet connection. Check connectivity and tap the sync icon."
Question: "My data is not updating, what should I do?"
Score: yes

Document: "Role-based permissions define what each user can see and edit in the Dashboard."
Question: "How do I delete my account?"
Score: no

Document: "Direct integration with Power BI, SAP, or other third-party platforms is not documented. MCL supports exporting checklists to Excel from the Dashboard."
Question: "Can I export data to Power BI?"
Score: yes

Document: "Password reset is not documented. Contact your MCL Administrator or support@x2-solutions.de."
Question: "How do I reset my password?"
Score: yes

Respond with only 'yes' or 'no'."""

        user_prompt = f"Document:\n{context_text}\n\nQuestion: {query}\nScore:"

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                timeout=30
            )
            grade = response.choices[0].message.content.strip().lower()
            if "yes" in grade:
                return {"grade": "relevant"}
            else:
                return {"grade": "irrelevant"}
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            return {"grade": "irrelevant"}  # Fail safe: route to clarify rather than hallucinate

    async def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Transform the query to produce a better question.
        """
        query = state["query"]
        retry_count = state.get("retry_count", 0)
        messages = state.get("messages", [])

        # Build short conversation history to resolve pronouns and follow-up references
        history_text = ""
        prior_messages = messages[:-1] if messages else []
        recent = prior_messages[-4:] if len(prior_messages) > 4 else prior_messages
        if recent:
            history_lines = []
            for msg in recent:
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    history_lines.append(f"Assistant: {msg.content}")
            if history_lines:
                history_text = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

        system_prompt = """You are MarieClaire, an expert AI assistant for the MCL mobile app and dashboard.
        Your task is to rewrite user questions to be optimized for vectorstore retrieval within the MCL knowledge base.

        Focus on standard MCL terminology (Tasks, Checklists, Inspections, Sync, Filters).
        If the question contains pronouns (it, them, that, this) or is a follow-up referencing prior context,
        use the conversation history to expand it into a self-contained question.
        Do NOT hallucinate features.
        Do NOT assume specific mappings for unknown terms like "event" unless semantically obvious in the MCL context.

        Key MCL domain knowledge for rewrites:
        - Checklists are CREATED in the MCL Dashboard (Checklist Wizard), NOT in the mobile app.
          If a user asks "how to create a checklist in the app", rewrite to "how to create a checklist in the MCL Dashboard".
        - The mobile app is used to RUN/START/COMPLETE checklists, not to create them.
        - Department order can only be changed in the mobile app (drag-and-drop), not the Dashboard.
        - Jumping between departments is iOS-only; not available on Android.
        - 'Audit' is an informal term. In MCL: a one-time/ad-hoc audit = Special Inspection; a recurring audit = Routine Inspection.
          If a user mentions 'audit' without specifying, rewrite to include both Special Inspection and Routine Inspection.
        - Finishing/completing a checklist in the MCL mobile app = tapping "Finish Report" (German: "Bericht abschließen").
          If a user asks how to finish/end/complete a checklist ("Wie beende ich eine Checkliste?"), rewrite to include "Finish Report" and "report types".
        - ALWAYS rewrite non-English questions into English for better retrieval.
        """

        user_prompt = f"{history_text}Initial question: {query}. Formulate an improved, self-contained question."

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                timeout=30
            )
            better_question = response.choices[0].message.content.strip()
            logger.info(f"Query rewritten: '{query}' → '{better_question}'")
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
        logger.info(f"[GRAPH] Path: clarify_ambiguity (retry_count={state.get('retry_count', 0)})")
        
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
                temperature=0,
                timeout=30
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
        messages = state.get("messages", [])
        logger.info(f"[GRAPH] Path: generate_answer (docs={len(documents)}, retry_count={state.get('retry_count', 0)})")

        # Trim documents to context budget before building the prompt
        documents = self._trim_to_budget(documents)

        # Build context text
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
            return {"answer": "I cannot find information regarding that specific topic in the current MCL guides. I can help you only with MCL related information."}

        # Build conversation history from last 3 turns (6 messages), excluding current query
        history_text = ""
        prior_messages = messages[:-1] if messages else []
        recent = prior_messages[-6:] if len(prior_messages) > 6 else prior_messages
        if recent:
            history_lines = []
            for msg in recent:
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    history_lines.append(f"Assistant: {msg.content}")
            if history_lines:
                history_text = "\n# Conversation History\n" + "\n".join(history_lines) + "\n"

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
    * **IMPORTANT:** If the user has already stated their device or platform in the conversation history (e.g., "I'm on a tablet", "I'm using iOS", "I'm in the Dashboard"), answer **exclusively** for that device/platform. Do NOT list instructions for other platforms unless explicitly asked. Tablet users cannot use the swipe method — only the Note method applies.
3.  **Device Specifics (Crucial):**
    * **Mobile vs. Tablet:** Watch for UI differences. On tablets, tasks are completed via the Note method in Tasks Overview — the swipe method is NOT available.
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
* **Audit:** If the user says "audit" without specifying, clarify: a one-time/ad-hoc audit = **Special Inspection**; a recurring/scheduled audit = **Routine Inspection**. Ask which type they need before answering.

## 3. Creating & Editing Content
* **Checklists:** Differentiate between "Routine" and "Special" inspections.
* **Tasks:** Differentiate between creating a task *inside* a checklist vs. the *Task Menu*.

# Tone
* Professional, helpful, and concise.
* Empathetic to technical frustrations.
"""

        user_prompt = f"""Context Information:
{context_text}
{history_text}
User Question: {query}

Answer as MarieClaire:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                timeout=30
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

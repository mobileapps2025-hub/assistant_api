import logging
import os
from typing import Dict, Any, List, Optional
from app.core.state import AgentState
from app.services.vector_store import VectorStoreService
from app.core.config import client
from app.core.logging import get_logger
from app.optimization.dspy_module import RAGModule
import cohere

logger = get_logger(__name__)

class AgentNodes:
    def __init__(self, vector_store: VectorStoreService, cohere_client: Optional[cohere.Client] = None):
        self.vector_store = vector_store
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
        lang = 'en'
        if query:
            text_lower = query.lower()
            if any(char in text_lower for char in ['ä', 'ö', 'ü', 'ß']): lang = 'de'
            elif any(word in text_lower for word in ['ich', 'kannst', 'mir', 'checkliste']): lang = 'de'
        
        return {"language": lang}

    async def retrieve_documents(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        existing_docs = state.get("documents", []) # Preserve existing docs (e.g. curated knowledge)
        
        try:
            # 1. Hybrid Search
            initial_results = self.vector_store.hybrid_search(query, limit=25, alpha=0.5)
            
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
                except Exception as e:
                    logger.error(f"Re-ranking failed: {e}")
                    final_results = initial_results[:10]
            else:
                final_results = initial_results[:10]
            
            # Merge with existing docs (Curated Knowledge should take precedence or be included)
            # We prepend existing docs so they appear first in context
            merged_results = existing_docs + final_results
                
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
        
        system_prompt = """You a question re-writer that converts an input question to a better version that is optimized 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
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

    async def generate_answer(self, state: AgentState) -> Dict[str, Any]:
        lang = state.get("language", "en")
        documents = state.get("documents", [])
        query = state.get("query", "")
        
        # Prepare context list for DSPy
        context_list = []
        if documents:
            for c in documents:
                source = c.get('source', 'Unknown')
                header_path = c.get('header_path', 'Root')
                text = c.get('text', '')
                context_list.append(f"[Source: {source} | Section: {header_path}]: {text}")
        
        try:
            # Use DSPy Module
            # Note: DSPy modules are synchronous by default, but we can run them in a thread if needed.
            # For now, direct call is fine as it's just an API call wrapper.
            
            # If language is German, we might need to adjust the prompt or handle it.
            # The current DSPy signature is English-centric. 
            # For MVP, we'll append a language instruction to the query if needed.
            
            final_query = query
            if lang == 'de':
                final_query += " (Please answer in German)"
            
            prediction = self.rag_module.forward(question=final_query, context=context_list)
            content = prediction.answer
            
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

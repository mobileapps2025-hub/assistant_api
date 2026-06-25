import json
import logging
import re
from typing import Dict, Any, List, Optional
from app.core.state import AgentState
from app.services.vector_store import VectorStoreService
from app.core.config import client, RERANK_TOP_N, RERANK_THRESHOLD, RERANK_HIGH_CONFIDENCE, MAX_CONTEXT_CHARS, SEARCH_LIMIT
from app.core.logging import get_logger
from app.services.language_service import LanguageService
from app.instructions import get_system_prompt
from langchain_core.messages import HumanMessage, AIMessage
import cohere

logger = get_logger(__name__)

IMAGE_MARKDOWN_RE = re.compile(r'!\[[^\]]*\]\([^)]*images/[^)]*\)')
VISUAL_CLAIM_RE = re.compile(
    r'\b(visual guide|visual aid|screenshot|image below|shown below|'
    r'shown in (?:the )?(?:image|screenshot)|see (?:the )?(?:image|screenshot)|pictured)\b',
    re.IGNORECASE,
)

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

    def _source_label(self, item: Dict[str, Any]) -> str:
        """Prefer human titles, then filenames; never return vague Doc labels."""
        for key in ("source_title", "source", "document_name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "Unknown Source"

    def _chunk_number(self, item: Dict[str, Any]) -> int:
        try:
            return int(item.get("chunk_index", 0)) + 1
        except (TypeError, ValueError):
            return 1

    def _known_source_values(self, item: Dict[str, Any]) -> List[str]:
        values = []
        for key in ("source", "source_title", "document_name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip().lower())
        return values

    def _extract_image_markdown(self, text: str) -> List[str]:
        return [match.group(0) for match in IMAGE_MARKDOWN_RE.finditer(text or "")]

    def _visual_caption(self, visual_aid: Dict[str, Any], query: str) -> str:
        header_path = (visual_aid.get("header_path") or "").strip()
        subject = header_path if header_path and header_path != "Root" else self._source_label(visual_aid)
        topic = (query or "this question").strip().rstrip(".?!")[:90]
        return f"*Visual aid: {subject} for your question about {topic}.*"

    def _line_looks_like_visual_caption(self, line: str) -> bool:
        stripped = line.strip().strip("*_").lower()
        return stripped.startswith(("visual aid:", "caption:", "screenshot:", "image:"))

    def _strip_unavailable_image_markdown(
        self,
        answer: str,
        visual_aids: List[Dict[str, Any]],
    ) -> str:
        allowed_images = {
            image
            for visual_aid in visual_aids
            for image in self._extract_image_markdown(visual_aid.get("text", ""))
        }

        def replace_image(match: re.Match) -> str:
            image_markdown = match.group(0)
            return image_markdown if image_markdown in allowed_images else ""

        return IMAGE_MARKDOWN_RE.sub(replace_image, answer)

    def _ensure_visual_captions(
        self,
        answer: str,
        visual_aids: List[Dict[str, Any]],
        query: str,
    ) -> str:
        image_to_caption = {}
        for visual_aid in visual_aids:
            for image in self._extract_image_markdown(visual_aid.get("text", "")):
                image_to_caption[image] = self._visual_caption(visual_aid, query)

        if not image_to_caption:
            return answer

        lines = []
        for line in answer.splitlines():
            image = line.strip()
            if image in image_to_caption:
                previous = next((candidate.strip() for candidate in reversed(lines) if candidate.strip()), "")
                if not self._line_looks_like_visual_caption(previous):
                    lines.append(image_to_caption[image])
            lines.append(line)
        return "\n".join(lines)

    def _remove_visual_claims_without_images(self, answer: str) -> str:
        if IMAGE_MARKDOWN_RE.search(answer):
            return answer

        cleaned_lines = []
        for line in answer.splitlines():
            if not VISUAL_CLAIM_RE.search(line):
                cleaned_lines.append(line)
                continue

            sentences = re.split(r'(?<=[.!?])\s+', line)
            kept_sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip() and not VISUAL_CLAIM_RE.search(sentence)
            ]
            if kept_sentences:
                cleaned_lines.append(" ".join(kept_sentences))

        return "\n".join(cleaned_lines).strip()

    def _postprocess_visual_aids(
        self,
        answer: str,
        visual_aids: List[Dict[str, Any]],
        query: str,
    ) -> str:
        answer = self._strip_unavailable_image_markdown(answer, visual_aids)
        answer = self._ensure_visual_captions(answer, visual_aids, query)
        answer = self._remove_visual_claims_without_images(answer)
        return re.sub(r'\n{3,}', '\n\n', answer).strip()

    def _grade_context_items(self, query: str, items: List[Dict[str, Any]], label: str) -> tuple[List[Dict[str, Any]], int]:
        """Return context items that are relevant to the query."""
        if not items:
            return [], 0

        items_to_grade = items[:8]
        doc_list_lines = []
        for i, item in enumerate(items_to_grade):
            snippet = item.get("text", "")[:400].replace("\n", " ")
            escaped = json.dumps(snippet)
            doc_list_lines.append(f'  {{"doc_index": {i}, "text": {escaped}}}')
        doc_list_json = "[\n" + ",\n".join(doc_list_lines) + "\n]"

        system_prompt = f"""You are a grader for an MCL mobile-checklist-app support system.
For each {label}, decide if it could help answer the user question.
Mark relevant=true if it contains anything useful for the question.
Mark relevant=false ONLY if it has zero overlap with the question topic.

Return ONLY a JSON object with this exact shape (no markdown, no prose):
{{"grades": [{{"doc_index": 0, "relevant": true}}, {{"doc_index": 1, "relevant": false}}, ...]}}"""

        user_prompt = (
            f"Question: {query}\n\n"
            f"{label.title()} items:\n{doc_list_json}\n\n"
            "Grade each item and return the JSON:"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=30
        )
        raw = response.choices[0].message.content.strip()
        grades_data = json.loads(raw)
        grades = grades_data.get("grades", [])
        relevant_indices = {g["doc_index"] for g in grades if g.get("relevant") in (True, "true", "yes")}
        relevant_items = [items_to_grade[i] for i in sorted(relevant_indices)]
        return relevant_items, len(items_to_grade)

    def detect_language(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        lang_before = state.get("language", "<not set>")
        lang = self.language_service.detect_language(query)
        logger.info(
            f"[TRACE] detect_language: "
            f"query='{query[:60]}' | "
            f"language_in_state_before='{lang_before}' | "
            f"detected='{lang}'"
        )
        return {"language": lang}

    async def contextualize_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Enrich the user's query with conversation context so retrieval is self-contained.
        Fast-paths (no LLM call) on the first turn.
        """
        query = state["query"]
        messages = state.get("messages", [])

        logger.info(
            f"[TRACE] contextualize_query: "
            f"language_in_state='{state.get('language')}' | "
            f"messages_count={len(messages)} | "
            f"query='{query[:60]}'"
        )
        # Fast-path: first real question is already self-contained. This also
        # covers the case where the only prior message is the assistant's
        # welcome greeting (no real user turn to resolve against) — rewriting
        # then only risks injecting unwanted assumptions.
        prior_messages = messages[:-1] if messages else []
        has_prior_user_turn = any(isinstance(m, HumanMessage) for m in prior_messages)
        if len(messages) <= 1 or not has_prior_user_turn:
            logger.info("[TRACE] contextualize_query: self-contained question, no LLM call")
            return {"contextualized_query": query}

        recent = prior_messages[-4:] if len(prior_messages) > 4 else prior_messages
        history_lines = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_lines.append(f"Assistant: {msg.content}")
        last_4_messages = "\n".join(history_lines)

        system_prompt = """You are a query preprocessing assistant for the MCL knowledge base,
which covers BOTH the MCL mobile app and the MCL Dashboard (web admin).
Rewrite the user's latest question into a self-contained search query that can retrieve
the right documents from a vector database without conversation context.

Rules:
- Resolve all pronouns ("it", "that", "them", "this") to the actual MCL entity.
- Expand follow-up questions into full standalone queries.
- If already self-contained, return it unchanged.
- Do NOT add platform, device, or surface words (e.g. "mobile app", "Dashboard",
  "iOS", "Android", "tablet", "web") that the user did not explicitly mention —
  many actions live on a different surface than the user assumes.
- Output ONLY the rewritten query, no explanation.
- Always output in English.

MCL domain: Tasks, Checklists, Routine Inspections, Special Inspections, Dashboard,
Finish Report, Sync, Filters, Departments, iOS, Android, Tablet, Phone layout."""

        user_prompt = f"""Conversation history (most recent first):
{last_4_messages}

Latest user question: {query}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                timeout=15
            )
            contextualized = response.choices[0].message.content.strip()
            logger.info(f"contextualize_query: '{query}' → '{contextualized}'")
            return {"contextualized_query": contextualized}
        except Exception as e:
            logger.warning(f"contextualize_query failed, using original: {e}")
            return {"contextualized_query": query}

    async def retrieve_documents(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("contextualized_query") or state["query"]
        retry_count = state.get("retry_count", 0)
        # Use alpha override set by rewrite_query for this retry, then clear it
        search_alpha = state.get("search_alpha_override") or 0.5
        logger.info(
            f"[TRACE] retrieve_documents: "
            f"language_in_state='{state.get('language')}' | "
            f"retry_count={retry_count} | "
            f"search_alpha={search_alpha} | "
            f"query='{query[:60]}'"
        )
        # On retry after rewrite, start fresh — don't prepend old irrelevant docs
        existing_docs = [] if retry_count > 0 else state.get("documents", [])

        logger.info(f"Retrieving documents for query: '{query}'" + (f" (original: '{state['query']}')" if query != state["query"] else ""))

        TEXT_DOC_TYPES = ["faq", "platform_note", "assistant_identity", "curated"]

        try:
            # 1. Text-track hybrid search — excludes visual_guide chunks
            initial_results = self.vector_store.hybrid_search(
                query, limit=SEARCH_LIMIT, alpha=search_alpha,
                doc_types=TEXT_DOC_TYPES,
            )
            logger.info(f"Text-track hybrid search: {len(initial_results)} results (alpha={search_alpha}).")

            # 2. Visual-track hybrid search — visual_guide chunks only, small k
            visual_results = self.vector_store.hybrid_search(
                query, limit=3, alpha=search_alpha,
                doc_types=["visual_guide"],
            )
            logger.info(f"Visual-track hybrid search: {len(visual_results)} results.")

            # 3. Re-ranking applies to text track only
            final_results = initial_results
            if self.cohere_client and initial_results:
                documents = [r.get("text", "") for r in initial_results]
                try:
                    response = self.cohere_client.rerank(
                        model="rerank-english-v3.0",
                        query=state.get("contextualized_query") or query,
                        documents=documents,
                        top_n=RERANK_TOP_N
                    )
                    reranked = []
                    for hit in response.results:
                        score = hit.relevance_score
                        if score < RERANK_THRESHOLD:
                            continue  # truly irrelevant — drop
                        original = initial_results[hit.index].copy()
                        original["rerank_score"] = score
                        # Soft tier: high (≥0.5) or medium (0.15–0.5) — both pass grading
                        original["confidence_tier"] = "high" if score >= RERANK_HIGH_CONFIDENCE else "medium"
                        reranked.append(original)
                    final_results = reranked
                    logger.info(
                        f"Re-ranking kept {len(final_results)} documents "
                        f"(threshold={RERANK_THRESHOLD}, high_confidence={RERANK_HIGH_CONFIDENCE})."
                    )
                except Exception as e:
                    logger.error(f"Re-ranking failed: {e}")
                    final_results = initial_results[:RERANK_TOP_N]
            else:
                final_results = initial_results[:RERANK_TOP_N]
                logger.info(f"Re-ranking skipped. Using top {len(final_results)} results.")

            merged_results = existing_docs + final_results
            logger.info(
                f"Total documents for grading: {len(merged_results)} "
                f"(curated={len(existing_docs)}, retrieved={len(final_results)})"
            )
            return {"documents": merged_results, "visual_aids": visual_results, "search_alpha_override": None}
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"documents": existing_docs, "visual_aids": [], "error": str(e)}

    async def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """
        Grade EACH retrieved document individually in a single LLM call via JSON output.
        Eliminates bulk-grading non-determinism: one weak doc can no longer flip the whole set.
        """
        query = state["query"]
        documents = state.get("documents", [])
        visual_aids = state.get("visual_aids", [])
        logger.info(
            f"[TRACE] grade_documents: "
            f"language_in_state='{state.get('language')}' | "
            f"docs_count={len(documents)} | visual_aids_count={len(visual_aids)}"
        )

        if not documents:
            if visual_aids:
                try:
                    relevant_visual_aids, total_visual_graded = self._grade_context_items(query, visual_aids, "visual aid")
                    logger.info(
                        f"grade_documents: {len(relevant_visual_aids)}/{total_visual_graded} visual aids passed with no text docs"
                    )
                    if relevant_visual_aids:
                        return {
                            "grade": "relevant",
                            "documents": [],
                            "visual_aids": relevant_visual_aids,
                            "relevant_count": 0,
                            "total_graded": total_visual_graded,
                        }
                except Exception as e:
                    logger.error(f"Visual-aid grading failed without text docs: {e}")
            logger.info("[TRACE] grade_documents: no documents → irrelevant")
            return {"grade": "irrelevant", "relevant_count": 0, "total_graded": 0}

        # Grade up to 8 docs; beyond that use the first 8 (already ranked by relevance)
        docs_to_grade = documents[:8]

        # Build numbered doc list for the prompt (proper JSON escaping)
        doc_list_lines = []
        for i, doc in enumerate(docs_to_grade):
            snippet = doc.get("text", "")[:400].replace("\n", " ")
            escaped = json.dumps(snippet)  # handles ", \, control chars
            doc_list_lines.append(f'  {{"doc_index": {i}, "text": {escaped}}}')
        doc_list_json = "[\n" + ",\n".join(doc_list_lines) + "\n]"

        system_prompt = """You are a grader for an MCL mobile-checklist-app support system.
For each document, decide if it could help answer the user question.
Mark relevant=true if the document contains ANYTHING useful:
  - Direct answers to the question
  - Statements that a feature is not supported
  - Redirects to the correct place (e.g. "do this in the Dashboard")
  - Related troubleshooting steps

Mark relevant=false ONLY if the document has zero overlap with the question topic.

Return ONLY a JSON object with this exact shape (no markdown, no prose):
{"grades": [{"doc_index": 0, "relevant": true}, {"doc_index": 1, "relevant": false}, ...]}"""

        user_prompt = (
            f"Question: {query}\n\n"
            f"Documents:\n{doc_list_json}\n\n"
            "Grade each document and return the JSON:"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=30
            )
            raw = response.choices[0].message.content.strip()
            grades_data = json.loads(raw)
            grades = grades_data.get("grades", [])

            # Filter: keep only docs graded relevant=true (truthy check handles bool & string)
            relevant_indices = {g["doc_index"] for g in grades if g.get("relevant") in (True, "true", "yes")}
            relevant_docs = [docs_to_grade[i] for i in sorted(relevant_indices)]

            # Append any docs beyond the graded window unchanged
            extra_docs = documents[8:]

            total_graded = len(docs_to_grade)
            relevant_count = len(relevant_docs)
            logger.info(
                f"grade_documents: {relevant_count}/{total_graded} docs passed "
                f"(+{len(extra_docs)} ungraded overflow)"
            )

            if relevant_count == 0 and not extra_docs:
                if visual_aids:
                    try:
                        relevant_visual_aids, total_visual_graded = self._grade_context_items(query, visual_aids, "visual aid")
                        logger.info(
                            f"grade_documents: {len(relevant_visual_aids)}/{total_visual_graded} visual aids passed after text docs failed"
                        )
                        if relevant_visual_aids:
                            return {
                                "grade": "relevant",
                                "documents": [],
                                "visual_aids": relevant_visual_aids,
                                "relevant_count": 0,
                                "total_graded": total_graded + total_visual_graded,
                            }
                    except Exception as e:
                        logger.error(f"Visual-aid grading failed after text docs failed: {e}")
                return {
                    "grade": "irrelevant",
                    "documents": [],
                    "relevant_count": 0,
                    "total_graded": total_graded,
                }

            return {
                "grade": "relevant",
                "documents": relevant_docs + extra_docs,
                "relevant_count": relevant_count,
                "total_graded": total_graded,
            }

        except Exception as e:
            logger.error(f"Grading failed: {e}. Falling back to all docs as relevant.")
            # On JSON parse error, pass all docs through to avoid false negatives
            return {
                "grade": "relevant",
                "documents": documents,
                "relevant_count": len(documents),
                "total_graded": len(documents),
            }

    async def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Transform the query to produce a better question.
        Three distinct strategies based on retry_count so each attempt
        explores a different retrieval dimension.
        """
        query = state["query"]
        retry_count = state.get("retry_count", 0)
        messages = state.get("messages", [])

        # Build short conversation history
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

        # ── Strategy selection ──────────────────────────────────────────────
        # retry_count==0 → Strategy 1: MCL terminology mapping (same as before)
        # retry_count==1 → Strategy 2: Broad synonym expansion (semantic-heavy α=0.8)
        # retry_count==2 → Strategy 3: Query decomposition into sub-questions (keyword-heavy α=0.2)

        if retry_count == 0:
            # Strategy 1: MCL terminology alignment
            system_prompt = """You are MarieClaire, an expert AI assistant for the MCL mobile app and dashboard.
Your task is to rewrite the user question to be optimized for vectorstore retrieval within the MCL knowledge base.

Focus on standard MCL terminology (Tasks, Checklists, Inspections, Sync, Filters).
If the question contains pronouns (it, them, that, this) or is a follow-up, use conversation history to make it self-contained.
Do NOT hallucinate features.

Key MCL domain knowledge for rewrites:
- Checklists are CREATED in the MCL Dashboard (Checklist Wizard), NOT in the mobile app.
- The mobile app is used to RUN/START/COMPLETE checklists, not to create them.
- Department order can only be changed in the mobile app (drag-and-drop), not the Dashboard.
- Jumping between departments is iOS-only; not available on Android.
- 'Audit' in MCL: one-time/ad-hoc audit = Special Inspection; recurring audit = Routine Inspection.
- Finishing/completing a checklist = tapping "Finish Report" (German: "Bericht abschließen").
- ALWAYS rewrite non-English questions into English for better retrieval.

Output ONLY the rewritten question, no explanation."""
            alpha_override = None  # keep default 0.5

        elif retry_count == 1:
            # Strategy 2: Broad synonym expansion — semantic search will find near-matches
            system_prompt = """You are a search query expander for a vector database about the MCL app.
The previous retrieval attempt failed. Expand the user question by listing synonyms and related concepts.

Rules:
- Include alternative terms separated by OR (e.g., "Dashboard OR web interface OR admin panel OR management view")
- For MCL concepts: Dashboard = web admin, Task = work item = activity, Checklist = inspection form = audit form
- Add context words that co-occur with the concept in documentation
- Keep the result as a single expanded search string
- Do NOT add questions or explanation, output ONLY the expanded query

Example input: "What is the Dashboard?"
Example output: "Dashboard OR web admin panel OR management interface MCL features overview navigation sections"""
            alpha_override = 0.8  # tilt heavily toward semantic matching

        else:
            # Strategy 3: Decompose into sub-questions, keyword-heavy to catch exact terms
            system_prompt = """You are a query decomposition assistant for an MCL documentation search.
The previous retrieval attempts failed. Break the question into 2-3 shorter sub-questions that each target one specific concept.

Rules:
- Produce at most 3 sub-questions separated by " | "
- Each sub-question should contain specific MCL keywords (Dashboard, Task, Checklist, Sync, Filter, etc.)
- Focus on noun phrases — avoid filler words
- Output ONLY the sub-questions joined by " | ", no explanation

Example input: "Why am I not seeing my checklists in the app after my supervisor made changes?"
Example output: "checklist not visible mobile app | checklist sync not updating | supervisor changes not reflected app"""
            alpha_override = 0.2  # tilt heavily toward keyword/BM25 matching

        user_prompt = f"{history_text}Question: {query}\nRewritten:"

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
            logger.info(
                f"Query rewrite strategy {retry_count + 1}: "
                f"'{query[:60]}' → '{better_question[:60]}'"
                + (f" | alpha_override={alpha_override}" if alpha_override else "")
            )
            result: Dict[str, Any] = {
                "query": better_question,
                "contextualized_query": better_question,
                "retry_count": retry_count + 1,
            }
            if alpha_override is not None:
                result["search_alpha_override"] = alpha_override
            return result
        except Exception as e:
            logger.error(f"Query rewrite failed (strategy {retry_count + 1}): {e}")
            return {"retry_count": retry_count + 1}

    async def clarify_ambiguity(self, state: AgentState) -> Dict[str, Any]:
        """
        Reached only after 3 distinct retrieval strategies are exhausted.
        Instead of a dead-end, show what WAS found and suggest related MCL topics.
        """
        query = state["query"]
        lang = state.get("language", "en")
        documents = state.get("documents", [])
        logger.info(
            f"[TRACE] clarify_ambiguity: "
            f"language='{lang}' | "
            f"retry_count={state.get('retry_count', 0)} | "
            f"available_docs={len(documents)}"
        )

        # Collect any topic hints from whatever docs were retrieved
        found_topics: List[str] = []
        for doc in documents[:5]:
            header = doc.get("header_path", "")
            source = doc.get("document_name", doc.get("source", ""))
            if header and header != "Root" and header not in found_topics:
                found_topics.append(header)
            elif source and source not in found_topics:
                found_topics.append(source)
        found_topics = found_topics[:3]

        related_hint = ""
        if found_topics:
            related_hint = (
                f"\nRelated topics found in documentation: {', '.join(found_topics)}."
            )

        system_prompt = f"""You are MarieClaire, the MCL Support Specialist.
The user asked a question that could not be answered after exhaustive search of the MCL documentation.

Respond in **{lang.upper()}** language.

Your response MUST:
1. Acknowledge you could not find the specific information.
2. Mention any related topics that were found (if provided below).
3. Offer 3-4 concrete MCL topics you CAN answer, for example:
   - Dashboard overview and navigation
   - Creating and managing checklists (Checklist Wizard)
   - Tasks: creation, completion on mobile and tablet
   - Routine Inspections vs Special Inspections
   - Sync issues and troubleshooting
   - Roles and permissions
   - Filters and search in the mobile app
4. Ask if any of those might be what they are looking for, OR ask them to rephrase.

Do NOT say "I don't know" without offering alternatives.
Do NOT invent features or answers not backed by documentation.
{related_hint}"""

        user_prompt = f"User question: {query}\nGenerate a helpful clarification response in {lang.upper()}:"

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
            return {
                "answer": (
                    "I cannot find information regarding that specific topic in the current MCL guides. "
                    "I can help you with Dashboard navigation, Checklists, Tasks, Inspections, Sync, "
                    "Roles and Permissions, or Filters. Could you clarify what you are looking for?"
                )
            }

    async def generate_answer(self, state: AgentState) -> Dict[str, Any]:
        lang = state.get("language", "en")
        documents = state.get("documents", [])
        visual_aids = state.get("visual_aids", [])
        query = state.get("query", "")
        messages = state.get("messages", [])
        visual_aids_with_images = [
            visual_aid
            for visual_aid in visual_aids
            if self._extract_image_markdown(visual_aid.get("text", ""))
        ]
        logger.info(
            f"[TRACE] generate_answer: "
            f"language='{lang}' | "
            f"docs={len(documents)} | "
            f"visual_aids={len(visual_aids)} | "
            f"retry_count={state.get('retry_count', 0)} | "
            f"query='{query[:60]}'"
        )

        documents = self._trim_to_budget(documents)

        # Build TEXTUAL CONTEXT block
        context_text = ""
        if documents:
            for c in documents:
                source = c.get("source") or self._source_label(c)
                source_title = self._source_label(c)
                title_part = f" | Title: {source_title}" if source_title != source else ""
                header_path = c.get('header_path', 'Root')
                text = c.get('text', '')
                context_text += f"\n[Source: {source}{title_part} | Section: {header_path}]: {text}\n"

        # Build AVAILABLE VISUAL AIDS block (omitted entirely if empty)
        visual_context = ""
        if visual_aids_with_images:
            for va in visual_aids_with_images:
                source = va.get("source") or self._source_label(va)
                source_title = self._source_label(va)
                title_part = f" | Title: {source_title}" if source_title != source else ""
                header_path = va.get('header_path', 'Root')
                text = va.get('text', '')
                visual_context += f"\n[Caption: {header_path} | File: {source}{title_part}]\n{text}\n"

        # Helpful "no context" fallback — name the topic + 5 MCL pillars.
        # Visual-guide chunks include prose plus images and are enough context by themselves.
        if not context_text and not visual_context:
            topic_hint = query[:60] if query else "that topic"
            logger.warning(f"No context for query '{topic_hint}'. Returning informative fallback.")
            return {
                "answer": (
                    f"I could not find documentation specifically about \"{topic_hint}\" in the current MCL guides.\n\n"
                    "I CAN help you with these MCL topics:\n"
                    "• **Dashboard** — overview, navigation, and management features\n"
                    "• **Checklists** — creating and editing via the Checklist Wizard\n"
                    "• **Tasks** — creating, completing on mobile (phone/tablet), and tracking\n"
                    "• **Inspections** — Routine Inspections vs Special Inspections\n"
                    "• **Sync & Troubleshooting** — connectivity, filters, and permissions\n\n"
                    "Is any of these related to what you were looking for? Feel free to ask again."
                )
            }

        # Build conversation history
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

        # Layer 1 — Instruction File: CORE identity + RAG-mode addendum + language slot.
        # The dynamic context (TEXTUAL CONTEXT / VISUAL AIDS / history / question) stays in
        # the user_prompt below.
        system_prompt = get_system_prompt("rag", language=lang)

        user_prompt = f"""# TEXTUAL CONTEXT
{context_text}"""

        if visual_context:
            user_prompt += f"""
# AVAILABLE VISUAL AIDS
{visual_context}"""

        user_prompt += f"""
{history_text}
User Question: {query}

Answer as MarieClaire (cite sources inline with [Source: filename]):"""

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

            # Post-generation grounding validation (visual_aids count as known sources too)
            content = self._validate_grounding(content, documents, visual_aids_with_images)
            content = self._postprocess_visual_aids(content, visual_aids_with_images, query)

            # Append sources footer
            if documents:
                sources_header = {
                    'de': "\n\n📚 **Quellen:**\n",
                    'en': "\n\n📚 **Sources:**\n"
                }.get(lang, "\n\n📚 **Sources:**\n")
                unique_sources = sorted(list(set([
                    f"{self._source_label(c)} (Chunk {self._chunk_number(c)})"
                    for c in documents
                ])))
                content += sources_header + "\n".join([f"• {s}" for s in unique_sources])

            return {"answer": content}

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"answer": "Error generating response.", "error": str(e)}

    def _validate_grounding(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
        visual_aids: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Post-generation grounding check:
        1. Verifies that [Source: X] citations in the answer match actual retrieved documents.
        2. Strips sentences containing fabricated citations (source not in doc list).
        3. If the answer has NO citations at all, prepends a grounding caveat.

        Visual-aid sources (from the visual track) are also counted as valid known
        sources so that legitimate references to e.g. main-navigation-menu.md are
        not mistakenly flagged as fabricated.
        """
        if not documents and not visual_aids:
            return answer

        # Build set of known source filenames/titles (case-insensitive) from BOTH tracks
        known_sources = {
            source_value
            for doc in (documents or [])
            for source_value in self._known_source_values(doc)
        }
        for va in (visual_aids or []):
            known_sources.update(self._known_source_values(va))

        # Extract all [Source: X] citations
        citation_pattern = re.compile(r'\[Source:\s*([^\]]+)\]')
        found_citations = citation_pattern.findall(answer)

        if not found_citations:
            # No inline citations at all — prepend grounding note
            logger.info("_validate_grounding: no inline citations found; prepending grounding note.")
            return "Based on the available MCL documentation:\n\n" + answer

        # Check each cited source against known documents
        fabricated = [
            c for c in found_citations
            if c.strip().lower() not in known_sources
        ]

        if not fabricated:
            return answer  # All citations valid

        logger.warning(
            f"_validate_grounding: fabricated citations detected: {fabricated}. "
            "Stripping affected sentences."
        )

        # Split into sentences, remove any containing a fabricated citation
        fabricated_lower = {f.strip().lower() for f in fabricated}
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        clean_sentences = []
        for sentence in sentences:
            sentence_citations = citation_pattern.findall(sentence)
            if any(c.strip().lower() in fabricated_lower for c in sentence_citations):
                logger.debug(f"_validate_grounding: dropped sentence: {sentence[:80]}")
                continue
            clean_sentences.append(sentence)

        cleaned = " ".join(clean_sentences).strip()
        if not cleaned:
            # Everything was stripped — return safe fallback
            return (
                "I found some related documentation, but could not confirm specific details. "
                "Please consult your MCL administrator or contact support@x2-solutions.de."
            )
        return cleaned

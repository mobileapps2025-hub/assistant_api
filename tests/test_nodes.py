import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.graph.nodes import AgentNodes
from app.services.vector_store import VectorStoreService
from app.services.language_service import LanguageService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    mock = MagicMock(spec=VectorStoreService)
    mock.hybrid_search.return_value = []
    return mock


@pytest.fixture
def mock_language_service():
    mock = MagicMock(spec=LanguageService)
    mock.detect_language.return_value = "en"
    return mock


@pytest.fixture
def nodes(mock_vector_store, mock_language_service):
    return AgentNodes(mock_vector_store, mock_language_service, cohere_client=None)


@pytest.fixture
def mock_openai_response():
    """Reusable factory for a mock OpenAI completion response."""
    def _make(content: str):
        response = MagicMock()
        response.choices[0].message.content = content
        return response
    return _make


def base_state(**overrides):
    state = {
        "query": "How do I create a checklist?",
        "messages": [],
        "documents": [],
        "retry_count": 0,
        "language": "en",
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:

    def test_detects_english(self, nodes, mock_language_service):
        mock_language_service.detect_language.return_value = "en"
        result = nodes.detect_language(base_state(query="How do I sync?"))
        assert result["language"] == "en"

    def test_detects_german(self, nodes, mock_language_service):
        mock_language_service.detect_language.return_value = "de"
        result = nodes.detect_language(base_state(query="Wie synchronisiere ich?"))
        assert result["language"] == "de"


# ---------------------------------------------------------------------------
# _trim_to_budget
# ---------------------------------------------------------------------------

class TestTrimToBudget:

    def test_no_trimming_needed(self, nodes):
        docs = [{"text": "short"}, {"text": "also short"}]
        with patch("app.graph.nodes.MAX_CONTEXT_CHARS", 10000):
            result = nodes._trim_to_budget(docs)
        assert len(result) == 2

    def test_trims_to_budget(self, nodes):
        docs = [{"text": "a" * 500}, {"text": "b" * 500}, {"text": "c" * 500}]
        with patch("app.graph.nodes.MAX_CONTEXT_CHARS", 600):
            result = nodes._trim_to_budget(docs)
        assert len(result) == 1

    def test_always_keeps_at_least_one(self, nodes):
        docs = [{"text": "a" * 5000}]
        with patch("app.graph.nodes.MAX_CONTEXT_CHARS", 10):
            result = nodes._trim_to_budget(docs)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# retrieve_documents
# ---------------------------------------------------------------------------

class TestRetrieveDocuments:

    @pytest.mark.asyncio
    async def test_returns_hybrid_search_results(self, nodes, mock_vector_store):
        mock_vector_store.hybrid_search.return_value = [
            {"text": "doc1", "source": "s1"},
            {"text": "doc2", "source": "s2"},
        ]
        result = await nodes.retrieve_documents(base_state())
        assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_prepends_existing_documents(self, nodes, mock_vector_store):
        existing = [{"text": "curated", "source": "Learned Knowledge"}]
        mock_vector_store.hybrid_search.return_value = [{"text": "retrieved"}]
        result = await nodes.retrieve_documents(base_state(documents=existing))
        # Curated docs come first
        assert result["documents"][0]["text"] == "curated"
        assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_returns_existing_docs_on_retrieval_error(self, nodes, mock_vector_store):
        existing = [{"text": "curated"}]
        mock_vector_store.hybrid_search.side_effect = Exception("weaviate down")
        result = await nodes.retrieve_documents(base_state(documents=existing))
        assert result["documents"] == existing
        assert "error" in result


# ---------------------------------------------------------------------------
# grade_documents
# ---------------------------------------------------------------------------

class TestGradeDocuments:

    @pytest.mark.asyncio
    async def test_grades_relevant(self, nodes, mock_openai_response):
        response = mock_openai_response("yes")
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            state = base_state(documents=[{"text": "To create a checklist, tap +."}])
            result = await nodes.grade_documents(state)
        assert result["grade"] == "relevant"

    @pytest.mark.asyncio
    async def test_grades_irrelevant(self, nodes, mock_openai_response):
        # New: grade_documents uses JSON; a 'no' plain-text response cannot be parsed
        # → fails → fallback passes all docs as relevant (fail-open to avoid false negatives)
        response = mock_openai_response("no")  # not valid JSON
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            state = base_state(documents=[{"text": "Unrelated content."}])
            result = await nodes.grade_documents(state)
        # Fallback behaviour: relevant (all docs pass through)
        assert result["grade"] == "relevant"

    @pytest.mark.asyncio
    async def test_grades_irrelevant_on_api_error(self, nodes):
        # New: on API error, fail-open — pass all docs through to avoid false negatives
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("timeout")
            state = base_state(documents=[{"text": "some doc"}])
            result = await nodes.grade_documents(state)
        assert result["grade"] == "relevant"

    @pytest.mark.asyncio
    async def test_grades_irrelevant_when_no_documents(self, nodes):
        result = await nodes.grade_documents(base_state(documents=[]))
        assert result["grade"] == "irrelevant"


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------

class TestRewriteQuery:

    @pytest.mark.asyncio
    async def test_rewrites_query(self, nodes, mock_openai_response):
        response = mock_openai_response("How do I add a new checklist in MCL?")
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = await nodes.rewrite_query(base_state(query="how add chcklist", retry_count=0))
        assert result["query"] == "How do I add a new checklist in MCL?"
        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_increments_retry_on_error(self, nodes):
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("api down")
            result = await nodes.rewrite_query(base_state(retry_count=0))
        assert result["retry_count"] == 1
        # Original query preserved (not in result dict means state keeps it)
        assert "query" not in result


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:

    @pytest.mark.asyncio
    async def test_generates_answer_with_context(self, nodes, mock_openai_response):
        response = mock_openai_response("Tap the + button to create a checklist.")
        docs = [
            {
                "text": "To create a checklist, tap the + button.",
                "source": "guide.md",
                "header_path": "Checklists",
                "document_name": "guide.md",
                "chunk_index": 0,
            }
        ]
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            result = await nodes.generate_answer(base_state(documents=docs))
        assert "Tap the + button" in result["answer"]

    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_documents(self, nodes):
        result = await nodes.generate_answer(base_state(documents=[]))
        answer = result["answer"].lower()
        # New helpful fallback says "could not find" and lists MCL topics
        assert "could not find" in answer or "cannot find" in answer
        assert any(t in answer for t in ["dashboard", "checklist", "task", "inspection", "sync"])

    @pytest.mark.asyncio
    async def test_returns_error_message_on_api_failure(self, nodes):
        docs = [{"text": "some context", "source": "s", "header_path": "H", "document_name": "d", "chunk_index": 0}]
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("api error")
            result = await nodes.generate_answer(base_state(documents=docs))
        assert "error" in result["answer"].lower()


# ---------------------------------------------------------------------------
# _validate_grounding  (NEW)
# ---------------------------------------------------------------------------

class TestValidateGrounding:

    def test_valid_citation_passes_through(self, nodes):
        docs = [{"source": "platform_notes.md"}]
        answer = "The Dashboard has tasks [Source: platform_notes.md]. That covers it."
        result = nodes._validate_grounding(answer, docs)
        assert result == answer

    def test_adds_preamble_when_no_citations(self, nodes):
        docs = [{"source": "platform_notes.md"}]
        answer = "The Dashboard lets you manage everything."
        result = nodes._validate_grounding(answer, docs)
        assert result.startswith("Based on the available MCL documentation:")
        assert "Dashboard" in result

    def test_strips_fabricated_citation_sentence(self, nodes):
        docs = [{"source": "platform_notes.md"}]
        answer = (
            "Use the Dashboard [Source: platform_notes.md]. "
            "You can export to SAP [Source: sap_integration.md]. "
            "Filters are in the sidebar [Source: platform_notes.md]."
        )
        result = nodes._validate_grounding(answer, docs)
        # Sentence with fabricated source should be gone
        assert "sap_integration.md" not in result
        assert "SAP" not in result
        # Valid sentences must remain
        assert "Dashboard" in result
        assert "Filters" in result

    def test_empty_documents_returns_answer_unchanged(self, nodes):
        answer = "Some answer with no citations."
        result = nodes._validate_grounding(answer, [])
        assert result == answer

    def test_fabricated_only_returns_safe_fallback(self, nodes):
        docs = [{"source": "real.md"}]
        # Every sentence has a fabricated citation
        answer = "Step one [Source: fake1.md]. Step two [Source: fake2.md]."
        result = nodes._validate_grounding(answer, docs)
        assert "fake1.md" not in result
        assert "fake2.md" not in result
        # Must return the safe fallback, not an empty string
        assert len(result) > 0
        assert "consult" in result.lower() or "support" in result.lower()


# ---------------------------------------------------------------------------
# clarify_ambiguity  (NEW — helpful fallback after 3 exhausted retries)
# ---------------------------------------------------------------------------

class TestClarifyAmbiguity:

    @pytest.mark.asyncio
    async def test_offers_mcl_topics_in_response(self, nodes):
        """clarify_ambiguity must offer concrete MCL topics, not just 'I don't know'."""
        with patch("app.graph.nodes.client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = (
                "I couldn't find info on that. I can help with: "
                "Dashboard, Checklists, Tasks, Sync, Roles."
            )
            mock_client.chat.completions.create.return_value = mock_response

            result = await nodes.clarify_ambiguity(
                base_state(query="What is the DeepScan feature?", retry_count=3)
            )
        answer = result["answer"]
        # At least one MCL topic must be suggested
        assert any(
            topic.lower() in answer.lower()
            for topic in ["Dashboard", "Checklist", "Task", "Sync", "Inspection", "Role"]
        ), f"No MCL topics offered in clarification: {answer}"

    @pytest.mark.asyncio
    async def test_uses_correct_language(self, nodes):
        """clarify_ambiguity must inject the user's language into the prompt."""
        with patch("app.graph.nodes.client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Ich konnte keine Information finden."
            mock_client.chat.completions.create.return_value = mock_response

            await nodes.clarify_ambiguity(
                base_state(query="Was ist der DeepScan?", language="de", retry_count=3)
            )
            call_args = mock_client.chat.completions.create.call_args
            system_msg = call_args.kwargs["messages"][0]["content"]
            assert "DE" in system_msg, "Language 'de' not injected into clarify_ambiguity prompt"

    @pytest.mark.asyncio
    async def test_returns_fallback_on_api_error(self, nodes):
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("timeout")
            result = await nodes.clarify_ambiguity(base_state(retry_count=3))
        answer = result["answer"]
        assert answer
        assert any(
            topic.lower() in answer.lower()
            for topic in ["Dashboard", "Checklist", "Task", "Sync", "Inspection", "Role", "Filter"]
        )


# ---------------------------------------------------------------------------
# grade_documents — JSON per-doc grading  (NEW)
# ---------------------------------------------------------------------------

class TestGradeDocumentsJSON:

    @pytest.mark.asyncio
    async def test_returns_relevant_count(self, nodes):
        docs = [
            {"text": "Dashboard overview", "source": "a.md", "chunk_index": 0},
            {"text": "Unrelated content", "source": "b.md", "chunk_index": 0},
            {"text": "Dashboard tasks section", "source": "a.md", "chunk_index": 1},
        ]
        with patch("app.graph.nodes.client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = (
                '{"grades": ['
                '{"doc_index": 0, "relevant": true},'
                '{"doc_index": 1, "relevant": false},'
                '{"doc_index": 2, "relevant": true}'
                ']}'
            )
            mock_client.chat.completions.create.return_value = mock_response

            state = base_state(query="What is the Dashboard?")
            state["documents"] = docs
            result = await nodes.grade_documents(state)

        assert result["grade"] == "relevant"
        assert result["relevant_count"] == 2
        assert result["total_graded"] == 3
        assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_fallback_on_json_parse_error(self, nodes):
        """On JSON parse failure, all docs pass through (false negative prevention)."""
        docs = [
            {"text": "Some content", "source": "a.md", "chunk_index": 0},
        ]
        with patch("app.graph.nodes.client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "yes"  # not valid JSON
            mock_client.chat.completions.create.return_value = mock_response

            state = base_state(query="What is the Dashboard?")
            state["documents"] = docs
            result = await nodes.grade_documents(state)

        # Fallback: pass all docs through to avoid false negatives
        assert result["grade"] == "relevant"
        assert len(result["documents"]) == 1


# ---------------------------------------------------------------------------
# rewrite_query — 3 strategies  (NEW)
# ---------------------------------------------------------------------------

class TestRewriteQueryStrategies:

    @pytest.mark.asyncio
    async def test_strategy_0_terminology_alignment(self, nodes):
        """Strategy 0: MCL terminology rewrite, no alpha override."""
        with patch("app.graph.nodes.client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "How do I create a Routine Inspection?"
            mock_client.chat.completions.create.return_value = mock_response

            result = await nodes.rewrite_query(base_state(retry_count=0))

        assert result["retry_count"] == 1
        assert result.get("search_alpha_override") is None

    @pytest.mark.asyncio
    async def test_strategy_1_synonym_expansion_alpha_08(self, nodes):
        """Strategy 1: broad synonym expansion, alpha=0.8."""
        with patch("app.graph.nodes.client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Dashboard OR admin panel OR management view"
            mock_client.chat.completions.create.return_value = mock_response

            result = await nodes.rewrite_query(base_state(retry_count=1))

        assert result["retry_count"] == 2
        assert result.get("search_alpha_override") == 0.8

    @pytest.mark.asyncio
    async def test_strategy_2_decomposition_alpha_02(self, nodes):
        """Strategy 2: query decomposition into sub-questions, alpha=0.2."""
        with patch("app.graph.nodes.client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Dashboard navigation | Dashboard sections"
            mock_client.chat.completions.create.return_value = mock_response

            result = await nodes.rewrite_query(base_state(retry_count=2))

        assert result["retry_count"] == 3
        assert result.get("search_alpha_override") == 0.2

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
        response = mock_openai_response("no")
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.return_value = response
            state = base_state(documents=[{"text": "Unrelated content."}])
            result = await nodes.grade_documents(state)
        assert result["grade"] == "irrelevant"

    @pytest.mark.asyncio
    async def test_grades_irrelevant_on_api_error(self, nodes):
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("timeout")
            state = base_state(documents=[{"text": "some doc"}])
            result = await nodes.grade_documents(state)
        assert result["grade"] == "irrelevant"

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
        assert "cannot find" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_message_on_api_failure(self, nodes):
        docs = [{"text": "some context", "source": "s", "header_path": "H", "document_name": "d", "chunk_index": 0}]
        with patch("app.graph.nodes.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("api error")
            result = await nodes.generate_answer(base_state(documents=docs))
        assert "error" in result["answer"].lower()

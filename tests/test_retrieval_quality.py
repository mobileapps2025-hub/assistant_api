"""
Retrieval quality tests — validate threshold tuning, soft-scoring tiers,
rerank pass/fail behaviour, and the alpha-override mechanism.

No live network calls — Weaviate and Cohere are mocked.

Run with:
    python -m pytest tests/test_retrieval_quality.py -v
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.graph.nodes import AgentNodes
from app.core.state import AgentState
from app.core.config import RERANK_THRESHOLD, RERANK_HIGH_CONFIDENCE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_nodes() -> AgentNodes:
    mock_vs = MagicMock()
    mock_vs.hybrid_search.return_value = []
    mock_ls = MagicMock()
    mock_ls.detect_language.return_value = "en"
    nodes = AgentNodes(mock_vs, mock_ls)
    return nodes


def _make_state(query: str = "What is the Dashboard?", **overrides) -> AgentState:
    base = {
        "query": query,
        "documents": [],
        "language": "en",
        "messages": [],
        "answer": None,
        "grade": None,
        "error": None,
        "retry_count": 0,
        "contextualized_query": query,
        "relevant_count": None,
        "total_graded": None,
        "search_alpha_override": None,
    }
    base.update(overrides)
    return base


def _build_rerank_hit(index: int, score: float):
    hit = MagicMock()
    hit.index = index
    hit.relevance_score = score
    return hit


def _build_search_result(text: str = "some text", source: str = "doc.md") -> dict:
    return {"text": text, "source": source, "header_path": "Root", "chunk_index": 0}


# ── Config sanity ─────────────────────────────────────────────────────────────

def test_rerank_threshold_below_high_confidence():
    """RERANK_THRESHOLD < RERANK_HIGH_CONFIDENCE — otherwise soft-tier logic is broken."""
    assert RERANK_THRESHOLD < RERANK_HIGH_CONFIDENCE, (
        f"RERANK_THRESHOLD ({RERANK_THRESHOLD}) must be lower than "
        f"RERANK_HIGH_CONFIDENCE ({RERANK_HIGH_CONFIDENCE})"
    )


def test_rerank_threshold_is_permissive():
    """Threshold should be ≤ 0.2 to reduce false negatives."""
    assert RERANK_THRESHOLD <= 0.2, (
        f"RERANK_THRESHOLD ({RERANK_THRESHOLD}) is too high — "
        "consider lowering it to reduce false negatives."
    )


# ── Retrieve documents ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_uses_alpha_override_from_state():
    """Alpha from state.search_alpha_override is forwarded to hybrid_search."""
    nodes = _make_nodes()
    nodes.cohere_client = None

    state = _make_state(search_alpha_override=0.8)

    with patch.object(nodes.vector_store, "hybrid_search", return_value=[]) as mock_search:
        await nodes.retrieve_documents(state)
        assert mock_search.call_count == 2
        for call in mock_search.call_args_list:
            call_kwargs = call.kwargs
            assert call_kwargs.get("alpha") == 0.8, (
                f"Alpha override was not passed to hybrid_search. Got: {call_kwargs}"
            )


@pytest.mark.asyncio
async def test_retrieve_clears_alpha_override_after_use():
    """After retrieval, search_alpha_override should be reset to None."""
    nodes = _make_nodes()
    nodes.cohere_client = None
    nodes.vector_store.hybrid_search.return_value = []

    state = _make_state(search_alpha_override=0.8)
    result = await nodes.retrieve_documents(state)

    assert result.get("search_alpha_override") is None


@pytest.mark.asyncio
async def test_retrieve_defaults_to_alpha_05_when_no_override():
    """When search_alpha_override is None, alpha defaults to 0.5."""
    nodes = _make_nodes()
    nodes.cohere_client = None

    state = _make_state(search_alpha_override=None)

    with patch.object(nodes.vector_store, "hybrid_search", return_value=[]) as mock_search:
        await nodes.retrieve_documents(state)
        call_kwargs = mock_search.call_args.kwargs
        alpha_used = call_kwargs.get("alpha")
        assert alpha_used == 0.5, f"Expected default alpha=0.5, got {alpha_used}"


@pytest.mark.asyncio
async def test_rerank_drops_docs_below_threshold():
    """Documents with rerank score < RERANK_THRESHOLD must be excluded."""
    nodes = _make_nodes()
    initial = [
        _build_search_result("doc A"),
        _build_search_result("doc B"),
        _build_search_result("doc C"),
    ]
    nodes.vector_store.hybrid_search.return_value = initial

    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = MagicMock(
        results=[
            _build_rerank_hit(0, RERANK_THRESHOLD + 0.1),   # pass
            _build_rerank_hit(1, RERANK_THRESHOLD - 0.01),  # drop
            _build_rerank_hit(2, RERANK_THRESHOLD + 0.3),   # pass
        ]
    )
    nodes.cohere_client = mock_cohere

    state = _make_state()
    result = await nodes.retrieve_documents(state)

    assert len(result["documents"]) == 2, (
        f"Expected 2 docs after threshold filter, got {len(result['documents'])}"
    )


@pytest.mark.asyncio
async def test_rerank_assigns_high_confidence_tier():
    """Docs at or above RERANK_HIGH_CONFIDENCE must receive confidence_tier='high'."""
    nodes = _make_nodes()
    initial = [_build_search_result("doc A")]
    nodes.vector_store.hybrid_search.return_value = initial

    high_score = RERANK_HIGH_CONFIDENCE + 0.05
    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = MagicMock(
        results=[_build_rerank_hit(0, high_score)]
    )
    nodes.cohere_client = mock_cohere

    state = _make_state()
    result = await nodes.retrieve_documents(state)

    docs = result["documents"]
    assert docs, "No documents returned"
    assert docs[0].get("confidence_tier") == "high", (
        f"Expected confidence_tier='high' for score {high_score}, "
        f"got {docs[0].get('confidence_tier')}"
    )


@pytest.mark.asyncio
async def test_rerank_assigns_medium_confidence_tier():
    """Docs between RERANK_THRESHOLD and RERANK_HIGH_CONFIDENCE get tier='medium'."""
    nodes = _make_nodes()
    initial = [_build_search_result("doc A")]
    nodes.vector_store.hybrid_search.return_value = initial

    medium_score = (RERANK_THRESHOLD + RERANK_HIGH_CONFIDENCE) / 2
    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = MagicMock(
        results=[_build_rerank_hit(0, medium_score)]
    )
    nodes.cohere_client = mock_cohere

    state = _make_state()
    result = await nodes.retrieve_documents(state)

    docs = result["documents"]
    assert docs, "No documents returned"
    assert docs[0].get("confidence_tier") == "medium", (
        f"Expected confidence_tier='medium' for score {medium_score}, "
        f"got {docs[0].get('confidence_tier')}"
    )


@pytest.mark.asyncio
async def test_retrieve_returns_empty_on_all_below_threshold():
    """When ALL reranked docs fall below threshold, documents list should be empty."""
    nodes = _make_nodes()
    initial = [_build_search_result("doc A"), _build_search_result("doc B")]
    nodes.vector_store.hybrid_search.return_value = initial

    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = MagicMock(
        results=[
            _build_rerank_hit(0, RERANK_THRESHOLD - 0.05),
            _build_rerank_hit(1, RERANK_THRESHOLD - 0.08),
        ]
    )
    nodes.cohere_client = mock_cohere

    state = _make_state()
    result = await nodes.retrieve_documents(state)

    assert result["documents"] == [], (
        "Expected empty list when all docs are below threshold"
    )


# ── Grade documents ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_grade_documents_per_doc_json():
    """grade_documents uses JSON mode and individual doc grading."""
    docs = [
        {"text": "Dashboard overview", "source": "platform_notes.md", "chunk_index": 0},
        {"text": "Unrelated content about cats", "source": "other.md", "chunk_index": 0},
    ]
    state = _make_state(query="What is the Dashboard?")
    state["documents"] = docs

    with patch("app.graph.nodes.client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json_grades = (
            '{"grades": [{"doc_index": 0, "relevant": true}, '
            '{"doc_index": 1, "relevant": false}]}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        nodes = _make_nodes()
        result = await nodes.grade_documents(state)

        assert result["grade"] == "relevant"
        assert result["relevant_count"] == 1
        assert result["total_graded"] == 2
        # Only doc 0 should remain
        assert len(result["documents"]) == 1
        assert result["documents"][0]["text"] == "Dashboard overview"


@pytest.mark.asyncio
async def test_grade_documents_all_irrelevant():
    """When all docs grade irrelevant, grade='irrelevant' and docs=[]."""
    docs = [
        {"text": "Content about cats", "source": "other.md", "chunk_index": 0},
    ]
    state = _make_state(query="What is the Dashboard?")
    state["documents"] = docs

    with patch("app.graph.nodes.client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"grades": [{"doc_index": 0, "relevant": false}]}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        nodes = _make_nodes()
        result = await nodes.grade_documents(state)

        assert result["grade"] == "irrelevant"
        assert result["relevant_count"] == 0
        assert result["documents"] == []


@pytest.mark.asyncio
async def test_grade_documents_empty_input():
    """No documents → irrelevant without calling GPT."""
    state = _make_state()
    state["documents"] = []

    with patch("app.graph.nodes.client") as mock_client:
        nodes = _make_nodes()
        result = await nodes.grade_documents(state)

        assert result["grade"] == "irrelevant"
        mock_client.chat.completions.create.assert_not_called()


# ── Rewrite query strategies ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rewrite_strategy_0_no_alpha_override():
    """Strategy 0 (retry_count=0) must NOT set search_alpha_override."""
    state = _make_state(retry_count=0)

    with patch("app.graph.nodes.client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Better question about Dashboard"
        mock_client.chat.completions.create.return_value = mock_response

        nodes = _make_nodes()
        result = await nodes.rewrite_query(state)

        assert result.get("retry_count") == 1
        assert result.get("search_alpha_override") is None


@pytest.mark.asyncio
async def test_rewrite_strategy_1_sets_alpha_08():
    """Strategy 1 (retry_count=1) must set search_alpha_override=0.8."""
    state = _make_state(retry_count=1)

    with patch("app.graph.nodes.client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Dashboard OR admin panel OR web interface"
        mock_client.chat.completions.create.return_value = mock_response

        nodes = _make_nodes()
        result = await nodes.rewrite_query(state)

        assert result.get("retry_count") == 2
        assert result.get("search_alpha_override") == 0.8


@pytest.mark.asyncio
async def test_rewrite_strategy_2_sets_alpha_02():
    """Strategy 2 (retry_count=2) must set search_alpha_override=0.2."""
    state = _make_state(retry_count=2)

    with patch("app.graph.nodes.client") as mock_client:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Dashboard navigation | Dashboard features"
        mock_client.chat.completions.create.return_value = mock_response

        nodes = _make_nodes()
        result = await nodes.rewrite_query(state)

        assert result.get("retry_count") == 3
        assert result.get("search_alpha_override") == 0.2

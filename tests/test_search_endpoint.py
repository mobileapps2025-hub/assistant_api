from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.dependencies import get_vector_store_service
from app.main import app


def test_search_endpoint_uses_hybrid_search_and_returns_metadata():
    mock_vector_store = MagicMock()
    mock_vector_store.hybrid_search.return_value = [
        {
            "text": "x" * 600,
            "source": "checklist_wizard.md",
            "source_title": "MCL Checklist Wizard",
            "header_path": "Creating > Basics",
            "doc_type": "faq",
            "chunk_index": 2,
            "score": 0.91,
            "uuid": "abc-123",
        }
    ]
    app.dependency_overrides[get_vector_store_service] = lambda: mock_vector_store

    try:
        client = TestClient(app)
        response = client.post(
            "/api/search",
            json={"query": " create checklist ", "max_results": "3"},
        )
    finally:
        app.dependency_overrides = {}

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "create checklist"
    assert body["total_results"] == 1
    mock_vector_store.hybrid_search.assert_called_once_with(
        "create checklist",
        limit=3,
    )

    result = body["results"][0]
    assert result == {
        "source": "checklist_wizard.md",
        "source_title": "MCL Checklist Wizard",
        "header_path": "Creating > Basics",
        "doc_type": "faq",
        "chunk_index": 2,
        "score": 0.91,
        "uuid": "abc-123",
        "content_preview": "x" * 500,
    }


def test_search_endpoint_falls_back_to_source_title_from_source():
    mock_vector_store = MagicMock()
    mock_vector_store.hybrid_search.return_value = [
        {
            "text": "legacy content",
            "source": "legacy.md",
            "header_path": "Root",
            "doc_type": "faq",
            "chunk_index": 0,
            "score": 0.5,
            "uuid": "legacy-1",
        }
    ]
    app.dependency_overrides[get_vector_store_service] = lambda: mock_vector_store

    try:
        client = TestClient(app)
        response = client.post("/api/search", json={"query": "legacy"})
    finally:
        app.dependency_overrides = {}

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["source_title"] == "legacy.md"

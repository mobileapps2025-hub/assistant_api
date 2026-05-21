import pytest
from unittest.mock import MagicMock, patch

from app.graph.nodes import AgentNodes


def _nodes() -> AgentNodes:
    return AgentNodes(MagicMock(), MagicMock(), cohere_client=None)


def _mock_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def _state(**overrides):
    state = {
        "query": "How do roles and permissions work?",
        "messages": [],
        "documents": [],
        "visual_aids": [],
        "retry_count": 0,
        "language": "en",
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_sources_footer_prefers_source_title_when_present():
    nodes = _nodes()
    docs = [
        {
            "text": "Roles and permissions are managed in the Dashboard.",
            "source": "roles_permissions.md",
            "source_title": "Roles and Permissions",
            "document_name": "legacy-doc-name.md",
            "header_path": "Roles",
            "chunk_index": 1,
        }
    ]

    with patch("app.graph.nodes.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "Roles and permissions are managed in the Dashboard [Source: roles_permissions.md]."
        )
        result = await nodes.generate_answer(_state(documents=docs))

    footer = result["answer"].split("**Sources:**")[-1]
    assert "Roles and Permissions (Chunk 2)" in footer
    assert "legacy-doc-name.md" not in footer
    assert "Doc" not in footer


@pytest.mark.asyncio
async def test_sources_footer_falls_back_to_source_not_doc_label():
    nodes = _nodes()
    docs = [
        {
            "text": "Assigned tasks appear in the mobile app.",
            "source": "mobile_tasks.md",
            "document_name": "Doc",
            "header_path": "Tasks",
            "chunk_index": 0,
        }
    ]

    with patch("app.graph.nodes.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "Assigned tasks appear in the mobile app [Source: mobile_tasks.md]."
        )
        result = await nodes.generate_answer(_state(query="Where do I receive tasks?", documents=docs))

    footer = result["answer"].split("**Sources:**")[-1]
    assert "mobile_tasks.md (Chunk 1)" in footer
    assert "Doc (Chunk 1)" not in footer


def test_grounding_accepts_source_title_citations():
    nodes = _nodes()
    docs = [{"source": "roles_permissions.md", "source_title": "Roles and Permissions"}]
    answer = "Use the Roles area for permission details [Source: Roles and Permissions]."

    result = nodes._validate_grounding(answer, docs)

    assert result == answer

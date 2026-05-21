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
        "query": "Where is the main navigation menu?",
        "messages": [],
        "documents": [],
        "visual_aids": [],
        "retry_count": 0,
        "language": "en",
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_visual_claim_removed_when_no_image_markdown_available():
    nodes = _nodes()
    docs = [
        {
            "text": "Open tasks from the Tasks menu.",
            "source": "tasks.md",
            "header_path": "Tasks",
            "chunk_index": 0,
        }
    ]
    visual_aids = [
        {
            "text": "Main navigation notes without an image markdown link.",
            "source": "main-navigation-menu.md",
            "header_path": "Main Navigation Menu",
            "chunk_index": 0,
        }
    ]

    with patch("app.graph.nodes.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "Open tasks from the **Tasks** menu [Source: tasks.md].\n"
            "See the visual guide below."
        )
        result = await nodes.generate_answer(_state(documents=docs, visual_aids=visual_aids))

    assert "Open tasks" in result["answer"]
    assert "visual guide" not in result["answer"].lower()


@pytest.mark.asyncio
async def test_visual_image_gets_question_tied_caption():
    nodes = _nodes()
    docs = [
        {
            "text": "The main navigation menu opens app sections.",
            "source": "navigation.md",
            "header_path": "Navigation",
            "chunk_index": 0,
        }
    ]
    image = "![Main navigation menu](images/main-navigation-menu.png)"
    visual_aids = [
        {
            "text": f"This screenshot shows the main navigation menu.\n{image}",
            "source": "main-navigation-menu.md",
            "header_path": "Main Navigation Menu",
            "chunk_index": 0,
        }
    ]

    with patch("app.graph.nodes.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            f"Use the main navigation menu to open app sections [Source: navigation.md].\n{image}"
        )
        result = await nodes.generate_answer(
            _state(query="Where is the main navigation menu?", documents=docs, visual_aids=visual_aids)
        )

    lines = result["answer"].splitlines()
    image_index = lines.index(image)
    caption = lines[image_index - 1]
    assert caption.startswith("*Visual aid:")
    assert "Main Navigation Menu" in caption
    assert "main navigation menu" in caption.lower()


@pytest.mark.asyncio
async def test_invented_visual_image_markdown_is_removed():
    nodes = _nodes()
    docs = [
        {
            "text": "The main navigation menu opens app sections.",
            "source": "navigation.md",
            "header_path": "Navigation",
            "chunk_index": 0,
        }
    ]
    visual_aids = [
        {
            "text": "Allowed image:\n![Real menu](images/real-menu.png)",
            "source": "main-navigation-menu.md",
            "header_path": "Main Navigation Menu",
            "chunk_index": 0,
        }
    ]

    with patch("app.graph.nodes.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "Use the main navigation menu [Source: navigation.md].\n"
            "![Fake menu](images/fake-menu.png)"
        )
        result = await nodes.generate_answer(_state(documents=docs, visual_aids=visual_aids))

    assert "images/fake-menu.png" not in result["answer"]

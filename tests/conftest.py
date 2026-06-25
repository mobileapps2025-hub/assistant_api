"""Shared pytest fixtures for the MCL Assistant test suite."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService


@pytest.fixture
def mock_vision_service():
    return MagicMock(spec=VisionService)


@pytest.fixture
def mock_image_validator():
    mock = MagicMock(spec=ImageValidatorService)
    mock.validate_image.return_value = {"is_mcl": True, "confidence": 0.95}
    return mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with a configurable chat completion response."""
    mock = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = "Mock answer from GPT-4o."
    mock.chat.completions.create.return_value = response
    return mock


@pytest.fixture
def sample_documents():
    """A small set of document chunks for reuse in tests."""
    return [
        {
            "text": "To create a checklist, tap the + button.",
            "source": "mcl_guide.md",
            "header_path": "Checklists > Creating",
            "chunk_index": 0,
            "document_name": "mcl_guide.md",
            "score": 0.85,
        },
        {
            "text": "Sync issues occur when there is no internet connection.",
            "source": "mcl_guide.md",
            "header_path": "Troubleshooting > Sync",
            "chunk_index": 1,
            "document_name": "mcl_guide.md",
            "score": 0.78,
        },
    ]

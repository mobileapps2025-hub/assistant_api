import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.services.chat_service import ChatService, _infer_graph_path
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.services.language_service import LanguageService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_service(mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service):
    """Construct a ChatService with all external dependencies mocked."""
    with patch("app.services.chat_service.COHERE_API_KEY", ""):
        return ChatService(
            mock_vector_store,
            mock_vision_service,
            mock_image_validator,
            mock_language_service,
        )


# ---------------------------------------------------------------------------
# _infer_graph_path (pure function — no mocking needed)
# ---------------------------------------------------------------------------

class TestInferGraphPath:
    def test_clarify_path(self):
        result = _infer_graph_path({"grade": "irrelevant", "retry_count": 0})
        assert result == "retrieve→grade→clarify"

    def test_rewrite_path(self):
        result = _infer_graph_path({"grade": "relevant", "retry_count": 1})
        assert result == "retrieve→grade→rewrite→retrieve→generate"

    def test_direct_generate_path(self):
        result = _infer_graph_path({"grade": "relevant", "retry_count": 0})
        assert result == "retrieve→grade→generate"

    def test_missing_keys_defaults_to_generate(self):
        result = _infer_graph_path({})
        assert result == "retrieve→grade→generate"


# ---------------------------------------------------------------------------
# ChatService construction
# ---------------------------------------------------------------------------

class TestChatServiceInit:
    def test_cohere_disabled_when_no_key(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        service = make_service(
            mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
        )
        assert service.cohere_client is None

    def test_cohere_enabled_when_key_present(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        with patch("app.services.chat_service.COHERE_API_KEY", "dummy-key"):
            with patch("app.services.chat_service.cohere.Client") as mock_cohere_cls:
                service = ChatService(
                    mock_vector_store,
                    mock_vision_service,
                    mock_image_validator,
                    mock_language_service,
                )
        mock_cohere_cls.assert_called_once_with("dummy-key")
        assert service.cohere_client is mock_cohere_cls.return_value


# ---------------------------------------------------------------------------
# process_chat_request routing
# ---------------------------------------------------------------------------

class TestProcessChatRequestRouting:
    """Verify vision vs. text routing based on message content."""

    @pytest.mark.asyncio
    async def test_routes_to_vision_when_image_present(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        service = make_service(
            mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
        )
        service._handle_vision_request = AsyncMock(
            return_value={"response": "vision answer", "success": True, "has_vision": True}
        )
        service._handle_text_request = AsyncMock()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        result = await service.process_chat_request(messages)

        service._handle_vision_request.assert_called_once()
        service._handle_text_request.assert_not_called()
        assert result["has_vision"] is True

    @pytest.mark.asyncio
    async def test_routes_to_text_when_no_image(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        service = make_service(
            mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
        )
        service._handle_vision_request = AsyncMock()
        service._handle_text_request = AsyncMock(
            return_value={"response": "text answer", "success": True, "has_vision": False}
        )

        messages = [{"role": "user", "content": "How do I sync?"}]
        result = await service.process_chat_request(messages)

        service._handle_text_request.assert_called_once()
        service._handle_vision_request.assert_not_called()
        assert result["has_vision"] is False

    @pytest.mark.asyncio
    async def test_returns_error_when_no_user_message(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        service = make_service(
            mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
        )
        result = await service.process_chat_request([{"role": "assistant", "content": "Hi"}])

        assert result["success"] is False
        assert "No user message" in result["response"]


# ---------------------------------------------------------------------------
# _handle_text_request (mocked workflow)
# ---------------------------------------------------------------------------

class TestHandleTextRequest:
    @pytest.mark.asyncio
    async def test_returns_answer_from_graph(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        service = make_service(
            mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
        )
        service.workflow = MagicMock()
        service.workflow.ainvoke = AsyncMock(
            return_value={"answer": "Here is the answer.", "grade": "relevant", "retry_count": 0}
        )
        service._get_curated_knowledge = AsyncMock(return_value=[])

        messages = [{"role": "user", "content": "How do I create a task?"}]
        latest = messages[0]
        result = await service._handle_text_request(messages, latest, None)

        assert result["success"] is True
        assert result["response"] == "Here is the answer."
        assert result["has_vision"] is False

    @pytest.mark.asyncio
    async def test_returns_error_on_graph_exception(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        service = make_service(
            mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
        )
        service.workflow = MagicMock()
        service.workflow.ainvoke = AsyncMock(side_effect=RuntimeError("graph failure"))
        service._get_curated_knowledge = AsyncMock(return_value=[])

        messages = [{"role": "user", "content": "test"}]
        result = await service._handle_text_request(messages, messages[0], None)

        assert result["success"] is False
        assert "error" in result["response"].lower()


# ---------------------------------------------------------------------------
# _get_curated_knowledge keyword filtering
# ---------------------------------------------------------------------------

class TestGetCuratedKnowledge:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_db(
        self, mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
    ):
        with patch("app.services.chat_service.AsyncSessionLocal", None):
            service = make_service(
                mock_vector_store, mock_vision_service, mock_image_validator, mock_language_service
            )
            result = await service._get_curated_knowledge("any query")
        assert result == []

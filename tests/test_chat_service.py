import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.chat_service import ChatService


def make_service(mock_vision_service, mock_image_validator):
    """Construct a ChatService with all external dependencies mocked."""
    return ChatService(mock_vision_service, mock_image_validator)


# ---------------------------------------------------------------------------
# process_chat_request routing
# ---------------------------------------------------------------------------

class TestProcessChatRequestRouting:
    """Verify vision vs. text routing based on message content."""

    @pytest.mark.asyncio
    async def test_image_message_goes_through_router_not_forked(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        service._handle_text_request = AsyncMock(
            return_value={"response": "screen help", "success": True, "has_vision": True}
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        with patch("app.services.chat_service.classify_route") as mock_route:
            mock_route.return_value.route = "KNOWLEDGE"
            mock_route.return_value.reason = "screen help"
            result = await service.process_chat_request(messages)

        mock_route.assert_called_once()  # the image went THROUGH the router, not around it
        service._handle_text_request.assert_called_once()
        assert result["has_vision"] is True

    @pytest.mark.asyncio
    async def test_routes_to_text_when_no_image(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        service._handle_vision_request = AsyncMock()
        service._handle_text_request = AsyncMock(
            return_value={"response": "text answer", "success": True, "has_vision": False}
        )

        messages = [{"role": "user", "content": "How do I sync?"}]
        with patch("app.services.chat_service.classify_route") as mock_route:
            mock_route.return_value.route = "KNOWLEDGE"
            mock_route.return_value.reason = "general how-to"
            result = await service.process_chat_request(messages)

        service._handle_text_request.assert_called_once()
        service._handle_vision_request.assert_not_called()
        assert result["has_vision"] is False

    @pytest.mark.asyncio
    async def test_returns_error_when_no_user_message(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        result = await service.process_chat_request([{"role": "assistant", "content": "Hi"}])

        assert result["success"] is False
        assert "No user message" in result["response"]


# ---------------------------------------------------------------------------
# _handle_text_request (KNOWLEDGE path → Ragie retrieval, mocked)
# ---------------------------------------------------------------------------

class TestHandleTextRequest:
    @pytest.mark.asyncio
    async def test_returns_answer_from_retrieval(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "How do I create a task?"}]
        with patch(
            "app.services.chat_service.run_retrieval",
            return_value={"answer": "Here is the answer.", "sources": []},
        ):
            result = await service._handle_text_request(messages, messages[0], None)

        assert result["success"] is True
        assert result["response"] == "Here is the answer."
        assert result["has_vision"] is False

    @pytest.mark.asyncio
    async def test_returns_error_on_retrieval_exception(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "test"}]
        with patch(
            "app.services.chat_service.run_retrieval", side_effect=RuntimeError("retrieval failure")
        ):
            result = await service._handle_text_request(messages, messages[0], None)

        assert result["success"] is False
        assert "error" in result["response"].lower()


# ---------------------------------------------------------------------------
# KNOWLEDGE + image → _answer_over_image (Decision 12 — Layer 1 + retrieval)
# ---------------------------------------------------------------------------

class TestAnswerOverImage:
    @pytest.mark.asyncio
    async def test_image_knowledge_uses_instruction_and_retrieval(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        chunk = types.SimpleNamespace(document_name="guide.pdf", text="Tap the + button.")
        latest = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this screen?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ],
        }
        with patch("app.services.chat_service.build_vision_query", return_value="checklist wizard departments") as mock_bq, \
             patch("app.services.chat_service.retrieve", return_value=[chunk]) as mock_retrieve, \
             patch("app.services.chat_service.client") as mock_client:
            resp = MagicMock()
            resp.choices[0].message.content = "This is the Checklist Wizard [Source: guide.pdf]."
            mock_client.chat.completions.create.return_value = resp
            # entry via the KNOWLEDGE handler, which now branches to the image path
            result = await service._handle_text_request([latest], latest, None, "")
            sent = mock_client.chat.completions.create.call_args.kwargs["messages"]

        assert result["success"] is True and result["has_vision"] is True
        mock_bq.assert_called_once()
        mock_retrieve.assert_called_once_with("checklist wizard departments")
        assert "MCL Support Specialist" in sent[0]["content"]
        assert any(m["role"] == "system" and "# TEXTUAL CONTEXT" in m["content"] for m in sent)
        assert "[Source: guide.pdf]" in result["response"]

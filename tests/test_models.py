import pytest
from pydantic import ValidationError
from app.models import (
    ChatRequest,
    ChatResponse,
    ContentItem,
    FeedbackRequest,
    Message,
    generate_response_id,
)


class TestContentItem:
    def test_text_item(self):
        item = ContentItem(type="text", text="Hello")
        assert item.type == "text"
        assert item.text == "Hello"
        assert item.image_url is None

    def test_image_url_item(self):
        item = ContentItem(type="image_url", image_url={"url": "data:image/png;base64,abc"})
        assert item.type == "image_url"
        assert item.image_url == {"url": "data:image/png;base64,abc"}


class TestMessage:
    def test_simple_text_message(self):
        msg = Message(role="user", content="How do I sync?")
        assert msg.role == "user"
        assert msg.content == "How do I sync?"

    def test_multimodal_message(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image_url"

    def test_assistant_message(self):
        msg = Message(role="assistant", content="Here is the answer.")
        assert msg.role == "assistant"


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hello"}])
        assert len(req.messages) == 1

    def test_empty_messages_allowed(self):
        req = ChatRequest(messages=[])
        assert req.messages == []

    def test_missing_messages_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:
    def test_valid_response(self):
        resp = ChatResponse(response="answer", response_id="resp_abc123", sources=[])
        assert resp.response == "answer"
        assert resp.response_id == "resp_abc123"
        assert resp.sources == []

    def test_sources_optional(self):
        resp = ChatResponse(response="answer", response_id="resp_abc123")
        assert resp.sources is None


class TestFeedbackRequest:
    def test_positive_feedback(self):
        fb = FeedbackRequest(response_id="resp_abc", feedback_type="positive")
        assert fb.feedback_type == "positive"
        assert fb.user_comment is None

    def test_negative_feedback_with_comment(self):
        fb = FeedbackRequest(
            response_id="resp_abc",
            feedback_type="negative",
            user_comment="Wrong answer",
        )
        assert fb.user_comment == "Wrong answer"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(feedback_type="positive")


class TestGenerateResponseId:
    def test_starts_with_resp(self):
        rid = generate_response_id()
        assert rid.startswith("resp_")

    def test_unique_ids(self):
        ids = {generate_response_id() for _ in range(50)}
        assert len(ids) == 50

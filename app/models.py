from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


class ContentItem(BaseModel):
    """Content item for multimodal messages (text or image)."""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # {"url": "data:image/png;base64,..."}

    class Config:
        json_schema_extra = {
            "examples": [
                {"type": "text", "text": "What is this screen?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
            ]
        }


class Message(BaseModel):
    """Message in a chat conversation, supporting text and images."""
    role: str  # "user", "assistant", "system"
    content: Optional[List[ContentItem] | str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    annotations: Optional[str] = None

    class Config:
        json_schema_extra = {
            "examples": [
                {"role": "user", "content": "Hello, how can I create a checklist?"},
                {"role": "user", "content": [
                    {"type": "text", "text": "What can I do on this screen?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]}
            ]
        }


class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Schema for receiving feedback from frontend."""
    response_id: str
    feedback_type: str  # 'positive' or 'negative'
    user_comment: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "resp_12345",
                "feedback_type": "positive",
                "user_comment": "Very helpful answer!"
            }
        }


class FeedbackResponse(BaseModel):
    """Schema for feedback response."""
    id: int
    response_id: str
    feedback_type: str
    user_comment: Optional[str]
    created_at: datetime
    processed: bool

    class Config:
        from_attributes = True


class CuratedQaRequest(BaseModel):
    """Schema for creating curated Q&A pairs."""
    question: str
    answer: str
    source_feedback_id: Optional[int] = None


class CuratedQaResponse(BaseModel):
    """Schema for curated Q&A response."""
    id: int
    question: str
    answer: str
    source_feedback_id: Optional[int]
    created_at: datetime
    active: bool

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    """Chat response with tracking ID."""
    response: str
    response_id: str
    sources: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Here is your answer...",
                "response_id": "resp_12345",
                "sources": []
            }
        }


def generate_response_id() -> str:
    """Generate a unique response ID for tracking."""
    return f"resp_{uuid.uuid4().hex[:8]}"

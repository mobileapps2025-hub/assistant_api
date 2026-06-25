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


class AuthContext(BaseModel):
    access_token: Optional[str] = None
    user_id: Optional[str] = None
    company_id: Optional[str] = None
    company_name: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None


class SessionRequest(BaseModel):
    """Request to establish a session from a token shared by the MCL app."""
    access_token: str


class SessionResponse(BaseModel):
    """Resolved session identity, derived from the shared token via UserInfo."""
    access_token: str
    user_id: str
    company_id: str
    company_name: str
    full_name: str
    email: str


class MarketInfo(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    soll_bestand: Optional[float] = None
    kasseneinsaetze: Optional[float] = None
    summe_kasseneinsatze: Optional[float] = None


class UserMarketsResponse(BaseModel):
    markets: List[MarketInfo]
    total: int


class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None
    auth_context: Optional[AuthContext] = None


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


class MemorySaveRequest(BaseModel):
    messages: List[Dict[str, Any]]
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class MemoryInfo(BaseModel):
    id: str
    title: str = ""
    category: str = ""
    importance: str = "low"
    tags: List[str] = []
    content: str = ""
    created: str = ""
    updated: str = ""


class MemoryListResponse(BaseModel):
    memories: List[MemoryInfo]


class MemorySaveResponse(BaseModel):
    saved: List[MemoryInfo] = []
    updated: List[MemoryInfo] = []
    deleted: List[str] = []


class MemoryUpdateRequest(BaseModel):
    content: str


class MemoryRecallResponse(BaseModel):
    context: str = ""
    memories: List[MemoryInfo] = []


class MemoryStoreRequest(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    user_id: Optional[str] = None

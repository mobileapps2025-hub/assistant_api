from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

# --- Unified Pydantic Models for Multi-Agent Assistant ---

class ContentItem(BaseModel):
    text: str
    type: str

class Message(BaseModel):
    role: str
    content: Optional[List[ContentItem] | str] = None
    tool_call_id: Optional[str] = None 
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None 
    annotations: Optional[str] = None  # Fixed typo

class ChatRequest(BaseModel):
    messages: List[Message]
    app_type: Optional[str] = None  # "spotplan" or "mcl" to distinguish between apps

class MCLQuery(BaseModel):
    """Model for MCL-specific queries"""
    question: str
    context: Optional[str] = None
    category: Optional[str] = None  # e.g., "usage", "troubleshooting", "features"

class SpotplanQuery(BaseModel):
    """Model for Spotplan-specific queries"""
    question: str
    store_id: Optional[str] = None
    year: Optional[int] = None
    week: Optional[int] = None

# -- events request model --
class EventsBetweenWeeksRequest(BaseModel):
    storeId: str
    startingWeek: int
    endingWeek: int
    year: int

# --- MCL Feedback System Models ---

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
    """Enhanced chat response with tracking ID."""
    response: str
    response_id: str
    sources: Optional[List[str]] = None
    app_type: Optional[str] = None  # Track which app was used
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Here is your answer...",
                "response_id": "resp_12345",
                "sources": ["document1.pdf", "document2.pdf"],
                "app_type": "mcl"
            }
        }

def generate_response_id() -> str:
    """Generate a unique response ID for tracking."""
    return f"resp_{uuid.uuid4().hex[:8]}"

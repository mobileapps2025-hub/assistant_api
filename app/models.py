from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# --- Pydantic Models for MCL Assistant ---

class ContentItem(BaseModel):
    text: str
    type: str

class Message(BaseModel):
    role: str
    content: Optional[List[ContentItem] | str] = None
    tool_call_id: Optional[str] = None 
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None 
    annotations: Optional[str] = None 

class ChatRequest(BaseModel):
    messages: List[Message]

class MCLQuery(BaseModel):
    """Model for MCL-specific queries"""
    question: str
    context: Optional[str] = None
    category: Optional[str] = None  # e.g., "usage", "troubleshooting", "features"

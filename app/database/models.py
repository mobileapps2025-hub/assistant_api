"""
Database models for the MCL AI Assistant feedback system.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional, List
import uuid

Base = declarative_base()


class AIFeedback(Base):
    """Table for storing user feedback on AI responses."""
    __tablename__ = "AIFeedback"
    
    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(String(255), unique=True, nullable=False, index=True)
    user_question = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    feedback_type = Column(String(20), nullable=False)  # 'positive' or 'negative'
    user_comment = Column(Text, nullable=True)
    retrieved_chunks = Column(Text, nullable=True)  # JSON string of chunks used
    created_at = Column(DateTime, default=func.getdate())
    processed = Column(Boolean, default=False)
    
    # Relationship to curated Q&A
    curated_qa = relationship("AICuratedQa", back_populates="source_feedback")


class AICuratedQa(Base):
    """Table for storing curated Q&A pairs from positive feedback."""
    __tablename__ = "AICurated_Qa"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_feedback_id = Column(Integer, ForeignKey("AIFeedback.id"), nullable=True)
    created_at = Column(DateTime, default=func.getdate())
    active = Column(Boolean, default=True)
    
    # Relationship to feedback
    source_feedback = relationship("AIFeedback", back_populates="curated_qa")


# Pydantic models for API
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "MCL is a mobile application...",
                "response_id": "resp_12345",
                "sources": ["document1.pdf", "document2.pdf"]
            }
        }


def generate_response_id() -> str:
    """Generate a unique response ID for tracking."""
    return f"resp_{uuid.uuid4().hex[:8]}"
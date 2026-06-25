"""
Database models for MCL Assistant feedback system.
"""
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Feedback(Base):
    """Feedback table for storing user feedback on AI responses."""
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    response_id = Column(String(100), unique=True, nullable=False, index=True)
    feedback_type = Column(String(20), nullable=False)  # 'positive' or 'negative'
    user_comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, response_id='{self.response_id}', type='{self.feedback_type}')>"

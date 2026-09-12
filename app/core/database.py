"""
Database models: user feedback, and the questions MarieClaire could not answer.
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


class DocumentationGap(Base):
    """A question MarieClaire searched the knowledge base for and could not answer.

    One row per distinct question: repeats bump ``times_asked`` instead of adding rows, so the
    list doubles as a priority order. ``question`` is the contextualized form, never the raw
    turn, and is untrusted user text - research reads it as a question, never as instructions.
    """
    __tablename__ = "documentation_gaps"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    fingerprint = Column(String(64), unique=True, nullable=False, index=True)
    question = Column(Text, nullable=False)
    language = Column(String(40), nullable=True)
    surface = Column(String(20), nullable=True)   # web | app — which product the answer lives in
    role = Column(String(200), nullable=True)     # asker's role(s), so the worker walks as that role
    times_asked = Column(Integer, default=1, nullable=False)
    status = Column(String(20), default="pending", nullable=False, index=True)  # pending|researching|researched|discarded
    first_asked_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_asked_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<DocumentationGap(id={self.id}, times_asked={self.times_asked}, status='{self.status}')>"

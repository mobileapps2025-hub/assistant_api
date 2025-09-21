"""
Repository layer for database operations related to feedback and curated Q&A.
"""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import and_, desc
from .models import AIFeedback, AICuratedQa, FeedbackRequest, CuratedQaRequest
import json


class FeedbackRepository:
    """Repository for AIFeedback operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_feedback(
        self, 
        response_id: str,
        user_question: str,
        ai_response: str,
        feedback_type: str,
        user_comment: Optional[str] = None,
        retrieved_chunks: Optional[List[str]] = None
    ) -> AIFeedback:
        """Create a new feedback record."""
        
        # Convert chunks list to JSON string if provided
        chunks_json = json.dumps(retrieved_chunks) if retrieved_chunks else None
        
        feedback = AIFeedback(
            response_id=response_id,
            user_question=user_question,
            ai_response=ai_response,
            feedback_type=feedback_type,
            user_comment=user_comment,
            retrieved_chunks=chunks_json
        )
        
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback
    
    async def get_feedback_by_response_id(self, response_id: str) -> Optional[AIFeedback]:
        """Get feedback by response ID."""
        result = await self.db.execute(
            select(AIFeedback).where(AIFeedback.response_id == response_id)
        )
        return result.scalar_one_or_none()
    
    async def get_unprocessed_feedback(self, limit: int = 50) -> List[AIFeedback]:
        """Get unprocessed feedback for review."""
        result = await self.db.execute(
            select(AIFeedback)
            .where(AIFeedback.processed == False)
            .order_by(desc(AIFeedback.created_at))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_positive_feedback(self, limit: int = 100) -> List[AIFeedback]:
        """Get positive feedback for creating curated Q&A."""
        result = await self.db.execute(
            select(AIFeedback)
            .where(AIFeedback.feedback_type == 'positive')
            .order_by(desc(AIFeedback.created_at))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def mark_feedback_processed(self, feedback_id: int) -> bool:
        """Mark feedback as processed."""
        result = await self.db.execute(
            select(AIFeedback).where(AIFeedback.id == feedback_id)
        )
        feedback = result.scalar_one_or_none()
        
        if feedback:
            feedback.processed = True
            await self.db.commit()
            return True
        return False
    
    async def get_feedback_stats(self) -> dict:
        """Get feedback statistics."""
        # Count positive feedback
        positive_result = await self.db.execute(
            select(AIFeedback).where(AIFeedback.feedback_type == 'positive')
        )
        positive_count = len(positive_result.scalars().all())
        
        # Count negative feedback
        negative_result = await self.db.execute(
            select(AIFeedback).where(AIFeedback.feedback_type == 'negative')
        )
        negative_count = len(negative_result.scalars().all())
        
        # Count unprocessed feedback
        unprocessed_result = await self.db.execute(
            select(AIFeedback).where(AIFeedback.processed == False)
        )
        unprocessed_count = len(unprocessed_result.scalars().all())
        
        return {
            "total_feedback": positive_count + negative_count,
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "unprocessed_feedback": unprocessed_count,
            "satisfaction_rate": round((positive_count / (positive_count + negative_count)) * 100, 2) if (positive_count + negative_count) > 0 else 0
        }


class CuratedQaRepository:
    """Repository for AICuratedQa operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_curated_qa(
        self,
        question: str,
        answer: str,
        source_feedback_id: Optional[int] = None
    ) -> AICuratedQa:
        """Create a new curated Q&A pair."""
        curated_qa = AICuratedQa(
            question=question,
            answer=answer,
            source_feedback_id=source_feedback_id
        )
        
        self.db.add(curated_qa)
        await self.db.commit()
        await self.db.refresh(curated_qa)
        return curated_qa
    
    async def get_active_curated_qa(self) -> List[AICuratedQa]:
        """Get all active curated Q&A pairs."""
        result = await self.db.execute(
            select(AICuratedQa)
            .where(AICuratedQa.active == True)
            .order_by(desc(AICuratedQa.created_at))
        )
        return result.scalars().all()
    
    async def get_curated_qa_by_id(self, qa_id: int) -> Optional[AICuratedQa]:
        """Get curated Q&A by ID."""
        result = await self.db.execute(
            select(AICuratedQa).where(AICuratedQa.id == qa_id)
        )
        return result.scalar_one_or_none()
    
    async def deactivate_curated_qa(self, qa_id: int) -> bool:
        """Deactivate a curated Q&A pair."""
        result = await self.db.execute(
            select(AICuratedQa).where(AICuratedQa.id == qa_id)
        )
        curated_qa = result.scalar_one_or_none()
        
        if curated_qa:
            curated_qa.active = False
            await self.db.commit()
            return True
        return False
    
    async def search_curated_qa(self, query: str, limit: int = 10) -> List[AICuratedQa]:
        """Search curated Q&A pairs by question content."""
        result = await self.db.execute(
            select(AICuratedQa)
            .where(
                and_(
                    AICuratedQa.active == True,
                    AICuratedQa.question.ilike(f"%{query}%")
                )
            )
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_curated_qa_content_for_kb(self) -> str:
        """Get all active curated Q&A pairs formatted for knowledge base."""
        curated_qas = await self.get_active_curated_qa()
        
        if not curated_qas:
            return ""
        
        content = "# Curated Q&A Knowledge Base\n\n"
        content += "This document contains high-quality question-answer pairs derived from positive user feedback.\n\n"
        
        for qa in curated_qas:
            content += f"## Question: {qa.question}\n\n"
            content += f"**Answer:** {qa.answer}\n\n"
            content += "---\n\n"
        
        return content
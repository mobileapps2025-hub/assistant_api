import uvicorn
import uuid
from contextlib import asynccontextmanager
import tempfile
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import exc

from app.models import ChatRequest, Message, ContentItem, ChatResponse, generate_response_id, FeedbackRequest, FeedbackResponse
from app.core.config import ENABLE_MCL_IMAGE_VALIDATION, get_db, engine, VECTOR_STORE_PATH, CORS_ORIGINS, AsyncSessionLocal
from app.core.database import Feedback, Base
from app.core.dependencies import get_vector_store_service, get_chat_service
from app.services.chat_service import ChatService
from app.services.vector_store import VectorStoreService
from app.services.ingestion_service import IngestionService
from app.core.context import analyze_situational_context
from app.routers import vision, admin
from app.core.logging import setup_logging, get_logger, request_id_var

# Setup logging
setup_logging()
logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a short UUID to every request for end-to-end tracing."""

    async def dispatch(self, request: Request, call_next):
        req_id = uuid.uuid4().hex[:8]
        token = request_id_var.set(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_var.reset(token)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCL knowledge base and database on startup."""
    logger.info("MCL Assistant startup sequence initiated...")
    
    try:
        # Initialize database tables
        logger.info("Initializing database...")
        if engine:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized")
        else:
            logger.warning("Database not available - feedback system disabled")
        
        # Initialize MCL knowledge base
        logger.info("Initializing MCL knowledge base...")
        # Just initialize the service holder - specific connection happens lazily
        vector_store = get_vector_store_service()
        
        if vector_store.client:
            logger.info("Verifying Weaviate schema...")
            vector_store.ensure_schema()
            
            logger.info("Checking for documents to ingest...")
            stats = vector_store.get_stats()
            if stats.get("count", 0) == 0:
                logger.info("Vector store is empty. Triggering initial ingestion...")
                ingestion_service = IngestionService(vector_store)
                # In docs folder
                ingestion_service.ingest_all("app/documents")
            else:
                logger.info(f"Vector store already contains {stats['count']} documents. Skipping ingestion.")
        
        logger.info("MCL knowledge base service initialized.")

        logger.info("MCL Assistant startup completed.")
        yield
    except Exception as e:
        logger.error(f"FATAL ERROR during MCL Assistant startup: {e}", exc_info=True)
        yield
    finally:
        logger.info("MCL Assistant shutting down...")
        if engine:
            await engine.dispose()

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="MCL Assistant API", 
    description="AI-powered knowledge base assistant for the MCL (Mobile Checklist) application",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vision.router)
app.include_router(admin.router)

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "MCL Assistant API",
        "version": "3.0.0",
        "description": "AI-powered knowledge base assistant for the MCL (Mobile Checklist) application",
        "features": [
            "Multilingual support (German, English, Spanish, French, Italian)",
            "Advanced RAG with semantic search",
            "Query expansion for better results",
            "Support for PDF, DOCX, PPTX, and Markdown documents"
        ],
        "endpoints": {
            "chat": "/api/chat",
            "feedback": "/api/feedback",
            "health": "/health",
            "chunks": "/api/chunks",
            "search": "/api/search"
        }
    }

@app.get("/health")
async def health_check(vector_store: VectorStoreService = Depends(get_vector_store_service)):
    """Health check endpoint — reports real infrastructure status."""
    stats = vector_store.get_stats()
    weaviate_status = stats.get("status", "unknown")
    weaviate_healthy = weaviate_status == "connected"
    db_healthy = AsyncSessionLocal is not None

    overall = "healthy" if (weaviate_healthy and db_healthy) else "degraded"

    return {
        "status": overall,
        "weaviate": weaviate_status,
        "document_count": stats.get("count", 0),
        "database": "connected" if db_healthy else "unavailable",
    }

@app.get("/api/chunks")
async def get_chunks_info(vector_store: VectorStoreService = Depends(get_vector_store_service)):
    """Get information about available MCL document chunks."""
    stats = vector_store.get_stats()
    
    return {
        "message": "Detailed chunk listing is available via search endpoint",
        "stats": stats
    }

@app.post("/api/search")
async def search_chunks(
    query_data: dict,
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """Search for relevant MCL chunks based on a query."""
    query = query_data.get("query", "")
    max_results = query_data.get("max_results", 5)
    
    if not query:
        return {"error": "Query is required"}
    
    relevant_chunks = vector_store.search(query, limit=max_results)
    
    return {
        "query": query,
        "total_results": len(relevant_chunks),
        "results": [
            {
                "document_name": chunk["document_name"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "content_preview": chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"],
                "content_hash": chunk["content_hash"],
                "similarity_score": chunk.get("similarity_score", 0)
            }
            for chunk in relevant_chunks
        ]
    }

@app.post("/api/chat")
async def chat(
    body: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """
    Chat endpoint for MCL Assistant with vision support.
    
    Supports both text-only and multimodal (text + images) messages.
    Images should be sent as base64-encoded data URLs in the content array.
    """
    try:
        # Generate unique response ID for tracking
        response_id = generate_response_id()
        
        logger.info(f"[CHAT API] New chat request received (ID: {response_id})")
        
        # Convert Pydantic messages to dicts for the AI service
        messages_for_ai = []
        for msg in body.messages:
            if msg.content:
                # Handle multimodal content (text + images)
                if isinstance(msg.content, list):
                    content_items = []
                    for item in msg.content:
                        if hasattr(item, 'type'):
                            if item.type == 'text' and item.text:
                                content_items.append({
                                    "type": "text",
                                    "text": item.text
                                })
                            elif item.type == 'image_url' and item.image_url:
                                content_items.append({
                                    "type": "image_url",
                                    "image_url": item.image_url
                                })
                    
                    messages_for_ai.append({
                        "role": msg.role,
                        "content": content_items
                    })
                else:
                    # Simple text content
                    content_text = str(msg.content)
                    messages_for_ai.append({
                        "role": msg.role,
                        "content": content_text
                    })

        context_analysis = analyze_situational_context(messages_for_ai)
        
        result = await chat_service.process_chat_request(
            messages_for_ai,
            situational_context=context_analysis,
            session_id=body.session_id
        )
        
        if result["success"]:
            return ChatResponse(
                response=result["response"],
                response_id=response_id,
                sources=[]
            )
        else:
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"[CHAT API] Error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Feedback Endpoint ---

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit user feedback for an AI response.
    
    Args:
        feedback: Feedback request containing response_id, feedback_type, and optional comment
        db: Database session
    
    Returns:
        FeedbackResponse with the created feedback record
    
    Raises:
        400: If feedback already exists for this response_id
        404: If response_id is not found (optional validation)
        422: If feedback_type is invalid
        500: If database error occurs
    """
    logger.info(f"[FEEDBACK API] New feedback submission received")
    logger.info(f"[FEEDBACK API] Response ID: {feedback.response_id}")
    logger.info(f"[FEEDBACK API] Feedback Type: {feedback.feedback_type}")
    logger.info(f"[FEEDBACK API] Has Comment: {'Yes' if feedback.user_comment else 'No'}")
    
    # Validate feedback type
    if feedback.feedback_type not in ["positive", "negative"]:
        logger.warning(f"[FEEDBACK API] Invalid feedback type: {feedback.feedback_type}")
        raise HTTPException(
            status_code=422,
            detail="feedback_type must be 'positive' or 'negative'"
        )
    
    try:
        # Check if feedback already exists for this response
        result = await db.execute(
            select(Feedback).where(Feedback.response_id == feedback.response_id)
        )
        existing_feedback = result.scalar_one_or_none()
        
        if existing_feedback:
            logger.warning(f"[FEEDBACK API] Duplicate feedback for response: {feedback.response_id}")
            raise HTTPException(
                status_code=400,
                detail="Feedback already submitted for this response"
            )
        
        # Create new feedback record
        db_feedback = Feedback(
            response_id=feedback.response_id,
            feedback_type=feedback.feedback_type,
            user_comment=feedback.user_comment,
            created_at=datetime.utcnow(),
            processed=False
        )
        
        db.add(db_feedback)
        await db.commit()
        await db.refresh(db_feedback)
        
        logger.info(f"[FEEDBACK API] Feedback saved successfully (ID: {db_feedback.id})")
        
        # Return the feedback response
        return FeedbackResponse(
            id=db_feedback.id,
            response_id=db_feedback.response_id,
            feedback_type=db_feedback.feedback_type,
            user_comment=db_feedback.user_comment,
            created_at=db_feedback.created_at,
            processed=db_feedback.processed
        )
        
    except exc.IntegrityError as e:
        await db.rollback()
        logger.error(f"[FEEDBACK API] Database integrity error: {e}")
        raise HTTPException(
            status_code=400,
            detail="Feedback already submitted for this response"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"[FEEDBACK API] Database error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
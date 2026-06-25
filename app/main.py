import uvicorn
import uuid
import httpx
from contextlib import asynccontextmanager
import tempfile
import os
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import exc

from app.models import ChatRequest, Message, ContentItem, ChatResponse, generate_response_id, FeedbackRequest, FeedbackResponse, SessionRequest, SessionResponse, MarketInfo, UserMarketsResponse, AuthContext, MemorySaveRequest, MemoryInfo, MemoryListResponse, MemorySaveResponse, MemoryUpdateRequest, MemoryRecallResponse, MemoryStoreRequest
from app.core.config import ENABLE_MCL_IMAGE_VALIDATION, get_db, engine, VECTOR_STORE_PATH, CORS_ORIGINS, AsyncSessionLocal, RAGIE_API_KEY
from app.core.database import Feedback, Base
from app.core.dependencies import get_vector_store_service, get_chat_service, get_speech_service
from app.services.chat_service import ChatService
from app.services.vector_store import VectorStoreService
from app.services.ingestion_service import IngestionService
from app.services.speech_service import SpeechService
from app.routers import vision, admin
from app.core.logging import setup_logging, get_logger, request_id_var
from app.clients.mcl_service_client import MCLServiceClient
from app.services.memory_service import MemoryService

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

# Serve visual guide images as static files
app.mount("/images", StaticFiles(directory="static/images"), name="images")

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
    query = str(query_data.get("query", "")).strip()
    try:
        max_results = int(query_data.get("max_results", 5))
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, max_results)
    
    if not query:
        return {"error": "Query is required"}
    
    relevant_chunks = vector_store.hybrid_search(query, limit=max_results)
    
    return {
        "query": query,
        "total_results": len(relevant_chunks),
        "results": [
            {
                "source": chunk.get("source"),
                "source_title": chunk.get("source_title") or chunk.get("source"),
                "header_path": chunk.get("header_path"),
                "doc_type": chunk.get("doc_type"),
                "chunk_index": chunk.get("chunk_index", 0),
                "score": chunk.get("score", 0),
                "uuid": chunk.get("uuid"),
                "content_preview": (chunk.get("text") or "")[:500],
            }
            for chunk in relevant_chunks
        ]
    }

def _content_item_to_dict(item: ContentItem) -> dict[str, Any] | None:
    if item.type == "text" and item.text:
        return {"type": "text", "text": item.text}
    if item.type == "image_url" and item.image_url:
        return {"type": "image_url", "image_url": item.image_url}
    return None


def _message_to_ai_dict(message: Message) -> dict[str, Any]:
    if isinstance(message.content, list):
        converted = (_content_item_to_dict(item) for item in message.content)
        return {"role": message.role, "content": [item for item in converted if item]}
    return {"role": message.role, "content": str(message.content)}


def _to_ai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [_message_to_ai_dict(message) for message in messages if message.content]


def _build_chat_response(result: dict[str, Any], response_id: str) -> ChatResponse:
    if not result["success"]:
        error_message = result.get("error", "Unknown error occurred")
        logger.error(f"[CHAT API] Error: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)
    return ChatResponse(response=result["response"], response_id=response_id, sources=[])


@app.post("/api/chat")
async def chat(
    body: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    response_id = generate_response_id()
    logger.info(f"[CHAT API] New chat request received (ID: {response_id})")

    try:
        messages = _to_ai_messages(body.messages)
        result = await chat_service.process_chat_request(
            messages,
            session_id=body.session_id,
            auth_context=body.auth_context,
        )
        return _build_chat_response(result, response_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/api/ragie/image")
async def ragie_image(document_id: str, chunk_id: str):
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=503, detail="Image service unavailable")

    upstream_url = (
        f"https://api.ragie.ai/documents/{document_id}/chunks/{chunk_id}/content"
        "?media_type=image/jpeg"
    )
    async with httpx.AsyncClient() as http:
        upstream = await http.get(
            upstream_url, headers={"Authorization": f"Bearer {RAGIE_API_KEY}"}, timeout=30
        )
    if upstream.status_code != 200:
        raise HTTPException(status_code=502, detail="Could not fetch image from Ragie")
    return Response(
        content=upstream.content,
        media_type=upstream.headers.get("content-type", "image/jpeg"),
    )

# --- Feedback Endpoint ---

@app.post("/api/auth/session", response_model=SessionResponse)
async def resolve_session(body: SessionRequest):
    """Establish a session from a token shared by the MCL app.

    The MCL app hands off the user's bearer token (no login here). We call
    MCL's UserInfo to resolve the identity (user_id, company_id, ...) needed
    by the user-specific data endpoints, and return it to the frontend to
    store for the duration of the chat session.
    """
    if not body.access_token:
        raise HTTPException(status_code=400, detail="access_token is required")
    try:
        client = MCLServiceClient()
        info = await client.get_user_info(body.access_token)
        return SessionResponse(
            access_token=body.access_token,
            user_id=info.get("id", ""),
            company_id=info.get("companyId", ""),
            company_name=info.get("companyName", ""),
            full_name=info.get("fullName", ""),
            email=info.get("email", ""),
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"[SESSION] UserInfo failed: {e.response.status_code}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except Exception as e:
        logger.error(f"[SESSION] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/user-markets", response_model=UserMarketsResponse)
async def get_user_markets(body: AuthContext):
    """Get markets assigned to the authenticated user."""
    if not body.access_token or not body.company_id or not body.user_id:
        raise HTTPException(status_code=400, detail="Missing auth context fields")
    try:
        client = MCLServiceClient()
        markets_data = await client.get_user_markets(
            body.access_token, body.company_id, body.user_id
        )
        markets = [
            MarketInfo(
                id=m.get("id", ""),
                name=m.get("name", ""),
                soll_bestand=m.get("sollBestand"),
                kasseneinsaetze=m.get("kasseneinsaetze"),
                summe_kasseneinsatze=m.get("summeKasseneinsatze"),
            )
            for m in markets_data
        ]
        return UserMarketsResponse(markets=markets, total=len(markets))
    except httpx.HTTPStatusError as e:
        logger.error(f"[MARKETS] Failed: {e.response.status_code}")
        raise HTTPException(status_code=502, detail="Failed to fetch markets from MCL")
    except Exception as e:
        logger.error(f"[MARKETS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory/store", response_model=dict)
async def store_messages(body: MemoryStoreRequest):
    """Store raw messages for a session (no GPT extraction — fast)."""
    try:
        service = MemoryService(body.user_id)
        service.store_messages(body.session_id, body.messages)
        return {"stored": True, "session_id": body.session_id, "count": len(body.messages)}
    except Exception as e:
        logger.error(f"[MEMORY] Store error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory/save", response_model=MemorySaveResponse)
async def save_memories(body: MemorySaveRequest):
    """Extract and save important memories from a conversation. If session_id is provided, reads from store."""
    try:
        service = MemoryService(body.user_id)
        messages = body.messages
        if (not messages or len(messages) == 0) and body.session_id:
            messages = service.get_stored_messages(body.session_id)
        if not messages:
            return MemorySaveResponse(saved=[], updated=[], deleted=[])
        result = await service.process_and_save(messages)
        if body.session_id:
            service.clear_stored_messages(body.session_id)
        return MemorySaveResponse(
            saved=[MemoryInfo(**m) for m in result["saved"]],
            updated=[MemoryInfo(**m) for m in result["updated"]],
            deleted=result["deleted"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MEMORY] Save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/list", response_model=MemoryListResponse)
async def list_memories(user_id: Optional[str] = None):
    """List the user's saved memories."""
    try:
        service = MemoryService(user_id)
        memories = service.list_memories()
        return MemoryListResponse(memories=[MemoryInfo(**m) for m in memories])
    except Exception as e:
        logger.error(f"[MEMORY] List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/{memory_id}", response_model=MemoryInfo)
async def get_memory(memory_id: str, user_id: Optional[str] = None):
    """Get a single memory by ID."""
    service = MemoryService(user_id)
    mem = service.get_memory(memory_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryInfo(**mem)


@app.put("/api/memory/{memory_id}", response_model=MemoryInfo)
async def update_memory(memory_id: str, body: MemoryUpdateRequest, user_id: Optional[str] = None):
    """Update the content of a memory (user edit)."""
    service = MemoryService(user_id)
    result = service.update_memory_content(memory_id, body.content)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryInfo(**result)


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str, user_id: Optional[str] = None):
    """Delete a memory file."""
    service = MemoryService(user_id)
    if service.delete_memory(memory_id):
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Memory not found")


@app.post("/api/memory/recall", response_model=MemoryRecallResponse)
async def recall_memories(user_id: Optional[str] = None):
    """Recall the user's memories as formatted context for system prompt injection."""
    try:
        service = MemoryService(user_id)
        context = service.recall_context()
        memories = service.list_memories()
        return MemoryRecallResponse(
            context=context,
            memories=[MemoryInfo(**m) for m in memories],
        )
    except Exception as e:
        logger.error(f"[MEMORY] Recall error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



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


@app.post("/api/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    speech_service: SpeechService = Depends(get_speech_service)
):
    """
    Transcribe audio to text using OpenAI Whisper.

    Accepts a multipart form upload with an audio file (webm, wav, mp3, etc.)
    and returns the transcribed text.
    """
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an audio file"
        )

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        text = await speech_service.transcribe(audio_bytes, filename=file.filename or "audio.webm")
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SPEECH API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

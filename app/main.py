import json
import uvicorn
from datetime import datetime

from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatRequest, Message, ContentItem
from app.database.models import ChatResponse, generate_response_id, FeedbackRequest, FeedbackResponse, CuratedQaRequest, CuratedQaResponse
from app.database.repositories import FeedbackRepository, CuratedQaRepository
from app.config import get_db
from app.services import start_mcl_knowledge_base, get_mcl_ai_response, _document_chunks, find_relevant_chunks

# Global variable to store the vector store ID
VECTOR_STORE_ID = None

# Global cache to store response data for feedback tracking
RESPONSE_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global VECTOR_STORE_ID
    print("MCL Assistant startup sequence initiated...")
    try:
        VECTOR_STORE_ID = start_mcl_knowledge_base()

        if not VECTOR_STORE_ID:
            print("CRITICAL: Failed to initialize the MCL knowledge base. The vector store ID is missing.")
        else:
            print(f"MCL knowledge base loaded successfully. Vector Store ID: {VECTOR_STORE_ID}")
        print("MCL Assistant startup sequence completed.")
        yield
    except Exception as e:
        print(f"FATAL ERROR during MCL Assistant startup: {e}")
        yield  # Ensure yield is called even on exception for proper shutdown handling
    finally:
        print("MCL Assistant will now shut down")

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="MCL Assistant API",
    description="An AI-powered assistant for the MCL (Mobile Checklist) application",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_messages_to_objects(messages_str: str) -> ChatRequest:
    """Convert string representation of messages to ChatRequest object."""
    try:
        request_data = json.loads(messages_str)
        
        messages_list = []
        for msg_data in request_data.get("messages", []):
            content_input = msg_data.get("content")
            processed_content_list = []

            if isinstance(content_input, str):
                processed_content_list.append(ContentItem(text=content_input, type="text"))
            elif isinstance(content_input, list):
                for item in content_input:
                    if isinstance(item, dict) and "text" in item and "type" in item:
                        processed_content_list.append(ContentItem(text=item["text"], type=item["type"]))
            
            messages_list.append(Message(role=msg_data.get("role"), content=processed_content_list))
        
        chat_request_obj = ChatRequest(messages=messages_list)
        return chat_request_obj
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in messages_str: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field in messages_str content: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing messages_str: {str(e)}")

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "MCL Assistant API",
        "version": "1.0.0",
        "description": "AI-powered assistant for the MCL (Mobile Checklist) application",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "knowledge_base_loaded": VECTOR_STORE_ID is not None,
        "vector_store_id": VECTOR_STORE_ID if VECTOR_STORE_ID else "Not loaded",
        "total_document_chunks": len(_document_chunks)
    }

@app.get("/api/chunks")
async def get_chunks_info():
    """Get information about available document chunks."""
    if not _document_chunks:
        return {"message": "No document chunks available", "chunks": []}
    
    # Group chunks by document
    documents_info = {}
    for chunk in _document_chunks:
        doc_name = chunk["document_name"]
        if doc_name not in documents_info:
            documents_info[doc_name] = {
                "document_name": doc_name,
                "document_type": chunk["document_type"],
                "total_chunks": 0,
                "chunks": []
            }
        documents_info[doc_name]["total_chunks"] += 1
        documents_info[doc_name]["chunks"].append({
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "content_hash": chunk["content_hash"]
        })
    
    return {
        "total_chunks": len(_document_chunks),
        "total_documents": len(documents_info),
        "documents": list(documents_info.values())
    }

@app.post("/api/search")
async def search_chunks(query_data: dict):
    """Search for relevant chunks based on a query."""
    query = query_data.get("query", "")
    max_results = query_data.get("max_results", 5)
    
    if not query:
        return {"error": "Query is required"}
    
    relevant_chunks = find_relevant_chunks(query, max_chunks=max_results)
    
    return {
        "query": query,
        "total_results": len(relevant_chunks),
        "results": [
            {
                "document_name": chunk["document_name"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "content_preview": chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"],
                "content_hash": chunk["content_hash"]
            }
            for chunk in relevant_chunks
        ]
    }

@app.post("/api/chat")
async def chat(body: ChatRequest) -> ChatResponse:
    """Main chat endpoint for MCL Assistant."""
    global VECTOR_STORE_ID
    
    if not VECTOR_STORE_ID:
        raise HTTPException(
            status_code=503, 
            detail="MCL knowledge base is not available. Please try again later."
        )

    try:
        # Generate unique response ID for tracking
        response_id = generate_response_id()
        
        # Convert Pydantic messages to dicts for the AI service
        messages_for_ai = []
        user_question = ""
        
        for msg in body.messages:
            if msg.content:
                if isinstance(msg.content, list):
                    # Extract text from ContentItem objects
                    content_text = " ".join([item.text for item in msg.content if hasattr(item, 'text')])
                else:
                    content_text = str(msg.content)
                
                # Store the last user message as the question
                if msg.role == "user":
                    user_question = content_text
                
                messages_for_ai.append({
                    "role": msg.role,
                    "content": content_text
                })

        print(f"Messages sent to MCL AI: {messages_for_ai}")
        
        # Get AI response
        ai_response_obj = get_mcl_ai_response(messages_for_ai)
        print(f"MCL AI response received")
        
        response_message = ai_response_obj.choices[0].message
        ai_response_text = response_message.content

        print(f"Final response prepared with response ID: {response_id}")
        
        # Store response data for feedback tracking
        RESPONSE_CACHE[response_id] = {
            "user_question": user_question,
            "ai_response": ai_response_text,
            "retrieved_chunks": [],  # TODO: Get actual chunks from find_relevant_chunks
            "timestamp": datetime.now().isoformat()
        }
        
        # Keep cache size manageable (keep only last 1000 responses)
        if len(RESPONSE_CACHE) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(RESPONSE_CACHE.items(), key=lambda x: x[1]["timestamp"])
            for old_response_id, _ in sorted_cache[:100]:  # Remove oldest 100
                del RESPONSE_CACHE[old_response_id]
        
        # Return the new ChatResponse format with tracking
        return ChatResponse(
            response=ai_response_text,
            response_id=response_id,
            sources=[]  # TODO: Add source documents from retrieved chunks
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Feedback and Learning Endpoints ---

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback_request: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
) -> FeedbackResponse:
    """Submit user feedback for an AI response."""
    try:
        feedback_repo = FeedbackRepository(db)
        
        # Check if feedback already exists for this response
        existing_feedback = await feedback_repo.get_feedback_by_response_id(feedback_request.response_id)
        if existing_feedback:
            raise HTTPException(
                status_code=400, 
                detail="Feedback already submitted for this response"
            )
        
        # Get cached response data
        cached_data = RESPONSE_CACHE.get(feedback_request.response_id)
        if not cached_data:
            raise HTTPException(
                status_code=404,
                detail="Response not found. Cannot submit feedback for this response."
            )
        
        # Create feedback with actual question and response data
        feedback = await feedback_repo.create_feedback(
            response_id=feedback_request.response_id,
            user_question=cached_data["user_question"],
            ai_response=cached_data["ai_response"],
            feedback_type=feedback_request.feedback_type,
            user_comment=feedback_request.user_comment,
            retrieved_chunks=cached_data.get("retrieved_chunks", [])
        )
        
        # Clean up cache entry (optional - you might want to keep it longer)
        # del RESPONSE_CACHE[feedback_request.response_id]
        
        return FeedbackResponse(
            id=feedback.id,
            response_id=feedback.response_id,
            feedback_type=feedback.feedback_type,
            user_comment=feedback.user_comment,
            created_at=feedback.created_at,
            processed=feedback.processed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@app.get("/api/admin/feedback/stats")
async def get_feedback_stats(db: AsyncSession = Depends(get_db)):
    """Get feedback statistics for admin dashboard."""
    try:
        feedback_repo = FeedbackRepository(db)
        stats = await feedback_repo.get_feedback_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback stats: {str(e)}")

@app.get("/api/admin/feedback/unprocessed")
async def get_unprocessed_feedback(
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get unprocessed feedback for admin review."""
    try:
        feedback_repo = FeedbackRepository(db)
        feedback_list = await feedback_repo.get_unprocessed_feedback(limit=limit)
        
        return {
            "total": len(feedback_list),
            "feedback": [
                {
                    "id": f.id,
                    "response_id": f.response_id,
                    "feedback_type": f.feedback_type,
                    "user_comment": f.user_comment,
                    "created_at": f.created_at,
                    "user_question": f.user_question,
                    "ai_response": f.ai_response
                }
                for f in feedback_list
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving unprocessed feedback: {str(e)}")

@app.post("/api/admin/feedback/{feedback_id}/mark-processed")
async def mark_feedback_processed(
    feedback_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Mark feedback as processed."""
    try:
        feedback_repo = FeedbackRepository(db)
        success = await feedback_repo.mark_feedback_processed(feedback_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        return {"message": "Feedback marked as processed", "feedback_id": feedback_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking feedback as processed: {str(e)}")

@app.post("/api/admin/curated-qa", response_model=CuratedQaResponse)
async def create_curated_qa(
    qa_request: CuratedQaRequest,
    db: AsyncSession = Depends(get_db)
) -> CuratedQaResponse:
    """Create a new curated Q&A pair from positive feedback."""
    try:
        curated_repo = CuratedQaRepository(db)
        
        curated_qa = await curated_repo.create_curated_qa(
            question=qa_request.question,
            answer=qa_request.answer,
            source_feedback_id=qa_request.source_feedback_id
        )
        
        return CuratedQaResponse(
            id=curated_qa.id,
            question=curated_qa.question,
            answer=curated_qa.answer,
            source_feedback_id=curated_qa.source_feedback_id,
            created_at=curated_qa.created_at,
            active=curated_qa.active
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating curated Q&A: {str(e)}")

@app.get("/api/admin/curated-qa")
async def get_curated_qa(db: AsyncSession = Depends(get_db)):
    """Get all active curated Q&A pairs."""
    try:
        curated_repo = CuratedQaRepository(db)
        qa_list = await curated_repo.get_active_curated_qa()
        
        return {
            "total": len(qa_list),
            "curated_qa": [
                {
                    "id": qa.id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "source_feedback_id": qa.source_feedback_id,
                    "created_at": qa.created_at,
                    "active": qa.active
                }
                for qa in qa_list
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving curated Q&A: {str(e)}")

@app.delete("/api/admin/curated-qa/{qa_id}")
async def deactivate_curated_qa(
    qa_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Deactivate a curated Q&A pair."""
    try:
        curated_repo = CuratedQaRepository(db)
        success = await curated_repo.deactivate_curated_qa(qa_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Curated Q&A not found")
        
        return {"message": "Curated Q&A deactivated", "qa_id": qa_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deactivating curated Q&A: {str(e)}")

# --- Debug Endpoints ---

@app.get("/api/debug/response-cache")
async def get_response_cache():
    """Debug endpoint to check response cache status."""
    return {
        "cache_size": len(RESPONSE_CACHE),
        "recent_responses": list(RESPONSE_CACHE.keys())[-10:] if RESPONSE_CACHE else [],
        "oldest_timestamp": min([data["timestamp"] for data in RESPONSE_CACHE.values()]) if RESPONSE_CACHE else None,
        "newest_timestamp": max([data["timestamp"] for data in RESPONSE_CACHE.values()]) if RESPONSE_CACHE else None
    }

@app.get("/api/debug/response-cache/{response_id}")
async def get_cached_response(response_id: str):
    """Debug endpoint to check specific response data."""
    cached_data = RESPONSE_CACHE.get(response_id)
    if not cached_data:
        raise HTTPException(status_code=404, detail="Response not found in cache")
    return cached_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
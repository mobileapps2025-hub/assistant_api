import json
import uvicorn
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ChatRequest, Message, ContentItem, ChatResponse, generate_response_id,
    FeedbackRequest, FeedbackResponse, CuratedQaRequest, CuratedQaResponse
)
from app.services import (
    # Spotplan functions
    get_stores, get_week_events, get_events_between_weeks, get_events_by_name,
    get_event_details, get_unplanned_events_between_weeks, get_store_unplanned_events,
    get_company_unplanned_events, get_store_sales_areas, get_unplanned_sales_areas_on_week,
    start_spotplan_knowledge_base, get_spotplan_ai_response, set_api_client_token,
    # MCL functions
    start_mcl_knowledge_base, get_mcl_ai_response, _mcl_document_chunks, find_relevant_chunks
)
from app.config import get_db

# Available Spotplan functions for tool calling
AVAILABLE_SPOTPLAN_FUNCTIONS = {
    "get_stores": get_stores,
    "get_week_events": get_week_events,
    "get_events_between_weeks": get_events_between_weeks,
    "get_events_by_name": get_events_by_name,
    "get_event_details": get_event_details,
    "get_store_unplanned_events": get_store_unplanned_events,
    "get_company_unplanned_events": get_company_unplanned_events,
    "get_store_sales_areas": get_store_sales_areas,
    "get_unplanned_sales_areas_on_week": get_unplanned_sales_areas_on_week,
    "get_unplanned_events_between_weeks": get_unplanned_events_between_weeks,
}

# Global variables for knowledge bases
SPOTPLAN_VECTOR_STORE_ID = None
MCL_VECTOR_STORE_ID = None

# Global cache to store response data for feedback tracking
RESPONSE_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global SPOTPLAN_VECTOR_STORE_ID, MCL_VECTOR_STORE_ID
    print("Unified Assistant startup sequence initiated...")
    
    try:
        # Initialize Spotplan knowledge base
        print("Initializing Spotplan knowledge base...")
        SPOTPLAN_VECTOR_STORE_ID = start_spotplan_knowledge_base()
        if SPOTPLAN_VECTOR_STORE_ID:
            print(f"Spotplan knowledge base loaded successfully. Vector Store ID: {SPOTPLAN_VECTOR_STORE_ID}")
        else:
            print("WARNING: Failed to initialize Spotplan knowledge base")

        # Initialize MCL knowledge base
        print("Initializing MCL knowledge base...")
        MCL_VECTOR_STORE_ID = start_mcl_knowledge_base()
        if MCL_VECTOR_STORE_ID:
            print(f"MCL knowledge base loaded successfully. Vector Store ID: {MCL_VECTOR_STORE_ID}")
        else:
            print("WARNING: Failed to initialize MCL knowledge base")

        print("Unified Assistant startup sequence completed.")
        yield
    except Exception as e:
        print(f"FATAL ERROR during Unified Assistant startup: {e}")
        yield
    finally:
        print("Unified Assistant will now shut down")

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Unified Assistant API", 
    description="An AI-powered assistant for both Spotplan and MCL applications",
    version="2.0.0",
    lifespan=lifespan
)
security = HTTPBearer(auto_error=False)  # Make authentication optional
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def detect_app_type(request: ChatRequest) -> str:
    """Detect app type from request or message content."""
    # First check if app_type is explicitly provided
    if hasattr(request, 'app_type') and request.app_type:
        return request.app_type.lower()
    
    # If not provided, try to detect from message content
    if request.messages:
        latest_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                if isinstance(msg.content, list):
                    latest_message = " ".join([item.text for item in msg.content if hasattr(item, 'text')])
                else:
                    latest_message = str(msg.content)
                break
        
        latest_message_lower = latest_message.lower()
        
        # MCL keywords (English and German)
        mcl_keywords = [
            "mcl", "mobile checklist", "checklist", "quiz", "question", "dashboard", "tablet",
            # German keywords
            "checkliste", "prüfliste", "kontrollliste", "aufgaben", "fragen", "quiz", 
            "dashboard", "tablet", "mobile", "anlegen", "erstellen", "ausfüllen"
        ]
        # Spotplan keywords  
        spotplan_keywords = ["spotplan", "store", "event", "sales area", "week", "planning", "unplanned"]
        
        mcl_score = sum(1 for keyword in mcl_keywords if keyword in latest_message_lower)
        spotplan_score = sum(1 for keyword in spotplan_keywords if keyword in latest_message_lower)
        
        if mcl_score > spotplan_score:
            return "mcl"
        elif spotplan_score > mcl_score:
            return "spotplan"
    
    # Default to spotplan if uncertain (maintains backward compatibility)
    return "spotplan"

async def get_spotplan_api_data(tool_call):
    """Handle Spotplan API function calls."""
    function_name = tool_call.function.name
    function_to_call = AVAILABLE_SPOTPLAN_FUNCTIONS.get(function_name)

    if not function_to_call:
        raise HTTPException(status_code=500, detail=f"Function '{function_name}' not found.")

    function_args = json.loads(tool_call.function.arguments)
    print(f"Function parameters: {function_args}")
    function_result = await function_to_call(**function_args)
    return function_result

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
        
        chat_request_obj = ChatRequest(
            messages=messages_list,
            app_type=request_data.get("app_type")
        )
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
        "message": "Unified Assistant API",
        "version": "2.0.0",
        "description": "AI-powered assistant for both Spotplan and MCL applications",
        "supported_apps": ["spotplan", "mcl"],
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health",
            "mcl_chunks": "/api/chunks",
            "mcl_search": "/api/search"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "spotplan_knowledge_base_loaded": SPOTPLAN_VECTOR_STORE_ID is not None,
        "mcl_knowledge_base_loaded": MCL_VECTOR_STORE_ID is not None,
        "spotplan_vector_store_id": SPOTPLAN_VECTOR_STORE_ID if SPOTPLAN_VECTOR_STORE_ID else "Not loaded",
        "mcl_vector_store_id": MCL_VECTOR_STORE_ID if MCL_VECTOR_STORE_ID else "Not loaded",
        "total_mcl_document_chunks": len(_mcl_document_chunks)
    }

@app.get("/api/chunks")
async def get_chunks_info():
    """Get information about available MCL document chunks."""
    if not _mcl_document_chunks:
        return {"message": "No MCL document chunks available", "chunks": []}
    
    # Group chunks by document
    documents_info = {}
    for chunk in _mcl_document_chunks:
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
        "total_chunks": len(_mcl_document_chunks),
        "total_documents": len(documents_info),
        "documents": list(documents_info.values())
    }

@app.post("/api/search")
async def search_chunks(query_data: dict):
    """Search for relevant MCL chunks based on a query."""
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
async def chat(
    body: ChatRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> ChatResponse:
    """Unified chat endpoint for both Spotplan and MCL Assistant."""
    
    # Detect which app we're dealing with
    app_type = detect_app_type(body)
    print(f"Detected app type: {app_type}")
    
    if app_type == "mcl":
        return await handle_mcl_chat(body)
    else:
        # Spotplan requires authentication
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for Spotplan operations"
            )
        token = credentials.credentials
        return await handle_spotplan_chat(body, token)

async def handle_mcl_chat(body: ChatRequest) -> ChatResponse:
    """Handle MCL-specific chat requests."""
    global MCL_VECTOR_STORE_ID
    
    if not MCL_VECTOR_STORE_ID:
        # Detect language for appropriate error message
        user_message = ""
        if body.messages:
            for msg in body.messages:
                if msg.role == "user" and msg.content:
                    if isinstance(msg.content, list):
                        user_message = " ".join([item.text for item in msg.content if hasattr(item, 'text')])
                    else:
                        user_message = str(msg.content)
                    break
        
        # Simple German detection for error message
        error_message = "MCL knowledge base is not available. Please try again later."
        if user_message and any(word in user_message.lower() for word in ['ich', 'du', 'der', 'die', 'das', 'kannst', 'mir']):
            error_message = "MCL-Wissensdatenbank ist nicht verfügbar. Bitte versuchen Sie es später erneut."
        
        raise HTTPException(
            status_code=503, 
            detail=error_message
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
                    content_text = " ".join([item.text for item in msg.content if hasattr(item, 'text')])
                else:
                    content_text = str(msg.content)
                
                if msg.role == "user":
                    user_question = content_text
                
                messages_for_ai.append({
                    "role": msg.role,
                    "content": content_text
                })

        print(f"Messages sent to MCL AI: {messages_for_ai}")
        
        # Get AI response
        ai_response_obj = get_mcl_ai_response(messages_for_ai)
        response_message = ai_response_obj.choices[0].message
        ai_response_text = response_message.content

        # Store response data for feedback tracking
        RESPONSE_CACHE[response_id] = {
            "user_question": user_question,
            "ai_response": ai_response_text,
            "retrieved_chunks": [],
            "timestamp": datetime.now().isoformat(),
            "app_type": "mcl"
        }
        
        # Keep cache size manageable
        if len(RESPONSE_CACHE) > 1000:
            sorted_cache = sorted(RESPONSE_CACHE.items(), key=lambda x: x[1]["timestamp"])
            for old_response_id, _ in sorted_cache[:100]:
                del RESPONSE_CACHE[old_response_id]
        
        return ChatResponse(
            response=ai_response_text,
            response_id=response_id,
            sources=[],
            app_type="mcl"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in MCL chat: {str(e)}")

async def handle_spotplan_chat(body: ChatRequest, token: str) -> ChatResponse:
    """Handle Spotplan-specific chat requests."""
    global SPOTPLAN_VECTOR_STORE_ID
    
    # Set API client token for Spotplan operations
    set_api_client_token(token)
    
    chat_request_obj = body
    response_id = generate_response_id()
    
    fetching_data = True
    count = 0
    
    try:
        while fetching_data:
            count += 1
            if count > 5:
                raise HTTPException(status_code=500, detail="Too many iterations, something went wrong.")
            
            # Convert Pydantic messages to dicts for the AI service
            messages_for_ai = [msg.model_dump(exclude_none=True) for msg in chat_request_obj.messages]
            print(f"Messages about to be sent to Spotplan AI: {messages_for_ai}")
            
            ai_response_obj = get_spotplan_ai_response(messages_for_ai)
            print(f"Spotplan AI response object: {ai_response_obj.model_dump()}")
            response_message_from_ai = ai_response_obj.choices[0].message

            # Convert AI response message to our Pydantic Message model
            ai_message_content_list = []
            if response_message_from_ai.content:
                ai_message_content_list.append(ContentItem(text=response_message_from_ai.content, type="text"))
            
            chat_request_obj.messages.append(response_message_from_ai)

            if response_message_from_ai.tool_calls:
                fetching_data = True
                for tool_call in response_message_from_ai.tool_calls:
                    print(f"Response message before tool call: {response_message_from_ai}")
                    function_result = await get_spotplan_api_data(tool_call)

                    # Convert tool response to Pydantic Message model
                    tool_response_pydantic_msg = Message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        content=[ContentItem(text=json.dumps(function_result), type="text")]
                    )
                    chat_request_obj.messages.append(tool_response_pydantic_msg)
            else:
                fetching_data = False

        # Get final response content
        final_response = ""
        for msg in reversed(chat_request_obj.messages):
            if msg.role == "assistant" and msg.content:
                if isinstance(msg.content, list):
                    final_response = " ".join([item.text for item in msg.content if hasattr(item, 'text')])
                else:
                    final_response = str(msg.content)
                break

        return ChatResponse(
            response=final_response,
            response_id=response_id,
            sources=[],
            app_type="spotplan"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in Spotplan chat: {str(e)}")

# --- MCL Feedback and Learning Endpoints ---

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback_request: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
) -> FeedbackResponse:
    """Submit user feedback for an AI response."""
    try:
        from app.database.repositories import FeedbackRepository
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
        from app.database.repositories import FeedbackRepository
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
        from app.database.repositories import FeedbackRepository
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
        from app.database.repositories import FeedbackRepository
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
        from app.database.repositories import CuratedQaRepository
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
        from app.database.repositories import CuratedQaRepository
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
        from app.database.repositories import CuratedQaRepository
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
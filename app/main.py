import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, Message, ContentItem, ChatResponse, generate_response_id
from app.services import start_mcl_knowledge_base, get_mcl_ai_response, _mcl_document_chunks, find_relevant_chunks

# Global variable for MCL knowledge base
MCL_VECTOR_STORE_ID = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCL knowledge base on startup."""
    global MCL_VECTOR_STORE_ID
    print("MCL Assistant startup sequence initiated...")
    
    try:
        # Initialize MCL knowledge base
        print("Initializing MCL knowledge base...")
        MCL_VECTOR_STORE_ID = start_mcl_knowledge_base()
        if MCL_VECTOR_STORE_ID:
            print(f"✓ MCL knowledge base loaded successfully. Vector Store ID: {MCL_VECTOR_STORE_ID}")
            print(f"✓ Total document chunks: {len(_mcl_document_chunks)}")
        else:
            print("⚠ WARNING: Failed to initialize MCL knowledge base")

        print("MCL Assistant startup completed.")
        yield
    except Exception as e:
        print(f"✗ FATAL ERROR during MCL Assistant startup: {e}")
        yield
    finally:
        print("MCL Assistant shutting down...")

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="MCL Assistant API", 
    description="AI-powered knowledge base assistant for the MCL (Mobile Checklist) application",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            "health": "/health",
            "chunks": "/api/chunks",
            "search": "/api/search"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "knowledge_base_loaded": MCL_VECTOR_STORE_ID is not None,
        "vector_store_id": MCL_VECTOR_STORE_ID if MCL_VECTOR_STORE_ID else "Not loaded",
        "total_document_chunks": len(_mcl_document_chunks),
        "documents_processed": len(set(chunk['document_name'] for chunk in _mcl_document_chunks))
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
async def chat(body: ChatRequest) -> ChatResponse:
    """Chat endpoint for MCL Assistant."""
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

        print(f"[MCL API] User question: {user_question}")
        
        # Get AI response
        ai_response_obj = get_mcl_ai_response(messages_for_ai)
        response_message = ai_response_obj.choices[0].message
        ai_response_text = response_message.content

        print(f"[MCL API] Response generated (length: {len(ai_response_text)} chars)")
        
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
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
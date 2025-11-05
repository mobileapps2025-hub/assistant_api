import uvicorn
from contextlib import asynccontextmanager
import tempfile
import os

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, Message, ContentItem, ChatResponse, generate_response_id
from app.services import (
    start_mcl_knowledge_base, 
    get_mcl_ai_response, 
    get_vision_enabled_response,
    _mcl_document_chunks, 
    find_relevant_chunks
)

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
    """
    Chat endpoint for MCL Assistant with vision support.
    
    Supports both text-only and multimodal (text + images) messages.
    Images should be sent as base64-encoded data URLs in the content array.
    
    Example text-only message:
    {
        "messages": [
            {"role": "user", "content": "How do I create a checklist?"}
        ]
    }
    
    Example multimodal message:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What can I do on this screen?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]
            }
        ]
    }
    """
    global MCL_VECTOR_STORE_ID
    
    if not MCL_VECTOR_STORE_ID:
        # Detect language for appropriate error message
        user_message = ""
        if body.messages:
            for msg in body.messages:
                if msg.role == "user" and msg.content:
                    if isinstance(msg.content, list):
                        user_message = " ".join([
                            item.text for item in msg.content 
                            if hasattr(item, 'text') and item.text
                        ])
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
        
        print("\n" + "="*80)
        print(f"[CHAT API] New chat request received (ID: {response_id})")
        print("="*80)
        
        # Convert Pydantic messages to dicts for the AI service
        messages_for_ai = []
        user_question = ""
        has_images = False
        
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
                                if msg.role == "user":
                                    user_question = item.text
                            elif item.type == 'image_url' and item.image_url:
                                content_items.append({
                                    "type": "image_url",
                                    "image_url": item.image_url
                                })
                                has_images = True
                    
                    messages_for_ai.append({
                        "role": msg.role,
                        "content": content_items
                    })
                else:
                    # Simple text content
                    content_text = str(msg.content)
                    if msg.role == "user":
                        user_question = content_text
                    
                    messages_for_ai.append({
                        "role": msg.role,
                        "content": content_text
                    })

        print(f"[CHAT API] User question: {user_question[:100]}..." if len(user_question) > 100 else f"[CHAT API] User question: {user_question}")
        print(f"[CHAT API] Has images: {'Yes 🖼️' if has_images else 'No 📝'}")
        print(f"[CHAT API] Total messages: {len(messages_for_ai)}")
        
        # Use vision-enabled response handler (handles both text-only and multimodal)
        result = get_vision_enabled_response(messages_for_ai, MCL_VECTOR_STORE_ID)
        
        if result["success"]:
            ai_response_text = result["response"]
            
            print(f"[CHAT API] ✅ Response generated successfully")
            print(f"[CHAT API] Response length: {len(ai_response_text)} characters")
            print(f"[CHAT API] Vision mode: {'Yes' if result.get('has_vision') else 'No'}")
            print("="*80 + "\n")
            
            return ChatResponse(
                response=ai_response_text,
                response_id=response_id,
                sources=[],
                app_type="mcl"
            )
        else:
            error_msg = result.get("error", "Unknown error occurred")
            print(f"[CHAT API] ❌ Error: {error_msg}")
            print("="*80 + "\n")
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Vision Assistant Endpoint ---

@app.post("/api/vision/analyze-screenshot")
async def analyze_screenshot(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Analyze an MCL App screenshot and provide contextual help using GPT-4o vision.
    
    Args:
        file: Screenshot image file (PNG, JPG, JPEG, GIF, WEBP)
        query: User's question about the screenshot
    
    Returns:
        {
            "response": str,        # AI response text
            "success": bool,        # Whether analysis succeeded
            "metadata": {           # Optional metadata
                "assistant_id": str,
                "thread_id": str,
                "file_id": str
            },
            "error": str           # Error message if failed
        }
    """
    
    print("\n" + "="*80)
    print("[VISION API] New screenshot analysis request received")
    print("="*80)
    
    # Log request details
    print(f"[VISION API] File name: {file.filename}")
    print(f"[VISION API] Content type: {file.content_type}")
    print(f"[VISION API] Query: {query[:100]}..." if len(query) > 100 else f"[VISION API] Query: {query}")
    
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        print(f"[VISION API] ❌ ERROR: Invalid file type: {file.content_type}")
        print(f"[VISION API] Allowed types: {', '.join(allowed_types)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {', '.join(allowed_types)}"
        )
    
    print(f"[VISION API] ✅ File type validated: {file.content_type}")
    
    # Validate file size (max 20MB)
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    temp_file_path = None
    
    try:
        print(f"[VISION API] 📥 Saving uploaded file temporarily...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            print(f"[VISION API] Temp file path: {temp_file_path}")
            
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > 20 * 1024 * 1024:  # 20MB limit
                    print(f"[VISION API] ❌ ERROR: File too large: {file_size / (1024*1024):.2f}MB (max 20MB)")
                    raise HTTPException(status_code=400, detail="File too large (max 20MB)")
                temp_file.write(chunk)
        
        print(f"[VISION API] ✅ File saved successfully ({file_size / 1024:.2f}KB)")
        
        # Initialize vision assistant
        print(f"[VISION API] 🤖 Initializing MCL Vision Assistant...")
        from app.vision_assistant import MCLVisionAssistant
        
        try:
            assistant = MCLVisionAssistant()
            print(f"[VISION API] ✅ Vision Assistant initialized")
        except Exception as e:
            print(f"[VISION API] ❌ ERROR: Failed to initialize Vision Assistant: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to initialize Vision Assistant: {str(e)}")
        
        # Analyze the screenshot
        print(f"[VISION API] 🔍 Starting screenshot analysis...")
        print(f"[VISION API] This may take 10-20 seconds...")
        
        try:
            result = assistant.analyze_screenshot(
                image_path=temp_file_path,
                user_query=query
            )
            
            print(f"[VISION API] 📊 Analysis completed")
            print(f"[VISION API] Success: {result.get('success', False)}")
            
            if result["success"]:
                response_length = len(result.get("response", ""))
                print(f"[VISION API] ✅ Response generated ({response_length} characters)")
                print(f"[VISION API] Assistant ID: {result.get('assistant_id', 'N/A')}")
                print(f"[VISION API] Thread ID: {result.get('thread_id', 'N/A')}")
                print(f"[VISION API] File ID: {result.get('file_id', 'N/A')}")
                
                # Preview response
                response_preview = result["response"][:200] + "..." if len(result["response"]) > 200 else result["response"]
                print(f"[VISION API] Response preview: {response_preview}")
                
                print("="*80)
                print("[VISION API] ✅ Request completed successfully")
                print("="*80 + "\n")
                
                return {
                    "response": result["response"],
                    "success": True,
                    "metadata": {
                        "assistant_id": result.get("assistant_id"),
                        "thread_id": result.get("thread_id"),
                        "file_id": result.get("file_id"),
                        "image_name": file.filename,
                        "query": query
                    }
                }
            else:
                error_msg = result.get("error", "Unknown error occurred")
                print(f"[VISION API] ❌ Analysis failed: {error_msg}")
                print("="*80 + "\n")
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"[VISION API] ❌ ERROR during analysis: {e}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")
            raise HTTPException(status_code=500, detail=f"Error analyzing screenshot: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[VISION API] ❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"[VISION API] 🗑️ Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"[VISION API] ⚠️ Warning: Could not delete temp file: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
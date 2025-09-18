import json
import uvicorn

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel

from app.models import ChatRequest, Message, ContentItem
from app.services import start_mcl_knowledge_base, get_mcl_ai_response, _document_chunks, find_relevant_chunks

# Global variable to store the vector store ID
VECTOR_STORE_ID = None

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
async def chat(body: ChatRequest):
    """Main chat endpoint for MCL Assistant."""
    global VECTOR_STORE_ID
    
    if not VECTOR_STORE_ID:
        raise HTTPException(
            status_code=503, 
            detail="MCL knowledge base is not available. Please try again later."
        )

    try:
        # Convert Pydantic messages to dicts for the AI service
        messages_for_ai = []
        for msg in body.messages:
            if msg.content:
                if isinstance(msg.content, list):
                    # Extract text from ContentItem objects
                    content_text = " ".join([item.text for item in msg.content if hasattr(item, 'text')])
                else:
                    content_text = str(msg.content)
                
                messages_for_ai.append({
                    "role": msg.role,
                    "content": content_text
                })

        print(f"Messages sent to MCL AI: {messages_for_ai}")
        
        # Get AI response
        ai_response_obj = get_mcl_ai_response(messages_for_ai)
        print(f"MCL AI response received")
        
        response_message = ai_response_obj.choices[0].message

        # Convert AI response to our format
        ai_message_content = []
        if response_message.content:
            ai_message_content.append(ContentItem(text=response_message.content, type="text"))
        
        # Add AI response to messages
        body.messages.append(Message(
            role=response_message.role,
            content=ai_message_content
        ))

        print(f"Final response prepared with {len(body.messages)} messages")
        return {"messages": [msg.model_dump(exclude_none=True) for msg in body.messages]}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
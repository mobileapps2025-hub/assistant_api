import json
import os
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO
from pathlib import Path
from app.config import client, AsyncSessionLocal
from app.database.repositories import CuratedQaRepository
from datetime import datetime
from typing import List, Dict, Any, Tuple
import hashlib
import asyncio

# Global variable to store the vector store ID and document chunks
_mcl_vector_store_id: str = None
_document_chunks: List[Dict[str, Any]] = []

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using PyMuPDF for better text extraction."""
    try:
        text_content = ""
        # Try PyMuPDF first (better text extraction)
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text:
                text_content += page_text + "\n\n"
        doc.close()
        return text_content.strip()
    except Exception as e:
        print(f"Error extracting text from {file_path} with PyMuPDF: {e}")
        # Fallback to PyPDF2
        try:
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n\n"
            return text_content.strip()
        except Exception as e2:
            print(f"Error extracting text from {file_path} with PyPDF2: {e2}")
            return ""

def create_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better retrieval."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for the last period, newline, or space within the chunk
            break_points = [text.rfind('.', start, end), text.rfind('\n', start, end), text.rfind(' ', start, end)]
            best_break = max([bp for bp in break_points if bp > start + chunk_size // 2], default=end)
            end = best_break + 1 if best_break != end else end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else end
    
    return chunks

async def get_curated_qa_content() -> str:
    """Fetch curated Q&A content from database and format for knowledge base."""
    try:
        async with AsyncSessionLocal() as db:
            curated_repo = CuratedQaRepository(db)
            content = await curated_repo.get_curated_qa_content_for_kb()
            return content
    except Exception as e:
        print(f"Error fetching curated Q&A content: {e}")
        return ""

def process_mcl_documents_with_chunks() -> Tuple[List[str], List[Dict[str, Any]]]:
    """Process all MCL documents and create searchable chunks with metadata."""
    global _document_chunks
    
    documents_path = Path("app/documents")
    file_ids = []
    _document_chunks = []
    chunk_id = 0
    
    print("Processing MCL documents with chunk tracking...")
    
    # First, try to get curated Q&A content from database
    try:
        curated_content = asyncio.run(get_curated_qa_content())
        if curated_content:
            print("Processing curated Q&A content from database...")
            chunks = create_text_chunks(curated_content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "document_name": "Curated_QA.md",
                    "document_type": "Curated Q&A",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content": chunk,
                    "content_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                }
                _document_chunks.append(chunk_metadata)
                
                formatted_chunk = f"""Document: Curated Q&A (High Priority)
Chunk {i+1}/{len(chunks)}

{chunk}

---
Source: Curated Q&A Database (Chunk {i+1})"""
                
                temp_file = BytesIO(formatted_chunk.encode('utf-8'))
                created_file = client.files.create(
                    file=(f"curated_qa_chunk_{i+1}.txt", temp_file),
                    purpose="assistants"
                )
                
                file_ids.append(created_file.id)
                chunk_id += 1
            
            print(f"Successfully processed curated Q&A -> {len(chunks)} chunks")
    except Exception as e:
        print(f"Error processing curated Q&A: {e}")
    
    # Process PDF files
    pdf_files = list(documents_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing PDF: {pdf_file.name}")
            
            # Extract text from PDF
            text_content = extract_text_from_pdf(str(pdf_file))
            
            if text_content.strip():
                # Create chunks from the document
                chunks = create_text_chunks(text_content)
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    # Create chunk metadata
                    chunk_metadata = {
                        "chunk_id": chunk_id,
                        "document_name": pdf_file.name,
                        "document_type": "PDF",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content": chunk,
                        "content_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                    }
                    _document_chunks.append(chunk_metadata)
                    
                    # Create a formatted chunk for OpenAI
                    formatted_chunk = f"""Document: {pdf_file.name}
Chunk {i+1}/{len(chunks)}

{chunk}

---
Source: {pdf_file.name} (Chunk {i+1})"""
                    
                    # Upload chunk to OpenAI
                    temp_file = BytesIO(formatted_chunk.encode('utf-8'))
                    created_file = client.files.create(
                        file=(f"{pdf_file.stem}_chunk_{i+1}.txt", temp_file),
                        purpose="assistants"
                    )
                    
                    file_ids.append(created_file.id)
                    chunk_id += 1
                
                print(f"Successfully processed {pdf_file.name} -> {len(chunks)} chunks")
            else:
                print(f"No text content extracted from {pdf_file.name}")
                
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    
    # Process existing markdown files
    md_files = list(documents_path.glob("*.md"))
    for md_file in md_files:
        try:
            print(f"Processing Markdown: {md_file.name}")
            with open(md_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            chunks = create_text_chunks(text_content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "document_name": md_file.name,
                    "document_type": "Markdown",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content": chunk,
                    "content_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                }
                _document_chunks.append(chunk_metadata)
                
                formatted_chunk = f"""Document: {md_file.name}
Chunk {i+1}/{len(chunks)}

{chunk}

---
Source: {md_file.name} (Chunk {i+1})"""
                
                temp_file = BytesIO(formatted_chunk.encode('utf-8'))
                created_file = client.files.create(
                    file=(f"{md_file.stem}_chunk_{i+1}.txt", temp_file),
                    purpose="assistants"
                )
                
                file_ids.append(created_file.id)
                chunk_id += 1
            
            print(f"Successfully processed {md_file.name} -> {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")
    
    print(f"Total chunks created: {len(_document_chunks)}")
    return file_ids, _document_chunks

def find_relevant_chunks(query: str, max_chunks: int = 5) -> List[Dict[str, Any]]:
    """Find the most relevant document chunks for a given query."""
    query_lower = query.lower()
    
    # Simple relevance scoring based on keyword matching
    chunk_scores = []
    
    for chunk in _document_chunks:
        content_lower = chunk["content"].lower()
        score = 0
        
        # Count keyword matches
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                score += content_lower.count(word)
        
        # Bonus for exact phrase matches
        if query_lower in content_lower:
            score += 10
        
        # Bonus for document type relevance
        doc_name_lower = chunk["document_name"].lower()
        if any(keyword in doc_name_lower for keyword in ["how-to", "guide", "manual"]):
            score += 2
        
        if score > 0:
            chunk_scores.append((score, chunk))
    
    # Sort by score and return top chunks
    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in chunk_scores[:max_chunks]]

def start_mcl_knowledge_base() -> str:
    """Initialize the MCL knowledge base with all documents."""
    global _mcl_vector_store_id
    
    print("Starting MCL knowledge base initialization...")
    try:
        # Process all MCL documents with chunk tracking
        file_ids, chunks = process_mcl_documents_with_chunks()
        
        if not file_ids:
            print("WARNING: No files were processed for the knowledge base")
            return None
        
        print(f"Creating vector store with {len(file_ids)} file chunks...")
        vector_store = client.vector_stores.create(
            name="mcl_knowledge_base_chunked",
            file_ids=file_ids
        )
        
        if not vector_store.id:
            raise ValueError("Failed to create a vector store with a valid ID.")

        _mcl_vector_store_id = vector_store.id
        print(f"MCL knowledge base setup complete. Vector Store ID: {vector_store.id}")
        print(f"Total document chunks indexed: {len(chunks)}")
        
        return vector_store.id

    except Exception as e:
        print(f"FATAL: An error occurred during MCL knowledge base setup: {e}")
        return None

def get_mcl_ai_response(messages_input: List[Dict[str, Any]]) -> Any:
    """Get AI response for MCL-related queries with source attribution."""
    
    # Extract the latest user message for relevance search
    latest_user_message = ""
    for msg in reversed(messages_input):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "")
            break
    
    # Find relevant document chunks
    relevant_chunks = find_relevant_chunks(latest_user_message, max_chunks=5)
    
    # Create context from relevant chunks
    context_parts = []
    sources = []
    
    for chunk in relevant_chunks:
        context_parts.append(f"[From {chunk['document_name']}, Chunk {chunk['chunk_index']+1}]:\n{chunk['content']}")
        source_info = f"{chunk['document_name']} (Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']})"
        if source_info not in sources:
            sources.append(source_info)
    
    context = "\n\n" + "\n\n---\n\n".join(context_parts) if context_parts else ""
    
    system_prompt = f"""You are "MCL Assistant," an expert AI assistant for the MCL (Mobile Checklist) application. 

    IMPORTANT: You must base your answers ONLY on the provided document excerpts below. If the information is not in the provided excerpts, clearly state that you don't have that specific information in the available documents.

    Available Document Excerpts:
    {context}

    Guidelines:
    - Answer based ONLY on the provided document excerpts
    - Always cite which document(s) you're referencing
    - If information is not in the excerpts, say so clearly
    - Provide step-by-step instructions when available in the documents
    - Be specific and detailed based on the actual documentation
    - At the end of your response, list the sources you used

    Remember: Only use information from the document excerpts provided above."""
    
    final_messages = [{"role": "system", "content": system_prompt}] + messages_input

    print(f"Sending messages to MCL AI with {len(relevant_chunks)} relevant document chunks")
    
    try:
        response = client.chat.completions.create( 
            model="gpt-4o",
            messages=final_messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        # Enhance the response with source information
        original_content = response.choices[0].message.content
        
        if sources:
            sources_text = "\n\n📚 **Sources:**\n" + "\n".join([f"• {source}" for source in sources])
            enhanced_content = original_content + sources_text
            
            # Create enhanced response
            response.choices[0].message.content = enhanced_content
        
        print(f"MCL AI response received successfully with {len(sources)} sources")
        return response
        
    except Exception as e:
        print(f"Error in MCL AI response: {e}")
        
        # Create a fallback response
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)
        
        class MockMessage:
            def __init__(self, content):
                self.role = "assistant"
                self.content = content
                self.tool_calls = None
        
        fallback_content = """I apologize, but I'm currently experiencing technical difficulties accessing the MCL knowledge base. 
        
        Please try your question again, or contact your system administrator for technical support."""
        
        return MockResponse(fallback_content)

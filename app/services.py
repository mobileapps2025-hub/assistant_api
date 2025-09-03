import json
import os
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO
from pathlib import Path
from app.config import client 
from datetime import datetime
from typing import List, Dict, Any

# Global variable to store the vector store ID
_mcl_vector_store_id: str = None

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

def process_mcl_documents() -> List[str]:
    """Process all MCL documents in the documents folder and return file IDs."""
    documents_path = Path("app/documents")
    file_ids = []
    
    print("Processing MCL documents...")
    
    # Process PDF files
    pdf_files = list(documents_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing PDF: {pdf_file.name}")
            
            # Extract text from PDF
            text_content = extract_text_from_pdf(str(pdf_file))
            
            if text_content.strip():
                # Create a temporary text file with the extracted content
                temp_content = f"# {pdf_file.stem}\n\n{text_content}"
                temp_file = BytesIO(temp_content.encode('utf-8'))
                
                # Upload to OpenAI
                created_file = client.files.create(
                    file=(f"{pdf_file.stem}.txt", temp_file),
                    purpose="assistants"
                )
                
                file_ids.append(created_file.id)
                print(f"Successfully processed {pdf_file.name} -> File ID: {created_file.id}")
            else:
                print(f"No text content extracted from {pdf_file.name}")
                
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    
    # Process existing markdown files
    md_files = list(documents_path.glob("*.md"))
    for md_file in md_files:
        try:
            print(f"Processing Markdown: {md_file.name}")
            with open(md_file, "rb") as file_content:
                created_file = client.files.create(
                    file=file_content,
                    purpose="assistants"
                )
                file_ids.append(created_file.id)
                print(f"Successfully processed {md_file.name} -> File ID: {created_file.id}")
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")
    
    # Process text files if any
    txt_files = list(documents_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            print(f"Processing Text: {txt_file.name}")
            with open(txt_file, "rb") as file_content:
                created_file = client.files.create(
                    file=file_content,
                    purpose="assistants"
                )
                file_ids.append(created_file.id)
                print(f"Successfully processed {txt_file.name} -> File ID: {created_file.id}")
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")
    
    return file_ids

def start_mcl_knowledge_base() -> str:
    """Initialize the MCL knowledge base with all documents."""
    global _mcl_vector_store_id
    
    print("Starting MCL knowledge base initialization...")
    try:
        # Process all MCL documents
        file_ids = process_mcl_documents()
        
        if not file_ids:
            print("WARNING: No files were processed for the knowledge base")
            return None
        
        print(f"Creating vector store with {len(file_ids)} files...")
        vector_store = client.vector_stores.create(
            name="mcl_knowledge_base",
            file_ids=file_ids
        )
        
        if not vector_store.id:
            raise ValueError("Failed to create a vector store with a valid ID.")

        _mcl_vector_store_id = vector_store.id
        print(f"MCL knowledge base setup complete. Vector Store ID: {vector_store.id}")
        
        return vector_store.id

    except Exception as e:
        print(f"FATAL: An error occurred during MCL knowledge base setup: {e}")
        return None

def get_mcl_ai_response(messages_input: List[Dict[str, Any]]) -> Any:
    """Get AI response for MCL-related queries using the knowledge base."""
    global _mcl_vector_store_id
    
    system_prompt = """You are "MCL Assistant," an expert AI assistant for the MCL (Mobile Checklist) application. 
    Your role is to provide comprehensive, accurate, and helpful information about the MCL app based on the extensive knowledge base of documents.

    Key Guidelines:
    - Provide precise, detailed answers based on the MCL documentation
    - If asked about features, guides, or troubleshooting, reference the relevant documentation
    - Explain step-by-step processes clearly when providing instructions
    - Cover all aspects of MCL including mobile app usage, dashboard functionality, tablet usage, checklists, questions, roles, and business processes
    - If you don't find specific information in the knowledge base, clearly state that and offer to help with related topics you do know about
    - Maintain a helpful, professional, and conversational tone
    - Focus specifically on MCL-related topics, but you can also provide information about business cases and technical updates if they are related to MCL
    
    Remember: You have access to comprehensive MCL documentation including user guides, release notes, technical updates, business cases, and training materials."""
    
    final_messages = [{"role": "system", "content": system_prompt}] + messages_input

    print(f"Sending messages to MCL AI: {len(final_messages)} messages")
    
    # Use the standard chat completions API - more reliable and compatible
    try:
        response = client.chat.completions.create( 
            model="gpt-4o",
            messages=final_messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        print(f"MCL AI response received successfully")
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
        
        However, I can still help you with general MCL questions. MCL (Mobile Checklist) is an application designed for creating and managing checklists across different devices including mobile phones, tablets, and web dashboards.
        
        For specific technical support, please refer to your MCL documentation or contact your system administrator."""
        
        return MockResponse(fallback_content)

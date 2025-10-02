import json
import requests
from io import BytesIO
import os
import PyPDF2
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("PyMuPDF not available, will use PyPDF2 only for PDF processing")
from pathlib import Path
from app.config import client, AsyncSessionLocal
from app.models import EventsBetweenWeeksRequest
from app.utils import FUNCTION_TOOLS 
from app.clients.api_client import APIClient
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Tuple
import hashlib
import asyncio

# Global API client instance - will be set by the middleware for Spotplan
_api_client: APIClient = None

# Global variables for MCL knowledge base
_mcl_vector_store_id: str = None
_spotplan_vector_store_id: str = None
_document_chunks: List[Dict[str, Any]] = []

def set_api_client_token(token: str):
    """Set the global API client with the provided token for Spotplan operations."""
    global _api_client
    _api_client = APIClient(token)

# --- Spotplan API Functions ---

async def get_stores():
    return await _api_client.make_request("GET", "Store/GetUserStores")

async def get_events_between_weeks(store_id: str, starting_week: int, ending_week: int, year: int):
    json_body = {
        "StoreId": store_id,
        "StartingWeek": starting_week,
        "EndingWeek": ending_week,
        "Year": year
    }
    print(f"Requesting events between weeks with body: {json_body}")
    return await _api_client.make_request("POST", "Event/GetEventsBetweenWeeks", json_body=json_body)

async def get_unplanned_events_between_weeks(store_id: str, starting_week: int, ending_week: int, year: int):
    json_body = {
        "StoreId": store_id,
        "StartingWeek": starting_week,
        "EndingWeek": ending_week,
        "Year": year
    }
    print(f"Requesting unplanned events between weeks with body: {json_body}")
    return await _api_client.make_request("POST", "Event/GetUnplannedEventsBetweenWeeks", json_body=json_body)

async def get_week_events(store_id: str, week: int, year: int):
    query_params = {
        "idStore": store_id,
        "week": week,
        "year": year
    }
    print(f"Requesting week events with body: {query_params}")
    return await _api_client.make_request("GET", "Event/GetWeekEvents", query_params=query_params)

async def get_events_by_name(event_name):
    query_params = {"name": event_name}
    return await _api_client.make_request("GET", "Event/GetEventsByName", query_params=query_params)
        
async def get_event_details(event_id):
    query_params = {"id": event_id}
    return await _api_client.make_request("GET", "Event/GetEvent", query_params=query_params)

async def get_store_unplanned_events(store_id):
    query_params = {"idStore": store_id}
    return await _api_client.make_request("GET", "Event/GetStoreUnplannedEvents", query_params=query_params)

async def get_company_unplanned_events():
    return await _api_client.make_request("POST", "Event/GetCompanyUnplannedEvents")

async def get_store_sales_areas(store_id):
    query_params = {"storeId": store_id}
    return await _api_client.make_request("GET", "Store/GetStoreSalesAreas", query_params=query_params)

async def get_unplanned_sales_areas_on_week(store_id, year, week):
    query_params = {"storeId": store_id, "year": year, "week": week}
    return await _api_client.make_request("GET", "Store/GetUnplannedSalesAreasOnWeek", query_params=query_params)

# --- Spotplan Knowledge Base Functions ---

def create_file(openai_client, file_path):
    print(f"Attempting to create OpenAI file from: {file_path}")
    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            response = requests.get(file_path, timeout=30)
            response.raise_for_status()  

            file_content = BytesIO(response.content)
            file_name = os.path.basename(file_path) or "downloaded_knowledge_file.pdf"
            
            created_file = openai_client.files.create(
                file=(file_name, file_content),
                purpose="assistants"
            )

        except requests.RequestException as e:
            print(f"Error downloading file from URL {file_path}: {e}")
            raise
    else:
        if not os.path.exists(file_path):
            print(f"Local file {file_path} not found.")
            raise FileNotFoundError(f"Local file {file_path} not found.")
        with open(file_path, "rb") as file_content_stream:
            created_file = openai_client.files.create(
                file=file_content_stream,
                purpose="assistants"
            )
    print(f"File created successfully with ID: {created_file.id}")
    return created_file.id

def start_spotplan_knowledge_base():
    global _spotplan_vector_store_id
    print("Starting Spotplan knowledge base initialization...")
    try:
        file_path = "spotplan_guide.md" 
        print(f"Creating file object for: {file_path}")
        
        with open(file_path, "rb") as file_content_stream:
            knowledge_file = client.files.create(
                file=file_content_stream,
                purpose="assistants"
            )
        
        print(f"File created successfully with ID: {knowledge_file.id}")

        print(f"Creating vector store named 'spotplan_knowledge_base'...")
        vector_store = client.vector_stores.create(
            name="spotplan_knowledge_base",
            file_ids=[knowledge_file.id]
        )
        
        if not vector_store.id:
            raise ValueError("Failed to create a vector store with a valid ID.")

        _spotplan_vector_store_id = vector_store.id
        print(f"Spotplan knowledge base setup complete. Vector Store ID: {vector_store.id} is ready.")
        
        return vector_store.id

    except Exception as e:
        print(f"FATAL: An error occurred during Spotplan knowledge base setup: {e}")
        return None

def get_spotplan_ai_response(messages_input):
    current_year = datetime.now().year
    current_week = datetime.now().isocalendar()[1]  # Get the current week number

    system_prompt = f"""You are "Spotplan Assistant," an expert AI partner for the Spotplan application. 
    Your goal is to help users by calling the available API function tools. Follow the workflow instructions in your function descriptions.

    - **The Golden Rule of Clarification:** If a user's data request is ambiguous, you MUST default to the `get_stores()` workflow to ask the user for clarification.
    - **Use Your Memory:** Before asking the user for information (like a `store_id`).
    - **If user does not specify the year, default to the current year that is {current_year}**
    - **If user does not specify a week, default to the current week of the year that is {current_week}.**
    """
    
    final_messages = [{"role": "system", "content": system_prompt}] + messages_input

    print(f"Sending messages to Spotplan AI for function-calling: {final_messages}")
    
    response = client.chat.completions.create( 
        model="gpt-4o",
        messages=final_messages, 
        tools=FUNCTION_TOOLS,
        tool_choice="auto"
    )
    print(f"Received Spotplan AI response object: {type(response)}") 
    return response

# --- MCL Knowledge Base Functions ---

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using PyMuPDF for better text extraction."""
    text_content = ""
    
    # Try PyMuPDF first if available (better text extraction)
    if HAS_PYMUPDF:
        try:
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
    
    # Fallback to PyPDF2 (or use directly if PyMuPDF not available)
    try:
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
        if not AsyncSessionLocal:
            return ""
        async with AsyncSessionLocal() as db:
            from app.database.repositories import CuratedQaRepository
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
    if documents_path.exists():
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
    """Find the most relevant document chunks for a given query.
    
    This function now uses GPT-based translation for better cross-language matching.
    """
    print(f"\n[CHUNK SEARCH] Starting search for query: '{query}'")
    
    # Detect language and translate if needed
    detected_lang = detect_language(query)
    search_query = query
    
    if detected_lang != 'en':
        print(f"[CHUNK SEARCH] Detected non-English query ({detected_lang}), translating for better matching...")
        translated_query = translate_query_to_english(query, detected_lang)
        # Use translated query for searching (English documents)
        search_query = translated_query
        print(f"[CHUNK SEARCH] Will search with translated query: '{search_query}'")
    
    query_lower = search_query.lower()
    print(f"[CHUNK SEARCH] Searching with: '{query_lower}'")
    
    # Simple relevance scoring based on keyword matching
    chunk_scores = []
    
    for chunk in _document_chunks:
        content_lower = chunk["content"].lower()
        score = 0
        
        # Count keyword matches
        query_words = query_lower.split()
        matched_words = []
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                word_count = content_lower.count(word)
                score += word_count
                if word_count > 0:
                    matched_words.append(word)
        
        # Bonus for exact phrase matches
        if query_lower in content_lower:
            score += 10
            matched_words.append("[EXACT_PHRASE]")
        
        # Bonus for document type relevance
        doc_name_lower = chunk["document_name"].lower()
        if any(keyword in doc_name_lower for keyword in ["how-to", "guide", "manual", "creating"]):
            score += 2
        
        # Extra bonus for "Creating Checklists" document if query is about creating
        if "creat" in query_lower and "creating" in doc_name_lower and "checklist" in doc_name_lower:
            score += 5
            print(f"[CHUNK SEARCH] Bonus for 'Creating Checklists' document")
        
        if score > 0:
            chunk_scores.append((score, chunk, matched_words))
            print(f"[CHUNK SEARCH] Document: {chunk['document_name']}, Chunk {chunk['chunk_index']+1}, Score: {score}, Matched: {matched_words[:5]}")
    
    # Sort by score and return top chunks
    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk, words in chunk_scores[:max_chunks]]
    
    print(f"[CHUNK SEARCH] Found {len(chunk_scores)} chunks with matches, returning top {len(top_chunks)}")
    for i, (score, chunk, words) in enumerate(chunk_scores[:max_chunks]):
        print(f"[CHUNK SEARCH] Top {i+1}: {chunk['document_name']} (Chunk {chunk['chunk_index']+1}) - Score: {score}")
    
    if len(top_chunks) == 0:
        print(f"[CHUNK SEARCH WARNING] No relevant chunks found! This may cause poor responses.")
    
    return top_chunks

def start_mcl_knowledge_base() -> str:
    """Initialize the MCL knowledge base with all documents."""
    global _mcl_vector_store_id
    
    print("Starting MCL knowledge base initialization...")
    try:
        # Process all MCL documents with chunk tracking
        file_ids, chunks = process_mcl_documents_with_chunks()
        
        if not file_ids:
            print("WARNING: No files were processed for the MCL knowledge base")
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

def detect_language(text: str) -> str:
    """Detect the language of the user's query using GPT-4.
    Returns language code: 'de' for German, 'en' for English, etc.
    """
    if not text or len(text.strip()) < 2:
        return 'en'  # Default to English
    
    try:
        # Quick heuristic check first (fast path)
        text_lower = text.lower()
        
        # Check for German-specific characters
        if any(char in text_lower for char in ['ä', 'ö', 'ü', 'ß']):
            return 'de'
        
        # Check for obvious German words
        strong_german_words = ['ich', 'kannst', 'mir', 'checkliste', 'erstellen', 'anlegen', 
                               'wie', 'was', 'warum', 'erklären', 'hilfe', 'bitte']
        if any(word in text_lower for word in strong_german_words):
            return 'de'
        
        # If still uncertain and text is long enough, use GPT for detection
        if len(text.split()) > 3:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Faster and cheaper for language detection
                messages=[
                    {"role": "system", "content": "Detect the language of the following text. Respond with ONLY the ISO 639-1 language code (e.g., 'en' for English, 'de' for German, 'es' for Spanish, etc.). No explanation."},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=10
            )
            detected_lang = response.choices[0].message.content.strip().lower()
            return detected_lang if detected_lang in ['de', 'en', 'es', 'fr', 'it'] else 'en'
        
        return 'en'  # Default to English
        
    except Exception as e:
        print(f"Error detecting language: {e}")
        return 'en'  # Default to English on error

def translate_query_to_english(query: str, source_language: str) -> str:
    """Translate a user query to English for document search.
    
    Args:
        query: The user's query in any language
        source_language: The detected source language code
    
    Returns:
        English translation of the query
    """
    if source_language == 'en':
        return query  # Already in English
    
    try:
        print(f"🔄 Translating query from {source_language} to English...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and accurate for translation
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional translator. Translate the following text to English. Preserve the meaning and intent. Respond with ONLY the English translation, no explanations or additional text."
                },
                {
                    "role": "user", 
                    "content": f"Translate to English: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        translated = response.choices[0].message.content.strip()
        print(f"✅ Translation: '{query}' → '{translated}'")
        return translated
        
    except Exception as e:
        print(f"❌ Error translating query: {e}")
        return query  # Return original on error

def get_mcl_ai_response(messages_input: List[Dict[str, Any]]) -> Any:
    """Get AI response for MCL-related queries with source attribution."""
    
    print("\n" + "="*80)
    print("[MCL AI] Starting MCL AI Response Generation")
    print("="*80)
    
    # Extract the latest user message for relevance search
    latest_user_message = ""
    for msg in reversed(messages_input):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "")
            break
    
    print(f"[MCL AI] User message: '{latest_user_message}'")
    
    # Find relevant document chunks
    relevant_chunks = find_relevant_chunks(latest_user_message, max_chunks=5)
    
    # Create context from relevant chunks
    context_parts = []
    sources = []
    
    print(f"\n[MCL AI] Building context from {len(relevant_chunks)} relevant chunks:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"[MCL AI] Context Chunk {i+1}: {chunk['document_name']} (Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']})")
        print(f"[MCL AI] Preview: {chunk['content'][:150]}...")
        
        context_parts.append(f"[From {chunk['document_name']}, Chunk {chunk['chunk_index']+1}]:\n{chunk['content']}")
        source_info = f"{chunk['document_name']} (Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']})"
        if source_info not in sources:
            sources.append(source_info)
    
    context = "\n\n" + "\n\n---\n\n".join(context_parts) if context_parts else ""
    
    if not context:
        print("[MCL AI WARNING] No context available! AI will have no document excerpts to work with.")
    else:
        print(f"[MCL AI] Total context length: {len(context)} characters")
    
    # Detect user's language
    detected_lang = detect_language(latest_user_message)
    user_language_name = {
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian'
    }.get(detected_lang, 'English')
    
    print(f"\n[MCL AI] Detected language: {user_language_name} ({detected_lang})")
    
    # Create language-specific system prompt
    if detected_lang == 'de':
        print("[MCL AI] Creating German-language system prompt")
        system_prompt = f"""🇩🇪 WICHTIG: Du musst auf DEUTSCH antworten! Der Benutzer stellt eine Frage auf Deutsch.

You are "MCL Assistant," an expert AI assistant for the MCL (Mobile Checklist) application.

⚠️ CRITICAL LANGUAGE RULE: The user is writing in GERMAN. You MUST respond in GERMAN (Deutsch).
- Translate ALL information from the English documents into clear, natural German
- Use proper German terminology:
  * "Checklist" → "Checkliste" or "Prüfliste"
  * "Task" → "Aufgabe"
  * "Question" → "Frage"
  * "Dashboard" → "Dashboard" (same)
  * "Create" → "erstellen" or "anlegen"
  * "User" → "Benutzer"
  * "Wizard" → "Assistent"
  * "Click" → "Klicken"
  * "Button" → "Schaltfläche" or "Button"
- Write complete sentences in German with proper grammar
- Provide step-by-step instructions in German

Available Document Excerpts (in English - you must translate):
{context}

Guidelines:
- Answer based ONLY on the provided document excerpts above
- TRANSLATE the information into German before presenting it
- Always cite which document(s) you're referencing (in German: "Quelle:" or "Aus:")
- If information is not in the excerpts, say clearly in German: "Diese Information finde ich nicht in den verfügbaren Dokumenten."
- Provide step-by-step instructions in German when available
- Be specific and detailed based on the actual documentation
- At the end of your response, list the sources in German: "📚 Quellen:"

REMEMBER: Your ENTIRE response must be in GERMAN (Deutsch)!"""
    else:
        print("[MCL AI] Creating English-language system prompt")
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
- Pay special attention to documents about "Creating Checklists" when users ask about checklist creation

Remember: Only use information from the document excerpts provided above."""
    
    final_messages = [{"role": "system", "content": system_prompt}] + messages_input
    
    print(f"\n[MCL AI] System prompt length: {len(system_prompt)} characters")
    print(f"[MCL AI] Total messages to send: {len(final_messages)}")
    print(f"[MCL AI] Sending request to OpenAI (Language: {user_language_name})...")

    try:
        response = client.chat.completions.create( 
            model="gpt-4o",
            messages=final_messages,
            temperature=0.2,  # Slightly higher for more natural translations
            max_tokens=2000
        )
        
        print(f"[MCL AI] Received response from OpenAI")
        
        # Enhance the response with source information
        original_content = response.choices[0].message.content
        
        print(f"[MCL AI] Response content length: {len(original_content)} characters")
        print(f"[MCL AI] Response preview: {original_content[:200]}...")
        
        if sources:
            # Use language-appropriate source header
            source_headers = {
                'de': "\n\n📚 **Quellen:**\n",
                'en': "\n\n📚 **Sources:**\n",
                'es': "\n\n📚 **Fuentes:**\n",
                'fr': "\n\n📚 **Sources:**\n"
            }
            sources_header = source_headers.get(detected_lang, source_headers['en'])
            sources_text = sources_header + "\n".join([f"• {source}" for source in sources])
            
            enhanced_content = original_content + sources_text
            
            # Create enhanced response
            response.choices[0].message.content = enhanced_content
        
        print(f"[MCL AI] ✓ Response generation completed successfully")
        print(f"[MCL AI] Sources attached: {len(sources)}")
        print("="*80 + "\n")
        
        return response
        
    except Exception as e:
        print(f"[MCL AI ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        
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
        
        # Create language-appropriate fallback message
        fallback_messages = {
            'de': """Entschuldigung, aber ich habe derzeit technische Schwierigkeiten beim Zugriff auf die MCL-Wissensdatenbank.
            
            Bitte stellen Sie Ihre Frage erneut, oder wenden Sie sich für technischen Support an Ihren Systemadministrator.""",
            'en': """I apologize, but I'm currently experiencing technical difficulties accessing the MCL knowledge base.
            
            Please try your question again, or contact your system administrator for technical support.""",
            'es': """Lo siento, pero actualmente tengo dificultades técnicas para acceder a la base de conocimientos de MCL.
            
            Por favor, intente su pregunta nuevamente o contacte a su administrador del sistema para soporte técnico."""
        }
        
        fallback_content = fallback_messages.get(detected_lang, fallback_messages['en'])
        return MockResponse(fallback_content)

def get_ai_format(messages_input, request: BaseModel):
    final_messages = [messages_input]

    print(f"Sending messages to AI for function-calling: {final_messages}")

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=final_messages,
        response_format=request.model_dump()
    )
    print(f"Received AI response object: {type(response)}")
    return response

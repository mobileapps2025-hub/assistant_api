# MCL Assistant API

An AI-powered knowledge base assistant for the **MCL (Mobile Checklist)** application. This intelligent assistant uses advanced RAG (Retrieval-Augmented Generation) techniques to answer questions based on MCL documentation, and now includes **Vision AI** capabilities for screenshot analysis.

## Features

### Text-Based Assistant
- 🌍 **Multilingual Support**: Automatically detects and responds in German, English, Spanish, French, and Italian
- 🔍 **Advanced Semantic Search**: Uses sentence transformers and FAISS for intelligent document retrieval
- 📚 **Multi-format Support**: Processes PDF, DOCX, PPTX, and Markdown documents
- 🎯 **Query Expansion**: Generates alternative phrasings for better search coverage
- 🤖 **GPT-4o Powered**: Uses OpenAI's latest GPT-4o model for accurate responses
- ⚡ **Fast & Efficient**: Hybrid search combining semantic and keyword matching
- 🧭 **Situational Guardrails**: Detects app/web context, device type, and confidence, then asks for clarification when signals are missing

### Vision Assistant (NEW!)
- 👁️ **Screenshot Analysis**: Upload MCL App screenshots for contextual help
- 🎯 **Screen Identification**: Automatically identifies which MCL screen is shown
- 📝 **Step-by-Step Guidance**: Provides detailed instructions based on visual context
- 💬 **Multi-Query Support**: Ask multiple questions about the same screenshot
- 🔒 **Secure & Private**: Images processed via OpenAI's secure API

## Architecture

### Document Processing Pipeline
1. **Document Ingestion**: Supports PDF (PyPDF2/PyMuPDF), DOCX, PPTX, and Markdown
2. **Text Chunking**: Splits documents into 1200-character chunks with 300-character overlap
3. **Embedding Creation**: Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings
4. **Vector Indexing**: FAISS IndexFlatIP for efficient cosine similarity search

### Query Processing
1. **Language Detection**: Automatic detection using heuristics + GPT fallback
2. **Query Translation**: Non-English queries translated to English for search
3. **Query Expansion**: GPT-4o-mini generates 2-3 alternative phrasings
4. **Hybrid Search**: Combines semantic search (FAISS) with enhanced keyword matching
5. **Re-ranking**: Results scored using 60% keyword + 40% semantic weights

### Response Generation
1. **Context Building**: Top 15 chunks assembled with source attribution
2. **Situational Awareness**: Heuristic analysis captures interface/OS assumptions and enforces guardrails
3. **Prompt Engineering**: Language-specific system prompts with reasoning guidelines
4. **GPT-4o Generation**: Context-aware response with cited sources
5. **Source Attribution**: Automatic listing of referenced documents

## Installation

### Prerequisites
- Python 3.11 or higher
- OpenAI API key

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Static
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
Create a `.env` file:
```properties
OPENAI_API_KEY="your-openai-api-key-here"
```

5. **Add MCL documents**
Place your MCL documentation files in `app/documents/` directory. Supported formats:
- PDF (.pdf)
- Word Documents (.docx)
- PowerPoint (.pptx)
- Markdown (.md)

## Usage

### Start the Server
```bash
python -m uvicorn app.main:app --reload
```

Server will start on `http://localhost:8000`

### Vision Assistant (Screenshot Analysis)

For analyzing MCL App screenshots, see the **[Vision Assistant Documentation](VISION_ASSISTANT_README.md)**.

**Quick Start:**
```python
from app.vision_assistant import MCLVisionAssistant

# Initialize assistant
assistant = MCLVisionAssistant()

# Analyze a screenshot
result = assistant.analyze_screenshot(
    image_path="./screenshots/mcl-dashboard.png",
    user_query="What can I do on this screen?"
)

print(result["response"])
```

**Interactive Demo:**
```bash
python demo_vision.py
```

**Simple Example:**
```bash
python simple_vision_example.py
```

For complete documentation, examples, and API reference, see **[VISION_ASSISTANT_README.md](VISION_ASSISTANT_README.md)**.

### Text-Based API Endpoints

#### 1. Chat (Main Endpoint)
```http
POST /api/chat
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "content": "How can I create a new checklist?"
    }
  ]
}
```

**Response:**
```json
{
  "response": "To create a new checklist in MCL...",
  "response_id": "resp_a1b2c3d4",
  "sources": ["Creating Checklists EN v5 05.22.2020_compressed.pdf"],
  "app_type": "mcl"
}
```

#### 2. Health Check
```http
GET /health
```

Returns knowledge base status and statistics.

#### 3. Search Chunks
```http
POST /api/search
Content-Type: application/json

{
  "query": "checklist creation",
  "max_results": 10
}
```

Returns relevant document chunks with previews.

#### 4. List Chunks
```http
GET /api/chunks
```

Returns all processed document chunks grouped by document.

## Document Guidelines

### Naming Conventions
MCL documents should contain one of these indicators in the filename:
- `mcl`
- `checklist`
- `quiz`
- `question`
- `dashboard`
- `tablet`
- `phone`
- `aufgabe` (German)
- `creating`
- `how-to`

### Excluded Documents
Files containing these terms are automatically excluded:
- `spotplan`
- `kb_assistant_complete_export`

### Best Practices
1. **Clear Structure**: Use headings and sections for better chunking
2. **Consistent Terminology**: Use standard MCL terminology across documents
3. **Comprehensive Content**: Include step-by-step instructions, screenshots descriptions
4. **Regular Updates**: Keep documentation current with app versions

## Advanced Features

### Query Expansion
The system automatically generates alternative phrasings:
- Original: "How can I create a checklist?"
- Variant 1: "What are the steps to make a checklist?"
- Variant 2: "How do I go about creating a checklist?"

### Multilingual Support
Automatic detection and response in user's language:
```json
{
  "messages": [
    {"role": "user", "content": "Wie kann ich eine Checkliste erstellen?"}
  ]
}
```

Response will be in German with translated information from English documents.

### Semantic Search
Uses FAISS for efficient vector similarity search:
- Embeds queries using sentence-transformers
- Searches 99+ document chunks in milliseconds
- Returns top 15 most relevant chunks
- Scores using cosine similarity (0.0 - 1.0)

## Development

### Project Structure
```
Static/
├── app/
│   ├── main.py              # FastAPI application
│   ├── services.py          # Core AI services
│   ├── models.py            # Pydantic models
│   ├── config.py            # Configuration
│   ├── vision_assistant.py  # NEW: Vision AI for screenshots
│   ├── documents/           # MCL documentation files
│   └── __init__.py
├── demo_vision.py           # NEW: Interactive vision demo
├── simple_vision_example.py # NEW: Simple vision example
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── VISION_ASSISTANT_README.md  # NEW: Vision assistant docs
```

### Key Functions (services.py)

- `process_mcl_documents_with_enhanced_chunking()`: Process all MCL documents
- `find_relevant_chunks(query, max_chunks)`: Advanced RAG retrieval
- `expand_query_with_variants(query)`: Generate query alternatives
- `semantic_search_chunks(query, max_results)`: FAISS semantic search
- `detect_language(text)`: Detect user's language
- `translate_query_to_english(query, lang)`: Translate for search
- `get_mcl_ai_response(messages)`: Generate AI response

### Vision Assistant Functions (vision_assistant.py)

- `MCLVisionAssistant()`: Initialize vision assistant
- `analyze_screenshot(image_path, user_query)`: Complete workflow (recommended)
- `get_or_create_assistant(name, instructions, model)`: Setup assistant
- `upload_image_for_vision(image_path)`: Upload screenshot
- `create_thread_and_add_message(text, file_id)`: Create multimodal conversation
- `run_assistant_and_wait(thread_id, assistant_id)`: Execute analysis
- `get_assistant_response(thread_id)`: Retrieve results

See **[VISION_ASSISTANT_README.md](VISION_ASSISTANT_README.md)** for detailed API reference.

### Testing

Test the API with sample queries:
```bash
# English query
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"How do I create a checklist?"}]}'

# German query
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Wie erstelle ich eine Checkliste?"}]}'
```

## Configuration

### Environment Variables
```properties
OPENAI_API_KEY=sk-...         # Required: OpenAI API key
```

### Customization

**Chunk Size** (services.py line ~500):
```python
chunk_size = 1200  # Characters per chunk
overlap = 300      # Overlap between chunks
```

**Max Chunks** (services.py line ~770):
```python
relevant_chunks = find_relevant_chunks(query, max_chunks=15)
```

**Semantic Model** (services.py line ~205):
```python
_mcl_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Performance

### Benchmarks
- **Document Processing**: 19 documents → 99 chunks in ~30 seconds
- **Embedding Creation**: 99 chunks → FAISS index in ~8 seconds
- **Query Processing**: < 2 seconds per query
- **Semantic Search**: < 100ms for 99 chunks

### Resource Usage
- **Memory**: ~500MB (with embeddings loaded)
- **Disk**: ~50MB (documents + embeddings)
- **CPU**: Moderate (sentence-transformers inference)

## Troubleshooting

### Issue: "No document chunks available"
**Solution**: Ensure MCL documents are placed in `app/documents/` and follow naming conventions.

### Issue: "Semantic search not available"
**Solution**: Install sentence-transformers and faiss:
```bash
pip install sentence-transformers faiss-cpu
```

### Issue: "DOCX/PPTX processing failed"
**Solution**: Install document processing libraries:
```bash
pip install python-docx python-pptx
```

### Issue: Emoji encoding errors on Windows
**Solution**: Use UTF-8 code page:
```bash
chcp 65001
```

## Contributing

### Adding New Document Types
1. Add extraction function in `services.py`
2. Update `extract_text_from_file()` to handle new extension
3. Test with sample documents

### Improving Search Quality
1. Adjust chunk size and overlap
2. Experiment with different embedding models
3. Tune keyword vs semantic weighting in re-ranking

## License

[Your License Here]

## Support

For issues or questions:
- GitHub Issues: [Your Repo]
- Email: [Your Email]

## Changelog

### v3.1.0 (Current)
- ✅ **NEW: Vision Assistant** - Screenshot analysis with GPT-4o vision
- ✅ **NEW: Interactive demos** - `demo_vision.py` and `simple_vision_example.py`
- ✅ **NEW: Complete documentation** - `VISION_ASSISTANT_README.md`
- ✅ **NEW: Multimodal support** - Text + image input for contextual help
- ✅ **NEW: Situational context engine** - Clarification prompts and prompt guardrails to avoid mixed answers
- ✅ Clean, modular code structure
- ✅ Comprehensive error handling
- ✅ Step-by-step workflow support

### v3.0.0
- ✅ Removed Spotplan agent and dependencies
- ✅ Simplified to MCL-only knowledge base
- ✅ Advanced RAG with query expansion
- ✅ Hybrid search (semantic + keyword)
- ✅ Multilingual support (5 languages)
- ✅ Support for PDF, DOCX, PPTX, MD

### v2.0.0
- Unified Spotplan + MCL agent
- Database feedback system

### v1.0.0
- Initial LangChain-based implementation

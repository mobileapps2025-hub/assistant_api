# Phase 1 Verification Guide

## Overview
Phase 1 (Foundation) has been implemented. This includes:
1.  **Weaviate Vector Store**: Replacing FAISS.
2.  **Ingestion Pipeline**: Structure-aware Markdown splitting.
3.  **Hybrid Retrieval**: Vector + Keyword search with Cohere Re-ranking.

## How to Verify

### 1. Prerequisites
Ensure your `.env` file contains the following:
```env
WEAVIATE_URL=http://localhost:8080  # Or your cloud URL
WEAVIATE_API_KEY=your-key           # If using cloud
COHERE_API_KEY=your-key             # Required for re-ranking
OPENAI_API_KEY=your-key             # Required for embeddings
```

### 2. Run the Verification Script
We have created a script `verify_phase_1.py` that tests the entire pipeline.

Run it from the terminal:
```powershell
.\.venv\Scripts\Activate.ps1
python verify_phase_1.py
```

### 3. Expected Output
You should see green checkmarks (✅) for:
- Service Initialization
- Ingestion (it will create a dummy file if needed)
- Retrieval (it should find the test content)

### 4. Troubleshooting
- **Connection Refused**: Ensure Weaviate is running (`docker-compose up -d` if local).
- **API Key Errors**: Check `.env` file.
- **Import Errors**: Ensure dependencies are installed (`pip install -r requirements.txt`).

# Project Summary

## Overview
**Project Name**: MCL AI Assistant
**Type**: Monolith Backend API
**Status**: Active / Brownfield

## Technology Stack
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Database**: SQLAlchemy (Async support implied)
- **AI/ML**: OpenAI GPT-4o, LangChain, FAISS (Vector Store)
- **Deployment**: Uvicorn

## Key Features
1.  **RAG (Retrieval Augmented Generation)**:
    - Ingests PDF documents (`app/documents/`).
    - Provides context-aware answers based on "MCL" specific knowledge (Checklists, Dashboard, Mobile App).
2.  **Vision AI**:
    - Analyzes screenshots and images via `vision_assistant.py`.
    - Supports multimodal chat interactions.
3.  **Feedback Loop**:
    - Captures user feedback on responses (`positive`/`negative`).
    - Stores feedback in a database for quality improvement.
4.  **API First**:
    - RESTful API design.
    - Swagger/OpenAPI documentation available at `/docs`.

## Architecture
- **Entry Point**: `app/main.py` defines the FastAPI application and routes.
- **Data Layer**: `app/database.py` and `app/models.py` define the schema and ORM.
- **Service Layer**: `app/services.py` contains the core business logic and RAG implementation.
- **Vision Layer**: `app/vision_assistant.py` handles image processing and analysis.

## Documentation Status
- **API Contracts**: `docs/api-contracts-backend.md` (Generated)
- **Data Models**: `docs/data-models-backend.md` (Generated)
- **Source Tree**: `docs/source-tree.md` (Generated)

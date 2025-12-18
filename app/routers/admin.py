from fastapi import APIRouter, Depends, HTTPException, status
from app.services.ingestion_service import IngestionService
from app.core.dependencies import get_ingestion_service
from app.models import CuratedQaRequest, CuratedQaResponse
from app.core.database import CuratedQA
from app.core.config import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.optimization.trainer import run_training_pipeline
from pydantic import BaseModel

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)

class IngestResponse(BaseModel):
    success: bool
    message: str
    total_chunks: int = 0
    processed_files: int = 0
    failed_files: int = 0

@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """
    Trigger the ingestion process for all documents in the app/documents directory.
    """
    # Hardcoded path for now, could be configurable
    documents_path = "app/documents"
    
    result = ingestion_service.ingest_all(documents_path)
    
    if not result["success"] and result["processed_files"] == 0 and result["failed_files"] > 0:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
    
    return IngestResponse(
        success=result["success"],
        message=result["message"],
        total_chunks=result.get("total_chunks", 0),
        processed_files=result.get("processed_files", 0),
        failed_files=result.get("failed_files", 0)
    )

@router.post("/curated-qa", response_model=CuratedQaResponse)
async def add_curated_qa(
    qa_request: CuratedQaRequest,
    db: AsyncSession = Depends(get_db),
    vector_store: IngestionService = Depends(get_ingestion_service) # Using IngestionService to access vector store logic if needed, or better, get vector store directly
):
    """
    Add a new curated Q&A pair to the knowledge base for training AND retrieval.
    """
    # 1. Save to Database (for Training)
    new_qa = CuratedQA(
        question=qa_request.question,
        answer=qa_request.answer,
        source_feedback_id=qa_request.source_feedback_id,
        active=True
    )
    
    db.add(new_qa)
    await db.commit()
    await db.refresh(new_qa)
    
    # 2. Add to Vector Store (for Retrieval)
    # We treat the Q&A pair as a small document chunk
    try:
        from app.core.dependencies import get_vector_store_service
        vs_service = get_vector_store_service()
        
        # Create a synthetic chunk
        chunk = {
            "text": f"Question: {qa_request.question}\nAnswer: {qa_request.answer}",
            "header_path": "Curated Knowledge",
            "source": "User Feedback",
            "chunk_index": 0
        }
        
        success = vs_service.add_documents([chunk])
        if success:
            print(f"Successfully indexed curated QA: {new_qa.id}")
        else:
            print(f"Failed to index curated QA: {new_qa.id}")
            
    except Exception as e:
        print(f"Error indexing curated QA: {e}")
        # We don't fail the request if indexing fails, but we log it
    
    return new_qa

@router.post("/train")
async def trigger_training():
    """
    Trigger the DSPy optimization pipeline to improve the agent using curated Q&A.
    """
    try:
        result = await run_training_pipeline()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


from fastapi import APIRouter, Depends, HTTPException, Header, status
from app.services.ingestion_service import IngestionService
from app.core.dependencies import get_ingestion_service
from app.models import CuratedQaRequest, CuratedQaResponse
from app.core.database import CuratedQA
from app.core.config import get_db, ADMIN_API_KEY
from sqlalchemy.ext.asyncio import AsyncSession
from app.optimization.trainer import run_training_pipeline
from app.core.logging import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)


def verify_admin_key(x_admin_key: str = Header(...)):
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key not configured on server"
        )
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key"
        )


class IngestResponse(BaseModel):
    success: bool
    message: str
    total_chunks: int = 0
    processed_files: int = 0
    failed_files: int = 0


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_admin_key)])
async def trigger_ingestion(
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Trigger ingestion for all documents in the app/documents directory."""
    documents_path = "app/documents"
    logger.info(f"[ADMIN] Ingestion triggered for path: {documents_path}")

    result = ingestion_service.ingest_all(documents_path)

    if not result["success"] and result["processed_files"] == 0 and result["failed_files"] > 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )

    logger.info(f"[ADMIN] Ingestion complete: {result.get('processed_files', 0)} files, {result.get('total_chunks', 0)} chunks")
    return IngestResponse(
        success=result["success"],
        message=result["message"],
        total_chunks=result.get("total_chunks", 0),
        processed_files=result.get("processed_files", 0),
        failed_files=result.get("failed_files", 0)
    )


@router.post("/curated-qa", response_model=CuratedQaResponse, dependencies=[Depends(verify_admin_key)])
async def add_curated_qa(
    qa_request: CuratedQaRequest,
    db: AsyncSession = Depends(get_db),
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Add a curated Q&A pair to the knowledge base for training and retrieval."""
    new_qa = CuratedQA(
        question=qa_request.question,
        answer=qa_request.answer,
        source_feedback_id=qa_request.source_feedback_id,
        active=True
    )

    db.add(new_qa)
    await db.commit()
    await db.refresh(new_qa)
    logger.info(f"[ADMIN] Curated QA saved to DB (ID: {new_qa.id})")

    try:
        from app.core.dependencies import get_vector_store_service
        vs_service = get_vector_store_service()
        chunk = {
            "text": f"Question: {qa_request.question}\nAnswer: {qa_request.answer}",
            "header_path": "Curated Knowledge",
            "source": "User Feedback",
            "chunk_index": 0
        }
        success = vs_service.add_documents([chunk])
        if success:
            logger.info(f"[ADMIN] Curated QA indexed in vector store (ID: {new_qa.id})")
        else:
            logger.warning(f"[ADMIN] Failed to index curated QA in vector store (ID: {new_qa.id})")
    except Exception as e:
        logger.error(f"[ADMIN] Error indexing curated QA: {e}")

    return new_qa


@router.post("/train", dependencies=[Depends(verify_admin_key)])
async def trigger_training():
    """Trigger the DSPy optimization pipeline using curated Q&A."""
    logger.info("[ADMIN] Training pipeline triggered")
    try:
        result = await run_training_pipeline()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

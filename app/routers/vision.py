from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import base64
import os

from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.core.config import ENABLE_MCL_IMAGE_VALIDATION
from app.core.logging import get_logger
from app.core.dependencies import get_vision_service, get_image_validator_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])

@router.post("/analyze-screenshot")
async def analyze_screenshot(
    file: UploadFile = File(...),
    query: str = Form(...),
    vision_service: VisionService = Depends(get_vision_service),
    image_validator: ImageValidatorService = Depends(get_image_validator_service)
):
    """
    Analyze an MCL App screenshot and provide contextual help using GPT-4o vision.
    """
    logger.info(f"Received screenshot analysis request. File: {file.filename}, Query: {query}")

    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {', '.join(allowed_types)}"
        )

    try:
        # Read file content
        contents = await file.read()
        
        # Encode to base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Add data URI prefix if needed by the service (VisionService handles raw base64 now, but let's be safe)
        # Actually VisionService.analyze_image_base64 handles the prefix logic.
        
        # Validate Image if enabled
        if ENABLE_MCL_IMAGE_VALIDATION:
            logger.info("Validating image...")
            validation_result = image_validator.validate_image(base64_image)
            if not validation_result["is_mcl"]:
                logger.info(f"Image validation failed: {validation_result['suggestion']}")
                return {
                    "response": validation_result["suggestion"],
                    "success": False,
                    "metadata": {
                        "validation": validation_result
                    }
                }

        # Analyze Image
        logger.info("Analyzing image with VisionService...")
        response_text = vision_service.analyze_image_base64(base64_image, query)
        logger.info("Image analysis completed successfully")
        
        return {
            "response": response_text,
            "success": True,
            "metadata": {
                "image_name": file.filename,
                "query": query
            }
        }

    except Exception as e:
        logger.error(f"Error analyzing screenshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

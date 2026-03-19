from functools import lru_cache
from fastapi import Depends
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.services.language_service import LanguageService
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.core.config import VECTOR_STORE_PATH

# Singleton instance for VectorStoreService
_vector_store_service_instance = None

def get_vector_store_service() -> VectorStoreService:
    global _vector_store_service_instance
    if _vector_store_service_instance is None:
        _vector_store_service_instance = VectorStoreService()
    return _vector_store_service_instance

@lru_cache()
def get_vision_service() -> VisionService:
    return VisionService()

@lru_cache()
def get_language_service() -> LanguageService:
    return LanguageService()

def get_image_validator_service(
    vision_service: VisionService = Depends(get_vision_service)
) -> ImageValidatorService:
    return ImageValidatorService(vision_service)

_chat_service_instance = None

def get_chat_service(
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    vision_service: VisionService = Depends(get_vision_service),
    image_validator: ImageValidatorService = Depends(get_image_validator_service),
    language_service: LanguageService = Depends(get_language_service)
) -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService(vector_store, vision_service, image_validator, language_service)
    return _chat_service_instance


def get_ingestion_service(
    vector_store: VectorStoreService = Depends(get_vector_store_service)
) -> IngestionService:
    return IngestionService(vector_store)

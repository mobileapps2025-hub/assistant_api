from functools import lru_cache
from fastapi import Depends
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.services.chat_service import ChatService
from app.services.speech_service import SpeechService


@lru_cache()
def get_vision_service() -> VisionService:
    return VisionService()


def get_image_validator_service(
    vision_service: VisionService = Depends(get_vision_service)
) -> ImageValidatorService:
    return ImageValidatorService(vision_service)


_chat_service_instance = None


def get_chat_service(
    vision_service: VisionService = Depends(get_vision_service),
    image_validator: ImageValidatorService = Depends(get_image_validator_service),
) -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService(vision_service, image_validator)
    return _chat_service_instance


@lru_cache()
def get_speech_service() -> SpeechService:
    return SpeechService()

import io
import asyncio
import logging
from openai import OpenAI, OpenAIError
from app.core.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB


class SpeechService:

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = "whisper-1"

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> str:
        if not audio_bytes:
            raise ValueError("Empty audio data")

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError(f"Audio file too large: {len(audio_bytes)} bytes (max {MAX_AUDIO_BYTES})")

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        try:
            transcript = await asyncio.to_thread(
                self._create_transcription,
                audio_file
            )
            text = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()
            logger.info(f"[SPEECH] Transcription successful: {len(text)} chars")
            return text
        except OpenAIError as e:
            logger.error(f"[SPEECH] OpenAI transcription failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[SPEECH] Unexpected transcription error: {e}")
            raise

    def _create_transcription(self, audio_file: io.BytesIO) -> str:
        return self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            response_format="text",
            language="en"
        )
